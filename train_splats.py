import os
import cv2
import numpy as np
import torch
import torch.optim as optim

# ── device ────────────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

from render_splats import (
    read_cameras, read_images, read_points3D,
    qvec2rotmat, get_intrinsics,
)

os.makedirs("outputs", exist_ok=True)

# ── load COLMAP data ───────────────────────────────────────────────────────────
cams = read_cameras("colmap/text/cameras.txt")
imgs = read_images("colmap/text/images.txt")
pts  = read_points3D("colmap/text/points3D.txt")

pts = pts[::4]          # subsample for memory
print(f"Using {len(pts)} splats")

xyz_np   = np.array([p[0] for p in pts], dtype=np.float32)
color_np = np.array([p[1] for p in pts], dtype=np.float32)

xyz   = torch.from_numpy(xyz_np).to(device)
color = torch.from_numpy(color_np).to(device)

# Convert to logit space so sigmoid(color_raw) ≈ original colour
color_raw = torch.logit(color.clamp(1e-6, 1 - 1e-6)).detach().requires_grad_(True)
xyz.requires_grad_(True)

N = xyz.shape[0]

# ── learnable splat parameters ────────────────────────────────────────────────
# FIX #6: initialise scales small so splats start as point-like (a few pixels)
log_sx  = torch.full((N,), -3.0, device=device, requires_grad=True)
log_sy  = torch.full((N,), -3.0, device=device, requires_grad=True)
log_sz  = torch.full((N,), -3.0, device=device, requires_grad=True)
opacity = torch.full((N,),  0.0, device=device, requires_grad=True)  # sigmoid(0)=0.5

# ── pick first camera & image ─────────────────────────────────────────────────
img_id              = list(imgs.keys())[0]
# FIX #2: direct unpack – img_data is (qvec, tvec, cam_id, name)
qvec, tvec, cam_id, name = imgs[img_id]

R_np = qvec2rotmat(qvec)                              # world→cam rotation
R    = torch.tensor(R_np, dtype=torch.float32, device=device)
t    = torch.tensor(tvec,  dtype=torch.float32, device=device)

# FIX #1: use get_intrinsics() which handles any COLMAP camera model
cam_tuple              = cams[cam_id]
fx_orig, fy_orig, cx_orig, cy_orig = get_intrinsics(cam_tuple)
_, colmap_W, colmap_H, _          = cam_tuple   # (model, w, h, params)
# colmap_W / colmap_H are the pixel dimensions the focal length was calibrated at

# ── load & resize target image ────────────────────────────────────────────────
img_path = f"data/clean/{name}"
img_bgr  = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Target image not found: {img_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

MAX_DIM = 512
h0, w0  = img_rgb.shape[:2]
scale   = MAX_DIM / max(h0, w0)
if scale < 1.0:
    img_rgb = cv2.resize(img_rgb, (int(w0 * scale), int(h0 * scale)),
                         interpolation=cv2.INTER_AREA)

target = torch.tensor(img_rgb, dtype=torch.float32, device=device)
H, W   = target.shape[:2]

# Scale intrinsics from COLMAP full-res → render resolution
sx_intr = W / colmap_W
sy_intr = H / colmap_H
fx = fx_orig * sx_intr
fy = fy_orig * sy_intr
cx = cx_orig * sx_intr
cy = cy_orig * sy_intr

print(f"Render size: {W}×{H}  |  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

# pixel grid (H×W×2)
ys, xs = torch.meshgrid(
    torch.arange(H, device=device, dtype=torch.float32),
    torch.arange(W, device=device, dtype=torch.float32),
    indexing="ij",
)
pixels = torch.stack([xs, ys], dim=-1)   # (H, W, 2)

# ── render function ────────────────────────────────────────────────────────────
def render():
    sx, sy, sz = torch.exp(log_sx), torch.exp(log_sy), torch.exp(log_sz)
    alpha = torch.sigmoid(opacity)          # (N,)
    c     = torch.sigmoid(color_raw)        # (N, 3)  – FIX #5: one sigmoid only

    # Project points to camera space:  p_cam = R @ p_world + t
    xyz_cam = xyz @ R.T + t                 # (N, 3)
    xc, yc, zc = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]

    valid = zc > 0.1
    if not valid.any():
        return torch.zeros((H, W, 3), device=device)

    xv, yv, zv   = xc[valid], yc[valid], zc[valid]
    sxv, syv, szv = sx[valid], sy[valid], sz[valid]
    av, cv_      = alpha[valid], c[valid]

    # Perspective projection → screen coords
    u = xv * fx / zv + cx
    v = yv * fy / zv + cy

    # ── 2-D covariance via EWA splatting ──────────────────────────────────────
    # 3-D diagonal covariance (axis-aligned Gaussians in world space)
    # Σ_3D = diag(sx², sy², sz²)
    # Σ_2D = J · R · Σ_3D · Rᵀ · Jᵀ    where J is the projective Jacobian
    Nv = xv.shape[0]

    # Jacobian of perspective projection (2×3 per splat)
    J = torch.zeros((Nv, 2, 3), device=device)
    J[:, 0, 0] =  fx / zv
    J[:, 0, 2] = -fx * xv / (zv * zv)
    J[:, 1, 1] =  fy / zv
    J[:, 1, 2] = -fy * yv / (zv * zv)

    # 3-D covariance  (diagonal → only need diagonal elements for efficiency)
    # Full matrix form: Σ_3D = diag(sxv², syv², szv²)
    cov3D = torch.zeros((Nv, 3, 3), device=device)
    cov3D[:, 0, 0] = sxv * sxv
    cov3D[:, 1, 1] = syv * syv
    cov3D[:, 2, 2] = szv * szv

    # W = R  (camera rotation, maps world → cam; used in EWA formula)
    W_mat = R.unsqueeze(0).expand(Nv, 3, 3)            # (Nv, 3, 3)
    Sigma2 = J @ (W_mat @ cov3D @ W_mat.transpose(1, 2)) @ J.transpose(1, 2)
    # Add a small isotropic regulariser for numerical stability
    Sigma2 = Sigma2 + torch.eye(2, device=device).unsqueeze(0) * 0.3

    # Inverse of 2×2 cov
    det  = Sigma2[:, 0, 0] * Sigma2[:, 1, 1] - Sigma2[:, 0, 1] ** 2
    det  = det.clamp(min=1e-8)
    inv  = torch.zeros_like(Sigma2)
    inv[:, 0, 0] =  Sigma2[:, 1, 1] / det
    inv[:, 1, 1] =  Sigma2[:, 0, 0] / det
    inv[:, 0, 1] = -Sigma2[:, 0, 1] / det
    inv[:, 1, 0] = -Sigma2[:, 1, 0] / det

    # Splat radius for tile culling (3σ along largest axis)
    rad = 3.0 * torch.sqrt(torch.maximum(Sigma2[:, 0, 0], Sigma2[:, 1, 1]))  # (Nv,)

    # ── tile rasteriser ────────────────────────────────────────────────────────
    canvas = torch.zeros((H, W, 3), device=device)
    TILE   = 64

    for y0 in range(0, H, TILE):
        y1 = min(H, y0 + TILE)
        for x0 in range(0, W, TILE):
            x1 = min(W, x0 + TILE)

            # Cull: keep only splats that overlap this tile
            mask = (u + rad > x0) & (u - rad < x1) & \
                   (v + rad > y0) & (v - rad < y1)
            if not mask.any():
                continue

            # FIX #3: sort ASCENDING by z for front-to-back compositing
            tz   = zv[mask]
            idx  = torch.argsort(tz, descending=False)

            mu_t    = u[mask][idx]          # (K,)
            mv_t    = v[mask][idx]
            inv_t   = inv[mask][idx]        # (K, 2, 2)
            a_t     = av[mask][idx]         # (K,)
            c_t     = cv_[mask][idx]        # (K, 3)

            # Pixel coords for this tile  (Ph*Pw, 2)
            px = pixels[y0:y1, x0:x1].reshape(-1, 2)   # (P, 2)

            # Vectorised Gaussian evaluation
            # dx/dy: (P, K)
            dx = px[:, 0:1] - mu_t.unsqueeze(0)
            dy = px[:, 1:2] - mv_t.unsqueeze(0)

            # Mahalanobis: -0.5 * [dx dy] · Σ⁻¹ · [dx; dy]
            maha = (dx * dx * inv_t[:, 0, 0].unsqueeze(0) +
                    dy * dy * inv_t[:, 1, 1].unsqueeze(0) +
                    2.0 * dx * dy * inv_t[:, 0, 1].unsqueeze(0))   # (P, K)
            gauss = torch.exp(-0.5 * maha) * a_t.unsqueeze(0)     # (P, K)

            # FIX #7: vectorised front-to-back alpha compositing
            # transmittance T[p, k] = prod_{j<k} (1 - gauss[p,j])
            # colour contribution = T[p,k] * gauss[p,k] * c[k]
            # We compute T cumulatively with a scan then sum over k.
            one_minus_g = 1.0 - gauss                              # (P, K)
            # cumprod gives T_after; T_before = shift right by 1 (first = 1)
            T_after     = torch.cumprod(one_minus_g, dim=1)        # (P, K)
            # prepend ones: T_before[:, k] = product of (1-g) for j<k
            ones_col    = torch.ones((px.shape[0], 1), device=device)
            T_before    = torch.cat([ones_col, T_after[:, :-1]], dim=1)  # (P, K)

            weight = T_before * gauss                              # (P, K)
            tile_color = weight @ c_t                              # (P, 3)

            canvas[y0:y1, x0:x1] = tile_color.reshape(y1 - y0, x1 - x0, 3)

    return torch.clamp(canvas, 0.0, 1.0)


# ── optimiser ──────────────────────────────────────────────────────────────────
optimizer = optim.Adam([
    {"params": color_raw, "lr": 5e-3},
    {"params": xyz,       "lr": 3e-4},
    {"params": log_sx,    "lr": 4e-3},
    {"params": log_sy,    "lr": 4e-3},
    {"params": log_sz,    "lr": 4e-3},
    {"params": opacity,   "lr": 4e-3},
])

# ── training loop ──────────────────────────────────────────────────────────────
def train(num_steps=500):
    print("\n🚀 TRAINING...")
    for step in range(num_steps):
        optimizer.zero_grad()
        pred = render()
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        optimizer.step()

        print(f"STEP {step:4d} | Loss: {loss.item():.6f}")

        if step % 25 == 0:
            out = (pred.detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                f"outputs/step_{step:04d}.jpg",
                cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
            )

    # save checkpoint
    torch.save({
        "xyz":       xyz.detach(),
        "color_raw": color_raw.detach(),
        "log_sx":    log_sx.detach(),
        "log_sy":    log_sy.detach(),
        "log_sz":    log_sz.detach(),
        "opacity":   opacity.detach(),
    }, "outputs/checkpoint.pt")

    print("Checkpoint saved → outputs/checkpoint.pt")

if __name__ == "__main__":
    train()