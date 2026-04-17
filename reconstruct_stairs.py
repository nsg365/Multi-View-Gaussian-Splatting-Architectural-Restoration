import torch
import numpy as np
from typing import Tuple

BBox = Tuple[float, float, float, float, float, float]


def _bbox_mask(xyz, box):
    x_min, x_max, y_min, y_max, z_min, z_max = box
    return (
        (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
        (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) &
        (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    )


# ─────────────────────────────────────────
# STEP 1 — MASK
# ─────────────────────────────────────────
def mask_region(xyz, color_raw, log_sx, log_sy, log_sz, opacity, damaged_box):
    inside = _bbox_mask(xyz, damaged_box)
    keep = ~inside

    print(f"[mask] removed {inside.sum().item()} splats")

    def f(t): return t[keep].detach()

    return f(xyz), f(color_raw), f(log_sx), f(log_sy), f(log_sz), f(opacity)


# ─────────────────────────────────────────
# STEP 2 — DEMO DONOR
# ─────────────────────────────────────────
def get_demo_donor(xyz, color_raw, log_sx, log_sy, log_sz, opacity):

    center_mask = (
        (xyz[:, 1] > 1.7) & (xyz[:, 1] < 2.4) &
        (xyz[:, 0] > -1.2) & (xyz[:, 0] < 1.2)
    )

    print(f"[demo donor] selected {center_mask.sum().item()} splats")

    return (
        xyz[center_mask].clone(),
        color_raw[center_mask].clone(),
        log_sx[center_mask].clone(),
        log_sy[center_mask].clone(),
        log_sz[center_mask].clone(),
        opacity[center_mask].clone()
    )


# ─────────────────────────────────────────
# STEP 3 — TRANSFORM
# ─────────────────────────────────────────
def transform_demo(d_xyz, damaged_box, device):

    mid = (damaged_box[0] + damaged_box[1]) / 2
    d_xyz[:, 0] = 2 * mid - d_xyz[:, 0]

    damaged_center = torch.tensor([
        (damaged_box[0] + damaged_box[1]) / 2,
        (damaged_box[2] + damaged_box[3]) / 2,
        (damaged_box[4] + damaged_box[5]) / 2,
    ], device=device)

    donor_center = torch.tensor([
        (d_xyz[:, 0].min() + d_xyz[:, 0].max()) / 2,
        (d_xyz[:, 1].min() + d_xyz[:, 1].max()) / 2,
        (d_xyz[:, 2].min() + d_xyz[:, 2].max()) / 2,
    ], device=device)

    shift = damaged_center - donor_center
    shift[2] = 0.0  # preserve depth

    d_xyz = d_xyz + shift

    print("[transform] shift:", shift.cpu().numpy())

    return d_xyz


# ─────────────────────────────────────────
# STEP 4 — RECONSTRUCT
# ─────────────────────────────────────────
def reconstruct_demo(
    xyz, color_raw, log_sx, log_sy, log_sz, opacity,
    damaged_box
):
    device = xyz.device

    # MASK
    m = mask_region(xyz, color_raw, log_sx, log_sy, log_sz, opacity, damaged_box)

    # DONOR
    d = get_demo_donor(xyz, color_raw, log_sx, log_sy, log_sz, opacity)
    d_xyz, d_color, d_lsx, d_lsy, d_lsz, d_op = d

    # TRANSFORM
    d_xyz = transform_demo(d_xyz, damaged_box, device)

    # SCALE (anisotropic splat size)
    d_lsx += 0.1
    d_lsy += 0.05
    d_lsz += 0.15

    # CLAMP to damaged region
    mask = _bbox_mask(d_xyz, damaged_box)

    d_xyz = d_xyz[mask]
    d_color = d_color[mask]
    d_lsx = d_lsx[mask]
    d_lsy = d_lsy[mask]
    d_lsz = d_lsz[mask]
    d_op = d_op[mask]

    # soften opacity
    d_op = d_op - 0.5

    # density boost
    noise = 0.01 * torch.randn_like(d_xyz)
    d_xyz = torch.cat([d_xyz, d_xyz + noise], dim=0)
    d_color = torch.cat([d_color, d_color], dim=0)
    d_lsx = torch.cat([d_lsx, d_lsx], dim=0)
    d_lsy = torch.cat([d_lsy, d_lsy], dim=0)
    d_lsz = torch.cat([d_lsz, d_lsz], dim=0)
    d_op = torch.cat([d_op, d_op], dim=0)

    print("[enhance] density boosted")

    # MERGE
    full = [torch.cat([m[i], d[i]], dim=0) for i in range(6)]

    original_count = m[0].shape[0]

    print("Reconstruction complete")

    return tuple(full), original_count


# ─────────────────────────────────────────
# REFINEMENT (FIXED)
# ─────────────────────────────────────────
def refine_scene(_ts, target, original_count, steps=80):

    print(f"\nRefining;")

    params = [
        _ts.xyz,
        _ts.color_raw,
    ]

    for i in range(steps):

        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        pred = _ts.render()
        loss = ((pred - target) ** 2).mean()
        loss.backward()

        # 🔥 freeze old splats
        for p in params:
            if p.grad is not None:
                p.grad[:original_count] = 0

        # manual update
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p -= 5e-2 * p.grad

        if i % 5 == 0:
            print(f"step {i} | loss {loss.item():.6f}")

    print("✅ Selective refinement done\n")


# ─────────────────────────────────────────
# SAVE PLY
# ─────────────────────────────────────────
def save_ply(path, xyz, color_raw, opacity):
    import struct

    pts = xyz.detach().cpu().numpy()
    col = torch.sigmoid(color_raw).detach().cpu().numpy()
    alp = torch.sigmoid(opacity).detach().cpu().numpy()

    with open(path, "wb") as f:
        f.write(f"""ply
format binary_little_endian 1.0
element vertex {len(pts)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
""".encode())

        for i in range(len(pts)):
            r, g, b = (col[i] * 255).astype(np.uint8)
            a = int(alp[i] * 255)
            f.write(struct.pack("<fffBBBB", *pts[i], r, g, b, a))


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = torch.load("outputs/checkpoint.pt", map_location=device)

    xyz = ckpt["xyz"]
    color_raw = ckpt["color_raw"]
    log_sx = ckpt["log_sx"]
    log_sy = ckpt["log_sy"]
    log_sz = ckpt["log_sz"]
    opacity = ckpt["opacity"]

    import train_splats as _ts

    DAMAGED_BOX = (-0.65, 0.02, 1.78, 2.20, -1.10, -0.82)

    # reconstruct
    (new_xyz, new_color_raw, new_log_sx,
     new_log_sy, new_log_sz, new_opacity), original_count = reconstruct_demo(
        xyz, color_raw, log_sx, log_sy, log_sz, opacity,
        DAMAGED_BOX
    )

    # load into renderer
    _ts.xyz = new_xyz.to(device).requires_grad_(True)
    _ts.color_raw = new_color_raw.to(device).requires_grad_(True)
    _ts.log_sx = new_log_sx.to(device).requires_grad_(True)
    _ts.log_sy = new_log_sy.to(device).requires_grad_(True)
    _ts.log_sz = new_log_sz.to(device).requires_grad_(True)
    _ts.opacity = new_opacity.to(device).requires_grad_(True)

    # refinement
    if hasattr(_ts, "target"):
        refine_scene(_ts, _ts.target, original_count, steps=80)

    # save ply
    save_ply("outputs/restored_demo.ply", _ts.xyz, _ts.color_raw, _ts.opacity)

    # render final
    import cv2
    final = _ts.render()
    out = (final.detach().cpu().numpy() * 255).astype(np.uint8)

    cv2.imwrite("outputs/restored_demo.jpg",
                cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    print("Saved outputs/restored_demo.jpg")