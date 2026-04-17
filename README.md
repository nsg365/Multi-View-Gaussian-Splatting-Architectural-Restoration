# Multi-View-Gaussian-Splatting-Architectural-Restoration

This project implements **2D Gaussian Splatting** for 3D scene reconstruction and applies it to the **restoration of damaged architectural structures**.

## Dataset Used
We used an aerial photogrammetry dataset of the Chichen Itza Pyramid in Mexico for the project. The dataset is linked here: https://openheritage3d.org/project.php?id=qx4z-zw93

## Project Structure

```
Multi-View-Gaussian-Splatting-Architectural-Restoration/
│
├── src/
│   ├── train_splats.py
│   ├── render_splats.py
│   ├── reconstruct_stairs.py
│   ├── ply_helper.py
│   ├── inspect_scene.py
│   └── visualise.py
│
├── data/
│   ├── images/
│   └── clean/
│
├── colmap/
│
├── outputs/
│
├── requirements.txt
│
└── README.md
```

## How to Run

### 1. Install COLMAP

COLMAP is required to extract camera parameters and sparse 3D geometry.

**Mac (Homebrew):**
```bash
brew install colmap
colmap --help
```

### 2. Run COLMAP Pipeline
```bash
    colmap feature_extractor \
        --database_path colmap/database.db \
        --image_path data/images

    colmap exhaustive_matcher \
        --database_path colmap/database.db

    colmap mapper \
        --database_path colmap/database.db \
        --image_path data/images \
        --output_path colmap/sparse

    colmap model_converter \
        --input_path colmap/sparse/0 \
        --output_path colmap/text \
        --output_type TXT
```
the model_converter converts COLMAP's binary output into text format required by the training pipeline.
The colmap model can be visualised via the colmap gui. The visualisations are saved in the assets folder as Colmap_Side_view.png and Colmap_Top_View.png

### 3. Install Python Dependencies
```bash
    pip install -r requirements.txt
```

### 4. Run Training
```bash
    python3 src/train_splats.py
```
This generates a Gaussian Splat Representation of the scene. It runs over 500 iterations, and logs progress at every 25 interations as intermediate renders. The last iteration has been included in the assets folder as Gaussian_render.jpg

### 5. Introduce Structural Damage
Next, we introduce structural damage in the monument manually, and identify a matching section in another part of the monument that acts as a donor Region.

```bash
    python3 src/inspect_scene.py
```
On the first pass, Shift Click on 4 points that bound the damaged region. Click q to exit. On the second pass, Shift Click on 4 points that bound the donor region. Click q to exit.

The gaussians in the damaged region are removed, leaving a mask over that region. The masked image is saved in the assets folder as masked.jpg

### 6. Provide a Reconstruction for the Damaged Region
```bash
    python3 src/reconstruct_stairs.py
```

Finally, run the above reconstruction code. This transforms and fits donor-region Gaussians into the masked region to reconstruct the missing structure to complete the reconstruction of the image. The final reconstructed image is saved in the assets folder as restored_demo.jpg

## Helper Files
The files
```
ply_helper.py
visualise.py
```
are helper files used to visualise .ply files and .pt files respectively.

