# ImageTo3DSegmentedClothes

> This repo is one step towards a single 2D image to 3D garment extraction pipeline.

---

## 📚 Overview

- This repository leverages the implementation of [CloSe](https://github.com/KarimIbrahim11/CloSe), a project that focuses on 3D Segmentation of Clothes, 
[Human3Diffusion](https://github.com/KarimIbrahim11/Human3Diffusion), for 3DGaussian Splatting of human figures, [smplx](https://github.com/KarimIbrahim11/smplx) for SMPL human body models and [NICP](https://github.com/KarimIbrahim11/NICP) for iterative SMPL registration. 
---

## 📊 Results:
### Input:
![2DImage](assets/input.png)

### Mesh 3D:
![3DMesh](assets/mesh_spinning_rotated.gif)

### 3D Segmented Clothes:
![3DClothes](assets/segmented_spinning_rotated.gif)

## 🧩 Dependencies

This project depends on the following repositories (included as Git submodules):

- [CloSe (forked)](https://github.com/KarimIbrahim11/CloSe/tree/feature/image2garment) — 3D Segmentation Model
- [Human3Diffusion (forked)](https://github.com/KarimIbrahim11/Human3Diffusion/tree/feature/image2garment) — 3D Gaussians for Human Bodies
- [NICP (forked)](https://github.com/KarimIbrahim11/NICP) — SMPLH fitting
- [smplx](https://github.com/KarimIbrahim11/smplx) — SMPL models
  
To initialize submodules:

```bash
git submodule update --init --recursive
```

To install the dependencies:
```bash
conda env create -f environment.yaml
```

## 🧩 Inference

```bash
./scripts/infer.sh
```

