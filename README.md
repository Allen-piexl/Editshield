# EditShield
EditShield: Protecting Unauthorized Image Editing by Instruction-guided Diffusion Models.
[Project Page](https://github.com/Allen-piexl/Editshield/) | [Paper](https://arxiv.org/pdf/2311.12066)

## TL;DR: quickstart 

### Step 1. Set up a conda environment, and download a pre-trained model:
```
conda env create -f environment.yaml
conda activate Editshield
bash scripts/download_checkpoints.sh
```

### Step 2. Create a perturbation using the EditShield with Expectation Over Transformation (EOT), we use center cropping as an example:
```
python EOT_Center.py
```

We also provide codes for Gaussian kernel smoothing and Image rotation.

### Step 3. Generate descriptions of the caption using LLaVA:
```
python Cap_EOT_C.py
```

### Step 4. Calculate the CLIP image similarity and CLIP text-image direction similarity:
```
python --number 5 --method EOT-C Caption_Metrics.py
```

## Reference

```bibtex
@article{chen2023editshield,
  title={EditShield: Protecting Unauthorized Image Editing by Instruction-guided Diffusion Models},
  author={Chen, Ruoxi and Jin, Haibo and Chen, Jinyin and Sun, Lichao},
  journal={arXiv preprint arXiv:2311.12066},
  year={2023}
}
