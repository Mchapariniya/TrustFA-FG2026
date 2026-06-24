# Micro-Expression-Aware Avatar Fingerprinting via Inter-Frame Feature Differencing

Official code for the IEEE FG 2026 paper.

**Authors:** Masoumeh Chapariniya¹², Jean-Marc Odobez³⁴, Volker Dellwo¹, Teodora Vuković¹²

¹ Department of Computational Linguistics, University of Zurich · ² UZH Digital Society Initiative ·
³ Idiap Research Institute, Martigny · ⁴ École Polytechnique Fédérale de Lausanne (EPFL)

---

## Abstract

Avatar fingerprinting verifies *who drives* a synthetic talking-head video rather than whether it is
real. We propose a preprocessing-free framework that combines a micro-expression-aware **F5C backbone**
with **inter-frame feature differencing**: consecutive deep feature maps are subtracted so that
temporally stable appearance cancels by construction while driver-specific micro-motion is preserved.
Trained end-to-end from raw grayscale frames with a supervised contrastive objective, our 0.53M-parameter
model reaches **0.877 overall AUC on NVFAIR**, matching or exceeding the landmark-based baseline on most
cross-generator pairs while eliminating any external landmark tracking.

## Method at a glance

`T=64` grayscale frames → shared **F5C** backbone (ConvStack → FCC → CCC) → per-frame feature maps
`f_t ∈ R^{128×16×16}` → **inter-frame differencing** `d_t = f_{t+1} − f_t` → temporal identity head →
ℓ₂-normalized 256-d embedding, trained with supervised contrastive loss (τ = 0.07).

## Repository structure

```
.
├── src/
│   └── feat_diff.py                  # F5C + FEAT_DIFF model, training & evaluation (main entry point)
├── scripts/
│   ├── split_dataset.py              # build the identity-disjoint NVFAIR split (112/14/35)
│   ├── validate_splits.py            # sanity-check the split for identity leakage
│   ├── ablation_representation.py    # FEAT_DIFF/PIXEL_DIFF/RAW_FEAT/STATIC (Table VI)
│   ├── ablation_backbone_resnet18.py # ResNet18 backbone ablation (Table VIII)
│   ├── generalization_pixel_diff.py  # PIXEL_DIFF cross-generator matrix (Table V)
│   ├── generate_paper_figures.py     # render result figures from saved runs
│   └── run_clip_length_ablation.sh   # clip-length sweep T∈{16,32,64,128} (Table VII)
├── configs/
│   └── splits/                       # identity-level split definition + summary (manifests are generated)
├── docs/figures/                     # figures used in this README
├── requirements.txt
└── .gitignore
```

## Installation

```bash
git clone https://github.com/Mchapariniya/TrustFA-FG2026.git
cd TrustFA-FG2026
python -m venv .venv && source .venv/bin/activate   # Python 3.10 recommended
pip install -r requirements.txt
```

## Dataset setup (NVFAIR)

This work uses the **NVFAIR** avatar-fingerprinting benchmark (Prashnani et al., ECCV 2024) — 650k+
synthetic talking-head videos from 161 identities across three generators (Face-vid2vid, TPS, LIA).
Obtain it from the original authors; this repository does **not** redistribute the data.

1. Decode each video to grayscale frames, preserving the directory tree, under `nvfair_frames/`.
2. Build the identity-disjoint split (112 train / 14 val / 35 test). The split is **deterministic** —
   it is defined entirely by the identity-level files in [`configs/splits/`](configs/splits/)
   (`train_val_test_splits.txt`, `subject_sources.txt`), reproduced from the NVFAIR metadata. Edit the
   paths at the top of `scripts/split_dataset.py` to point at your NVFAIR copy, then generate the
   per-file manifests (these are large and are **not** committed):

   ```bash
   python scripts/split_dataset.py        # writes NVFAIR_split/{train,val,test}_files.csv
   python scripts/validate_splits.py      # verify no identity crosses splits
   ```

The training/eval scripts expect:
`--nvfair_split_root NVFAIR_split` (the generated `*_files.csv` manifests) and
`--frames_root nvfair_frames` (the decoded frames).

## Training

Train the main FEAT_DIFF model (T = 64, the 0.877-AUC configuration):

```bash
python src/feat_diff.py \
  --nvfair_split_root NVFAIR_split \
  --frames_root nvfair_frames \
  --clip_length 64 \
  --num_ids 16 --vids_per_id 8 \
  --num_epochs 150 --steps_per_epoch 200 \
  --lr 1e-3 --wd 1e-4 --temperature 0.07 \
  --output_dir outputs/feat_diff --results_dir results/feat_diff
```

## Evaluation

Evaluate a trained checkpoint with the paper-exact AUC protocol (mean AUC over the 35 test identities):

```bash
python src/feat_diff.py \
  --nvfair_split_root NVFAIR_split \
  --frames_root nvfair_frames \
  --clip_length 64 --eval_only \
  --output_dir outputs/feat_diff --results_dir results/feat_diff
```

## Reproducing the ablations

```bash
# Representation ablation: FEAT_DIFF vs PIXEL_DIFF vs RAW_FEAT vs STATIC  (Table VI)
python scripts/ablation_representation.py \
  --nvfair_split_root NVFAIR_split --frames_root nvfair_frames --clip_length 64

# Clip-length ablation, T ∈ {16, 32, 64, 128}  (Table VII)
bash scripts/run_clip_length_ablation.sh 0 1 2

# ResNet18 backbone ablation  (Table VIII)
python scripts/ablation_backbone_resnet18.py --variant feat_diff \
  --nvfair_split_root NVFAIR_split --frames_root nvfair_frames

# PIXEL_DIFF cross-generator matrix  (Table V)
python scripts/generalization_pixel_diff.py \
  --nvfair_split_root NVFAIR_split --frames_root nvfair_frames
```

## Results

Per-generator AUC on NVFAIR over the 35 test identities (Table III):

| Method                | Overall | FV2V  | LIA   | TPS   | Params |
|-----------------------|:-------:|:-----:|:-----:|:-----:|:------:|
| NVFAIR (landmark) †   | 0.853   | 0.870 | 0.840 | 0.850 | —      |
| PIXEL_DIFF (ours)     | 0.861   | 0.849 | 0.866 | 0.868 | 0.53M  |
| **FEAT_DIFF (ours)**  | **0.877** | 0.869 | 0.882 | 0.880 | **0.53M** |

† Landmark-based baseline reported in Prashnani et al. (2024).

Clip length improves performance monotonically (0.813 → 0.891 for T = 16 → 128); we use T = 64 as the
accuracy/cost trade-off. Replacing F5C with ResNet18 collapses feature-space differencing far below F5C
despite ~22× more parameters (11.6M vs. 0.53M), confirming both the backbone and the differencing
principle are essential.

## Citation

```bibtex
@inproceedings{chapariniya2026microexpression,
  title     = {Micro-Expression-Aware Avatar Fingerprinting via Inter-Frame Feature Differencing},
  author    = {Chapariniya, Masoumeh and Odobez, Jean-Marc and Dellwo, Volker and Vukovi\'{c}, Teodora},
  booktitle = {2026 IEEE 20th International Conference on Automatic Face and Gesture Recognition (FG)},
  year      = {2026},
  doi       = {10.1109/FG67764.2026.11556973}
}
```

## Acknowledgements

Built on the NVFAIR benchmark (Prashnani et al., ECCV 2024); the F5C backbone follows the
micro-expression model MOL (Shao et al., TPAMI 2025); training uses supervised contrastive learning
(Khosla et al., NeurIPS 2020).
