# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

---

## Project Overview

**Hierarchical Concept Bottleneck Model (H-CBM)** for explainable bird species classification using the CUB-200-2011 dataset.

The model predicts bird species (200 classes) through a **two-level concept hierarchy**:

- **Level 1 (L1) — Coarse Concepts:** ~13 body-part groups derived by parsing CUB attribute names (`bill`, `wing`, `tail`, `breast`, `belly`, `leg`, `eye`, `throat`, `forehead`, `nape`, `crown`, `back`, `head`)
- **Level 2 (L2) — Fine Concepts:** 312 binary attributes from CUB-200-2011 (`has_bill_shape::dagger`, `has_wing_color::brown`, ...)
- **Classifier:** 200 bird species — reads **only from L2**, not from backbone features

---

## Architecture

```
Input Image (3, 224, 224)
        │
        ▼
ResNet-50 (pretrained ImageNet, FC layer removed)
        │
   Feature Map F (2048,)
        │
        ├──────────────────────────────┐
        ▼                              │
Coarse Head g_c(F)                    │ F
Linear(2048→512)→ReLU→Dropout(0.3)   │
→Linear(512→13)                       │
        │                              │
   z_c (13 logits)                    │
   p_c = σ(z_c)                       │
        │                              ▼
        │              Fine Head g_f(F, p_c)
        │              concat([F, p_c]) → [2061]
        │              Linear(2061→512)→ReLU→Dropout(0.3)
        │              →Linear(512→312)
        │                      │
        │                 z_f (312 logits)
        │                 p_f_raw = σ(z_f)
        │                 mask[i] = p_c[parent(i)]   ← Masked Fine Head
        │                 p_f = p_f_raw × mask        ← hard constraint
        │                      │
        └──────────────────────┘
                               │ p_f (312,)
                               ▼
                      Classifier h(p_f)
                      Linear(312→200)
                               │
                          200 species
```

### Key Design Decisions

- **Masked Fine Head:** `p_f[i] = σ(z_f[i]) × p_c[parent(i)]` — if a part is absent, all its child attributes are suppressed. This is a **hard architectural constraint**, enforced at both training and inference time.
- **Bottleneck Property:** Classifier reads **only from p_f (312-dim)**, never from the backbone feature map. If the classifier also reads from features, concepts can be bypassed (interpretability illusion).
- **L1 from attribute parsing:** L1 labels are NOT from Part Locations. They are derived by OR-aggregating L2 attributes within each part group parsed from attribute names (`has_bill_shape::dagger → parent: "bill"`). Part Locations are used only for visibility masking in loss computation.

---

## Environment

- Python 3.11 virtual environment at `.venv/`
- Activate: `source .venv/bin/activate` (Linux) or `.venv\Scripts\activate` (Windows)
- Key dependencies: `torch`, `torchvision`, `pillow`, `matplotlib`, `numpy`
- Install: `pip install pillow matplotlib numpy torch torchvision`

---

## Project Structure

```
hierarchical-cbm-cub/
├── DB/
│   └── DB1/
│       ├── CUB_200_2011/CUB_200_2011/   ← images, annotations (double-nested)
│       ├── CUB_200_2011/attributes.txt   ← 312 attribute names (between outer and inner dir)
│       └── DB1-Mask/segmentations/       ← segmentation masks (for XAI eval only)
├── src/                                  ← extracted Python modules (future)
├── data.ipynb                            ← data pipeline
├── model.ipynb                           ← model definition
└── CLAUDE.md
```

---

## Data Pipeline Details

- **Dataset root in code:** `dataset_dir = 'DB/DB1/CUB_200_2011/CUB_200_2011/'` (the inner dir with actual CUB files)
- **attributes.txt** is at `DB/DB1/CUB_200_2011/attributes.txt` (between the outer and inner `CUB_200_2011/` dirs)
- **Official splits:** 5,994 train / 5,794 test. Val = 10% stratified from train
- **L1 label construction:**
  1. Parse `attributes.txt` → extract category before `::` (e.g., `has_bill_shape` → `bill`)
  2. Strip `has_` prefix and property suffix (`_color`, `_pattern`, `_shape`, `_length`) to get part name
  3. Merge compound parts: `under_tail`/`upper_tail` → `tail`; `upperparts` → `back`; `underparts` → `belly`
  4. Attributes without a clear body part (`primary_color`, `size`, `shape`) have no L1 parent — always unmasked
  5. Build `part_to_attr_indices` dict mapping each of the 13 parts to its attribute indices
  6. `L1[part] = max(L2[attr] for attr in part_to_attr_indices[part])` — OR aggregation
- **L2 label construction:** from `image_attribute_labels.txt`, filter `certainty ≥ 3`
- **Visibility masking:** use `part_locs.txt` visible flag to mask loss on invisible parts
- **Class labels:** 1-indexed in CUB → convert to 0-indexed in `__getitem__`
- **ImageNet normalization:** `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

### Image Preprocessing

```
Train:    Resize(256) → RandomCrop(224) → RandomHorizontalFlip
          → ColorJitter(mild) → Normalize
Val/Test: Resize(256) → CenterCrop(224) → Normalize
```

---

## Loss Function

```
L_total = λ_c · L_coarse + λ_f · L_fine + λ_cls · L_task + λ_h · L_consistency
```

| Term            | Formula                                   | Notes                                   |
| --------------- | ----------------------------------------- | --------------------------------------- |
| `L_coarse`      | `WeightedBCE(z_c, y_l1)`                  | `pos_weight = N_neg/N_pos` per part     |
| `L_fine`        | `FocalLoss(z_f, y_l2, α=0.25, γ=2)`       | masked on certainty≥3 and visible parts |
| `L_task`        | `CrossEntropy(species_logits, y_class)`   | standard classification loss            |
| `L_consistency` | `mean(max(0, p_f[child] - p_c[parent])²)` | penalizes child > parent                |

**Lambda weighting:** Use Uncertainty Weighting (Kendall et al. 2018) — learnable `σᵢ` per task:

```
L = Σᵢ (1/2σᵢ²) · Lᵢ + log(σᵢ)
```

This replaces manual grid search for λ values.

---

## Training Procedure

### Optimizer

```
AdamW, weight_decay=1e-4
Backbone (ResNet-50):  lr = 1e-4   ← lower to prevent catastrophic forgetting
Heads + Classifier:    lr = 1e-3
```

### Sequential Training Phases

Training is sequential to prevent the classifier from bypassing concepts.

| Phase  | Epochs | Backbone           | Active Losses                       |
| ------ | ------ | ------------------ | ----------------------------------- |
| **1a** | 1–15   | Frozen             | `L_coarse` only                     |
| **1b** | 16–30  | Frozen             | `L_coarse + L_fine + L_consistency` |
| **2**  | 31–60  | Unfrozen (lr=1e-4) | All — `L_task` with linear warm-up  |
| **3**  | 61–100 | Unfrozen           | All — final fine-tuning             |

**L_task warm-up:** `λ_task(t) = (t - 30) / 30` for t ∈ [30, 60], then 1.0

**Early stopping:** patience=15 on `val_loss`

**LR schedule:** Cosine Annealing or ReduceLROnPlateau (patience=5, factor=0.5)

---

## Evaluation

### Standard Metrics

- Top-1 and Top-5 Accuracy (200 species)
- L1 Concept Accuracy (13 coarse parts)
- L2 Concept macro-F1 (312 attributes — F1 preferred over accuracy due to imbalance)
- AUROC per concept → macro-average

### Hierarchy-Aware Metrics

- **Concept Intervention:** replace predicted concepts with ground-truth at rates [0%, 25%, 50%, 75%, 100%]. Plot accuracy vs intervention rate. Steep slope = meaningful concepts.
- **Hierarchical Distance:** use biological taxonomy (NABirds) to measure LCA depth of mistakes. Lower severity = model confuses semantically similar species.
- **Error Localization:** classify each mistake as L1-failure, L2-failure, or Both.
- **Hierarchy Consistency Rate:** fraction of predictions where `p_f[child] > p_c[parent]` (violations).

### Baselines (required for comparison)

1. ResNet-50 end-to-end (no CBM) — accuracy upper bound
2. Flat CBM — only L2, no hierarchy (Koh et al. 2020)
3. Parallel multi-task — L1 and L2 both from Feature Map (no mask)
4. **Our model** — Sequential/Masked Hierarchical CBM

### XAI Analysis

- Segmentation masks at `DB/DB1-Mask/segmentations/` — use for faithfulness evaluation only, NOT for training

---

## Implementation Status

| Component        | File          | Status                   |
| ---------------- | ------------- | ------------------------ |
| Data pipeline    | `data.ipynb`  | ✅ Complete              |
| Model definition | `model.ipynb` | ⚠️ Needs fix (see below) |
| Loss functions   | —             | ❌ Not implemented       |
| Training loop    | —             | ❌ Not implemented       |
| Evaluation       | —             | ❌ Not implemented       |

### Required Fixes in model.ipynb

Current implementation has three critical issues:

1. **Classifier reads from Feature Map** → must read from `p_f` only
2. **No Masked Fine Head** → `p_f[i] = σ(z_f[i]) × p_c[parent(i)]` must be added
3. **L1 has 7 concepts** → must be ~13 parts derived from attribute name parsing, not manually defined

---

## References

|     | Paper                                                      | Relevance                             |
| --- | ---------------------------------------------------------- | ------------------------------------- |
| [1] | Koh et al. — _Concept Bottleneck Models_ (ICML 2020)       | CBM foundation + Concept Intervention |
| [2] | Pittino et al. — _Hierarchical CBM for Vision_ (EAAI 2023) | Closest prior work — baseline         |
| [3] | Sun et al. — _Supervised Hierarchical CBM_ (under review)  | Sequential training                   |
| [4] | Poeta et al. — _Concept-based XAI Survey_ (ACM 2025)       | Survey — teacher is co-author         |
| [5] | Bertinetto et al. — _Making Better Mistakes_ (CVPR 2020)   | Hierarchical Distance metric          |
| [6] | Hase et al. — _Hierarchical Prototypes_ (AAAI 2019)        | Hierarchical architecture             |
| [7] | Lin et al. — _Focal Loss_ (ICCV 2017)                      | Focal Loss for attribute imbalance    |
| [8] | Kendall et al. — _Uncertainty Weighting_ (CVPR 2018)       | Learnable λ weights                   |
