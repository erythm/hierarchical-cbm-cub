# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

---

## Project Overview

**Hierarchical Concept Bottleneck Model (H-CBM)** for explainable bird species classification using the CUB-200-2011 dataset.

The model predicts bird species (200 classes) through a **two-level concept hierarchy**:

- **Level 1 (L1) — Coarse Concepts:** 13 body-part groups derived by parsing CUB attribute names (`bill`, `wing`, `tail`, `breast`, `belly`, `leg`, `eye`, `throat`, `forehead`, `nape`, `crown`, `back`, `head`)
- **Level 2 (L2) — Fine Concepts:** 312 binary attributes from CUB-200-2011 (`has_bill_shape::dagger`, `has_wing_color::brown`, ...)
- **Classifier:** 200 bird species — reads **only from p_f (L2)**, not from backbone features

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

- **Masked Fine Head:** `p_f[i] = σ(z_f[i]) × p_c[parent(i)]` — if a part is absent, all its child attributes are suppressed. Hard architectural constraint enforced at both training and inference time.
- **Bottleneck Property:** Classifier reads **only from p_f (312-dim)**, never from the backbone feature map.
- **L1 from attribute parsing:** L1 labels derived by OR-aggregating L2 attributes within each part group parsed from attribute names. Part Locations used only for visibility masking in loss.
- **No Consistency Loss:** Hierarchy enforced architecturally via Masked Fine Head.
- **No Sequential Training:** All losses active from epoch 1. Simple training loop with early stopping.

---

## Environment

- Python 3.11, Google Colab (recommended) or local `.venv/`
- Install: `pip install pillow matplotlib numpy torch torchvision scikit-learn`
- Dataset lives on Google Drive — not tracked by git

---

## Project Structure

```
hierarchical-cbm-cub/
├── src/              ← Python modules (future)
├── data.ipynb        ← data pipeline ✅
├── model.ipynb       ← model definition ✅
├── train.ipynb       ← training loop ❌
├── evaluate.ipynb    ← metrics + concept intervention ❌
├── xai.ipynb         ← faithfulness + segmentation masks ❌
├── CLAUDE.md
└── README.md
```

Dataset on Google Drive:

```
xai_dataset/
├── CUB_200_2011/
│   └── CUB_200_2011/   ← images + annotations (double-nested)
│       └── attributes.txt
└── segmentations/       ← XAI evaluation only
```

---

## Data Pipeline Details

- **Dataset root:** `'/content/drive/MyDrive/xai_dataset/CUB_200_2011/CUB_200_2011/'`
- **attributes.txt:** `os.path.join(dataset_dir, 'attributes.txt')`
- **Official splits:** 5,994 train / 5,794 test. Val = 10% stratified (`random_state=42`)
- **L1:** OR-aggregation per parsed part group — `certainty ≥ 3` filter + visibility masking
- **L2:** 312 binary attributes — `certainty ≥ 3` filter
- **Class labels:** 1-indexed in CUB → 0-indexed in `__getitem__`

### Saved Artifacts (data_pipeline.pkl)

```python
{
  'NUM_L1': 13,
  'NUM_L2': 312,
  'CONCEPT_NAMES': ['back','belly','bill','breast','crown','eye',
                    'forehead','head','leg','nape','tail','throat','wing'],
  'attr_parent_idx': np.array(312,)  # -1 = no L1 parent (always unmasked)
}
```

---

## Loss Function

```
L_total = λ_c · L_coarse + λ_f · L_fine + λ_cls · L_task
```

| Term       | Formula                                 | Notes                                   |
| ---------- | --------------------------------------- | --------------------------------------- |
| `L_coarse` | `WeightedBCE(z_c, y_l1)`                | masked by part visibility               |
| `L_fine`   | `FocalLoss(z_f, y_l2, α=0.25, γ=2)`     | masked on certainty≥3 and visible parts |
| `L_task`   | `CrossEntropy(species_logits, y_class)` | standard classification                 |

**Lambda weights (fixed):** `λ_c = 0.5, λ_f = 0.5, λ_cls = 1.0`

---

## Training Procedure

```
Optimizer:    AdamW (weight_decay=1e-4)
Backbone lr:  1e-4  ← lower to prevent catastrophic forgetting
Heads lr:     1e-3
Batch size:   32
Max epochs:   50  (early stopping — patience=10 on val_loss)
LR schedule:  ReduceLROnPlateau (patience=5, factor=0.5)
```

All losses active from epoch 1. No sequential phases.

---

## Evaluation

### Standard Metrics

- Top-1 / Top-5 Accuracy (200 species)
- L1 Concept Accuracy (13 parts)
- L2 Concept macro-F1 (312 attributes)
- AUROC per concept → macro-average

### Hierarchy-Aware Metrics

- **Concept Intervention:** inject ground-truth concepts at [0%, 25%, 50%, 75%, 100%]
- **Hierarchical Distance:** LCA depth of mistakes (NABirds taxonomy)
- **Error Localization:** L1-failure vs L2-failure vs Both
- **Hierarchy Consistency Rate:** violations of `p_f[child] ≤ p_c[parent]` (should be 0)

### Baselines

1. ResNet-50 end-to-end — accuracy upper bound
2. Flat CBM (Koh et al. 2020) — no hierarchy
3. Parallel multi-task — L1/L2 both from Feature Map, no mask
4. **H-CBM (ours)** — Masked Hierarchical CBM

### XAI Analysis (Phase 5)

- Segmentation masks at `xai_dataset/segmentations/` on Drive
- Faithfulness evaluation — do model attention maps align with actual bird region?
- NOT used during training

---

## Implementation Status

| Component            | File             | Status      |
| -------------------- | ---------------- | ----------- |
| Data pipeline        | `data.ipynb`     | ✅ Complete |
| Model definition     | `model.ipynb`    | ✅ Complete |
| Loss + Training      | `train.ipynb`    | ❌ Pending  |
| Evaluation + Metrics | `evaluate.ipynb` | ❌ Pending  |
| XAI Faithfulness     | `xai.ipynb`      | ❌ Pending  |

---

## References

|     | Paper                                                      | Relevance                             |
| --- | ---------------------------------------------------------- | ------------------------------------- |
| [1] | Koh et al. — _Concept Bottleneck Models_ (ICML 2020)       | CBM foundation + Concept Intervention |
| [2] | Pittino et al. — _Hierarchical CBM for Vision_ (EAAI 2023) | Closest prior work — baseline         |
| [3] | Poeta et al. — _Concept-based XAI Survey_ (ACM 2025)       | Survey — teacher is co-author         |
| [4] | Bertinetto et al. — _Making Better Mistakes_ (CVPR 2020)   | Hierarchical Distance metric          |
| [5] | Hase et al. — _Hierarchical Prototypes_ (AAAI 2019)        | Hierarchical architecture             |
| [6] | Lin et al. — _Focal Loss_ (ICCV 2017)                      | Focal Loss for attribute imbalance    |
