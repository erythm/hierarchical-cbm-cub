# Hierarchical Concept Bottleneck Model for Bird Classification

**Explainable and Trustworthy AI — Politecnico di Torino 2026**
**Project P1 — Hierarchical Concept-Based Explainable-by-Design Models**

---

## Overview

This project implements a **Hierarchical Concept Bottleneck Model (H-CBM)** for explainable bird species classification on the [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) dataset.

Unlike standard black-box classifiers, H-CBM expresses every prediction through two levels of human-understandable concepts:

```
Image → [bill ✅  wing ✅  tail ✅]        ← Level 1: body parts
      → [bill_shape_dagger ✅  wing_color_brown ✅  ...] ← Level 2: fine attributes
      → Acadian Flycatcher 🐦
```

A key architectural constraint — the **Masked Fine Head** — ensures that if a body part is predicted absent, none of its child attributes can be active. This enforces a hard hierarchical consistency at both training and inference time.

---

## Architecture

```
Input Image (3, 224, 224)
        │
        ▼
ResNet-50 (pretrained ImageNet)
        │
   Feature Map F (2048,)
        │
        ├──────────────────────────────┐
        ▼                              │ F
Coarse Head g_c(F)                    │
Linear(2048→512)→ReLU→Dropout(0.3)   │
→Linear(512→13)                       │
        │                              ▼
   p_c = σ(z_c)       Fine Head g_f(F, p_c)
   13 part probs       concat([F, p_c]) → [2061]
        │              Linear(2061→512)→ReLU→Dropout(0.3)→Linear(512→312)
        │                      │
        │              p_f[i] = σ(z_f[i]) × p_c[parent(i)]   ← Masked Fine Head
        │                      │
        └──────────────────────┘
                               │ p_f (312,)
                               ▼
                      Classifier h(p_f)
                      Linear(312→200)
                               │
                          200 species
```

### Design Principles

| Principle               | Description                                                           |
| ----------------------- | --------------------------------------------------------------------- |
| **Masked Fine Head**    | `p_f[i] = σ(z_f[i]) × p_c[parent(i)]` — hard architectural constraint |
| **Bottleneck Property** | Classifier reads only from `p_f`, never from raw features             |
| **L1 from parsing**     | Coarse concepts derived from attribute name structure, not hardcoded  |
| **Sequential Training** | Concepts are learned before task loss is introduced                   |

---

## Dataset

**CUB-200-2011** — 11,788 images of 200 North American bird species.

| Split | Size                             |
| ----- | -------------------------------- |
| Train | ~5,394                           |
| Val   | ~600 (10% stratified from train) |
| Test  | 5,794                            |

**Two annotation levels used:**

- **L1 (13 coarse parts):** derived by parsing 312 attribute names → `bill, wing, tail, breast, belly, leg, eye, throat, forehead, nape, crown, back, head`
- **L2 (312 fine attributes):** binary attributes filtered by `certainty ≥ 3`

Download the dataset:

- [CUB-200-2011 (1.2 GB)](https://data.caltech.edu/records/65de6-vp158)
- [Segmentation masks (39 MB)](https://data.caltech.edu/records/w9d68-gec53) — for XAI evaluation only

Place them under:

```
DB/
└── DB1/
    ├── CUB_200_2011/
    │   └── CUB_200_2011/     ← images + annotations (double-nested)
    └── DB1-Mask/
        └── segmentations/    ← XAI evaluation only
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/erythm/hierarchical-cbm-cub.git
cd hierarchical-cbm-cub

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install torch torchvision pillow matplotlib numpy scikit-learn
```

---

## Project Structure

```
hierarchical-cbm-cub/
├── DB/                          ← dataset (not tracked by git)
│   └── DB1/
│       ├── CUB_200_2011/
│       │   └── CUB_200_2011/
│       └── DB1-Mask/
├── src/                         ← Python modules (future)
├── data.ipynb                   ← data pipeline ✅
├── model.ipynb                  ← model definition ⚠️
├── CLAUDE.md                    ← Claude Code guidance
└── README.md
```

---

## Loss Function

```
L_total = λ_c · L_coarse + λ_f · L_fine + λ_cls · L_task + λ_h · L_consistency
```

| Term            | Details                                                  |
| --------------- | -------------------------------------------------------- |
| `L_coarse`      | Weighted BCE on L1 — masked by part visibility           |
| `L_fine`        | Focal Loss (α=0.25, γ=2) on L2 — masked by certainty ≥ 3 |
| `L_task`        | Cross-entropy on 200 species                             |
| `L_consistency` | `mean(max(0, p_f[child] - p_c[parent])²)`                |

Lambda weights are learned automatically via **Uncertainty Weighting** (Kendall et al. 2018).

---

## Training

Sequential training prevents the classifier from bypassing concepts:

| Phase | Epochs | Backbone | Active Losses                       |
| ----- | ------ | -------- | ----------------------------------- |
| 1a    | 1–15   | Frozen   | `L_coarse` only                     |
| 1b    | 16–30  | Frozen   | `L_coarse + L_fine + L_consistency` |
| 2     | 31–60  | Unfrozen | All — `L_task` with warm-up         |
| 3     | 61–100 | Unfrozen | All — final fine-tuning             |

```
Optimizer:  AdamW (weight_decay=1e-4)
Backbone:   lr = 1e-4
Heads:      lr = 1e-3
Epochs:     100 with early stopping (patience=15)
```

---

## Evaluation

### Standard Metrics

- Top-1 / Top-5 Accuracy (200 species)
- L1 Concept Accuracy (13 parts)
- L2 Concept macro-F1 (312 attributes)
- AUROC per concept → macro-average

### Hierarchy-Aware Metrics

- **Concept Intervention** — inject ground-truth concepts at [0%, 25%, 50%, 75%, 100%] and measure accuracy. Steep slope = meaningful concepts.
- **Hierarchical Distance** — measure LCA depth of mistakes using biological taxonomy (NABirds). Model should make semantically closer mistakes.
- **Error Localization** — classify each mistake as L1-failure, L2-failure, or Both.
- **Hierarchy Consistency Rate** — fraction of predictions violating `p_f[child] ≤ p_c[parent]`.

### Baselines

| Model                | Description                              |
| -------------------- | ---------------------------------------- |
| ResNet-50 end-to-end | No CBM — accuracy upper bound            |
| Flat CBM             | Only L2, no hierarchy (Koh et al. 2020)  |
| Parallel multi-task  | L1 and L2 both from Feature Map, no mask |
| **H-CBM (ours)**     | Sequential/Masked Hierarchical CBM       |

---

## Implementation Status

| Component        | File              | Status         |
| ---------------- | ----------------- | -------------- |
| Data pipeline    | `data.ipynb`      | ✅ Complete    |
| Model definition | `model.ipynb`     | ⚠️ In progress |
| Loss functions   | `src/loss.py`     | ❌ Pending     |
| Training loop    | `src/train.py`    | ❌ Pending     |
| Evaluation       | `src/evaluate.py` | ❌ Pending     |

---

## References

|     | Paper                                                                                    |
| --- | ---------------------------------------------------------------------------------------- |
| [1] | Koh et al. — _Concept Bottleneck Models_ — ICML 2020                                     |
| [2] | Pittino et al. — _Hierarchical CBM for Vision_ — EAAI 2023                               |
| [3] | Sun et al. — _Supervised Hierarchical CBM_ — Under review                                |
| [4] | Poeta et al. — _Concept-based XAI: A Survey_ — ACM Comput. Surv. 2025                    |
| [5] | Bertinetto et al. — _Making Better Mistakes_ — CVPR 2020                                 |
| [6] | Hase et al. — _Interpretable Image Recognition with Hierarchical Prototypes_ — AAAI 2019 |
| [7] | Lin et al. — _Focal Loss for Dense Object Detection_ — ICCV 2017                         |
| [8] | Kendall et al. — _Multi-Task Learning Using Uncertainty to Weigh Losses_ — CVPR 2018     |
