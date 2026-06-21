# Hierarchical Concept Bottleneck Model for Bird Classification

**Explainable and Trustworthy AI — Politecnico di Torino 2025/2026**
**Project P1 — Hierarchical Concept-Based Explainable-by-Design Models**

---

## What this project does (in one paragraph)

We classify bird species from the [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) dataset (200 classes, ~12k images), but we do it in a way that a human can read. Instead of going straight from pixels to a class name, the model first answers two human-friendly questions:

1. **Which body parts can I see?** — 13 parts (bill, wing, tail, …)
2. **For each visible part, what does it look like?** — 312 fine attributes (e.g. `bill_shape::dagger`, `wing_color::brown`)

Only then does it pick one of the 200 species. The final classifier is a single `Linear(312 → 200)` layer that sees **only** the 312 attribute probabilities — never the raw image features. So every prediction is fully explained by a small, named vector of concepts.

```
Image → [bill, wing, tail, …]                             ← Level 1: 13 body parts
      → [bill_shape::dagger, wing_color::brown, …]        ← Level 2: 312 fine attributes
      → Acadian Flycatcher                                ← Level 3: species (Linear from 312)
```

A key rule called the **Masked Fine Head** ties every fine attribute to the probability of its parent body part through a multiplicative gate that is applied **identically at training and at inference**:

$$p_f[i] = \sigma(z_f[i]) \;\times\; \bigl(0.5 + 0.5 \cdot p_c[\text{parent}(i)]\bigr)$$

The gate is in `[0.5, 1]` rather than `[0, 1]` on purpose: a hard `× p_c[parent]` collapses children to zero very early in training when `p_c ≈ 0.5` and the fine head can never recover. The soft form keeps gradients alive while still guaranteeing a *parent-conditioned ceiling* — when `p_c[parent] → 0`, every child is multiplied by 0.5 (cannot exceed half), and when `p_c[parent] → 1` the child is unrestricted. The forward pass in `02 / 03 / 04` all use this exact same formula, so concepts at evaluation time are produced under the same constraint they were trained under.

---

## Two parallel models (this is NOT a pipeline)

We did **not** build one model and then upgrade it. We built **two completely separate models** with the **same H-CBM design** but **different visual backbones**, trained them independently, explained each one in its own notebook, and then compared them in a final notebook. The whole point is the comparison.

```
                ┌──────────────────────────────────────┐
                │   01_data.ipynb                      │
                │   Shared data prep                   │
                │   (one notebook, used by both)       │
                └──────────────┬───────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
   ┌────────────────────────┐     ┌────────────────────────┐
   │  Branch A — ResNet-50  │     │  Branch B — ViT-B/16   │
   │                        │     │                        │
   │  02 ... train           │     │  03 ... train           │
   │  02 ... explainability  │     │  03 ... explainability  │
   └───────────┬────────────┘     └───────────┬────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
                ┌──────────────────────────────┐
                │   04 ... Comparison           │
                │   ResNet-50 vs ViT-B/16      │
                │   (head-to-head, same data)  │
                └──────────────────────────────┘
```

Both branches use:
- the **same 13 coarse parts** and **same 312 fine attributes**,
- the **same hierarchical mask**,
- the **same final `Linear(312 → 200)` classifier** (no extra MLP, no shortcut features).

Only the backbone changes. So any difference we see in the comparison notebook comes purely from the backbone choice.

---

## Files in this repo

| # | File | What it does |
|---|------|--------------|
| 01 | `01_data.ipynb` | Loads CUB, builds the 13/312 hierarchy, makes train/val/test splits, saves two pickles (`data_pipeline_resnet.pkl` for the ResNet branch, `data_pipeline_vit.pkl` for the ViT branch). |
| 02a | `02_H-CBM ResNet-50_train.ipynb` | Trains the H-CBM with a **ResNet-50** backbone. Saves `best_model.pth`. |
| 02b | `02_H-CBM ResNet-50_explainability.ipynb` | Loads the ResNet-50 checkpoint and runs the full explanation suite (concept fidelity, oracle, TTI, per-part ablation, calibration, classifier weight inspection, etc.). |
| 03a | `03_H-CBM ViT-B16_train.ipynb` | Trains the H-CBM with a **ViT-B/16** backbone. Saves `best_hcbm_vit_384.pth`. |
| 03b | `03_H-CBM ViT-B16_explainability.ipynb` | Same explanation suite as 02b, but for the ViT model — plus the native CLS-token attention map, which ResNet cannot produce. |
| 04 | `04_H-CBM Comparison ResNet-50 vs ViT-B16.ipynb` | Loads **both** trained models, runs the **same** evaluation on both, and reports head-to-head differences (accuracy, concept AUC, oracle gap, TTI curves, per-part necessity, calibration, classifier-weight structure, agreement matrix, final verdict table). |
| — | `src/` | Reserved for shared utility modules. |

The two branches are **independent**. You can run only branch A, or only branch B, and the explanation notebook for that branch will work on its own. Notebook 04 needs both checkpoints on disk.

---

## The H-CBM architecture (same skeleton in both branches)

```
        Input image
             │
             ▼
   ┌──── Backbone ───────┐         ResNet-50  → 2048-dim pooled vector
   │ (ResNet-50 OR ViT)  │         ViT-B/16   → 768-dim [CLS] token
   └─────────┬───────────┘
             │  features F
   ┌─────────┴─────────────┐
   ▼                       ▼
 Coarse head g_c(F)     Fine head g_f(F, p_c)
 → z_c (13 logits)      input = concat(F, p_c)
 → p_c = σ(z_c)         → z_f (312 logits)
                        → p_f_raw = σ(z_f)
                        → mask[i] = 0.5 + 0.5·p_c[parent(i)]   ← Masked Fine Head
                        → p_f = p_f_raw × mask              (parent-conditioned ceiling)
             │
             ▼
       p_f (312-dim)
             │
             ▼
   Classifier h(p_f) = Linear(312 → 200)
             │
             ▼
       species (1 of 200)
```

| Block | Layers |
|-------|--------|
| Coarse head | `Linear(D → 256) → BN → ReLU → Dropout(0.2) → Linear(256 → 13)` |
| Fine head   | `Linear(D+13 → 512) → BN → ReLU → Dropout(0.4) → Linear(512 → 312)` |
| Classifier  | `Linear(312 → 200)` (kept linear on purpose — see below) |

`D` is 2048 for ResNet-50 and 768 for ViT-B/16. Everything above the backbone is identical.

### Why the classifier is a single `Linear` layer

Because the whole point is interpretability. With a linear classifier, the weight `W[k, i]` is literally *“how much does attribute `i` count in favour of species `k`?”*. Any non-linear classifier on top of `p_f` would let the model invent hidden interactions that a human cannot read off the weights. So we keep the classifier linear in **both** branches.

### Why no residual / shortcut path

In some CBM papers the classifier also reads a side-channel directly from the backbone (a `W_r` term). We do **not** do this. The classifier sees only `p_f`. So if a concept is wrong, the prediction is wrong — there is no “escape route” around the bottleneck. This makes the explanations honest.

---

## Loss

$$L_{\text{total}} = \lambda_c \, L_{\text{coarse}} + \lambda_f \, L_{\text{fine}} + \lambda_t \, L_{\text{task}}$$

| Term | Formula | Mask |
|------|---------|------|
| `L_coarse` | Weighted BCE on `z_c` vs. the 13 L1 labels (`pos_weight = N_neg / N_pos` per part) | masked by per-image part visibility |
| `L_fine`   | Focal Loss (α=0.25, γ=2) on `z_f` vs. the 312 L2 labels | masked by certainty ≥ 3 |
| `L_task`   | Cross-Entropy with label smoothing (0.1) on the 200-way logits | none |

Focal Loss is used for L_fine because most attributes are negative for most birds (`has_wing_color::pink` is positive for under 2 % of images), and plain BCE would just learn to predict zero everywhere.

---

## Training schedule (3 phases — same idea in both branches)

| Phase | What is trainable | Loss | Why |
|-------|-------------------|------|-----|
| **1 — Joint heads** | Coarse head + Fine head + Classifier (backbone frozen) | `λ_c·L_coarse + λ_f·L_fine + 0.1·L_task` | Learn the 13/312 concept space on top of frozen features. Adding a small task term shapes the concepts to be discriminative for classification from the start. |
| **2 — Calibration** | Classifier only (heads + backbone frozen) | `L_task` on the model’s **predicted** `p_f` | Crucial trick: the classifier sees predicted `p_f` (not the GT `l2`) so its training distribution matches what it will see in Phase 3. This fixes the “P3 collapses on epoch 1” bug that comes from training on GT concepts. |
| **3 — Joint fine-tune** | Everything (backbone unfrozen) | `2·L_coarse + 2·L_fine + 1·L_task`, with **MixUp(α=0.2)** on Phase 3 batches | Refine end-to-end with concept-dominant weights and stronger weight decay so the model gently improves instead of overfitting in the first 10 epochs. |

Other shared training details:
- **Optimizer:** `AdamW` with two parameter groups (lower LR for the backbone, higher LR for the heads/classifier).
- **Phase-3 schedule:** linear warmup (5 epochs) → cosine annealing.
- **Hierarchical mask is soft at both training and inference:** `mask = 0.5 + 0.5 · p_c[parent]` (not 0/1) so child attributes do not collapse to zero early when `p_c` is still ~0.5, and so the eval-time concept distribution exactly matches what the classifier was trained on.
- **Best checkpoint is saved by best `val_acc`**, not by `val_loss`, because the concept-dominant loss is no longer monotonic with accuracy.

### Per-branch differences

Only a few hyper-parameters differ between the two branches, because the two backbones have very different sensitivities.

| Parameter | ResNet-50 branch | ViT-B/16 branch | Why |
|-----------|------------------|-----------------|-----|
| Input size | 336 × 336 | 384 × 384 | ViT-B/16 SWAG was pre-trained at 384. |
| Batch size (H100) | 96 | 96 | Both fit at BF16 with `torch.compile`. |
| Backbone LR (P1/P2) | 5e-5 | 1e-6 | ViT attention is fragile — a too-high LR destroys SWAG features instantly. |
| Heads LR (P1/P2) | 1e-4 | 3e-4 | Standard AdamW sweet spot. |
| Phase-3 backbone LR | 1e-5 | (similar order, slightly lower) | Joint refinement needs much lower LRs than Phase 1. |
| Pretrain weights | `IMAGENET1K_V2` | `IMAGENET1K_SWAG_E2E_V1` | SWAG was trained on 3.6 B Instagram images and then fine-tuned on ImageNet at 384. |
| Checkpoint name | `best_model.pth` | `best_hcbm_vit_384.pth` | So both branches can coexist on Drive without overwriting each other. |
| Data pickle | `data_pipeline_resnet.pkl` | `data_pipeline_vit.pkl` | The two pickles only differ in `IMG_SIZE`; the splits are byte-identical. |

---

## Why two backbones — what we wanted to find out

A CBM is only as good as its concepts. So we asked:

> Does the choice of backbone change the **interpretability** of an H-CBM, not just its accuracy?

A pure-CNN backbone (ResNet-50) compresses the image to a single pooled vector before the concept heads see it. That is a strong spatial bottleneck and may hurt fine-grained attributes like *bill shape* or *wing colour*.

A Vision Transformer (ViT-B/16) keeps a 196-patch token sequence and aggregates it through self-attention into the `[CLS]` token. Two consequences for XAI:

1. **Better concept features.** Self-attention can pool from any patch at any layer, so subtle attributes can survive into the final feature.
2. **Faithful spatial attribution for free.** The `[CLS]` token attention weights over the 196 patches are the model’s own internal evidence — not a post-hoc approximation like Grad-CAM. This shows up as an extra panel in the ViT explanation notebook (the 14×14 attention heatmap), which the ResNet branch simply cannot produce without a surrogate method.

Whether ViT actually wins on concept AUC, oracle gap, TTI curves, calibration and classifier-weight sparsity is exactly what notebook 04 measures.

---

## What each notebook actually shows

### `02_H-CBM ResNet-50_explainability.ipynb` and `03_H-CBM ViT-B16_explainability.ipynb`

Both notebooks follow the same recipe (same sections, same plots, same metrics) so the two are directly comparable:

- **Section 1 — One-image walkthrough.** For a single image: show the input, the 13 part probabilities, the top-8 active fine attributes, and the predicted species. The ViT version also shows the CLS-token attention heatmap.
- **Section 2 — Gallery.** Same walkthrough but for several images side-by-side.
- **Section 3 — Eval cache.** One forward pass over the full eval split; from this point on everything is fast.
- **Section 4 — Concept fidelity.** Per-attribute ROC-AUC distribution over the 312 attributes.
- **Section 5 — Oracle accuracy (class-prototype).** The classifier head was trained on the model's *continuous predicted* `p_f`, not on binary `{0,1}` GT. Feeding raw GT attributes as a fake oracle is therefore out-of-distribution and produces near-chance accuracy (~10%). The honest ceiling is the **class-prototype oracle**: replace each image's `p_f` with the in-class average `proto[k] = mean p_f over images of class k`, then run the frozen classifier. This stays exactly on the training distribution and gives a faithful upper bound. The naive GT-0/1 oracle is also reported for context.
- **Section 6 — TTI (Test-Time Intervention).** Mix predicted `p_f` with the **class-prototype** `p_f` along a fraction `r ∈ {0, 0.1, 0.25, 0.5, 0.75, 1}` (3 random seeds per fraction, random columns). By construction `r=0` recovers actual accuracy and `r=1` recovers the prototype-oracle accuracy; a monotonically rising curve is the genuine causal-control evidence.
- **Section 7 — Per-part causal necessity.** For each of the 13 body parts, **zero the child `p_f` columns** directly and re-measure accuracy. Zeroing `p_c[part]` instead would only halve those children through the soft mask `0.5 + 0.5·p_c[parent]`; zeroing the columns themselves removes that branch from `W·p_f` entirely and gives the honest counterfactual *"what would the model do if it could not see this body part at all?"*.
- **Section 8 — Classifier weight structure.** Inspect `W ∈ ℝ^{200×312}`: which attributes define which species, how sparse the definitions are (L1/L2 ratio + top-k mass curve), how concentrated they are on a few body parts (Herfindahl over parts).
- **Section 9 — Calibration.** 15-bin reliability diagram for the 312 attribute probabilities + ECE.

### `04_H-CBM Comparison ResNet-50 vs ViT-B16.ipynb`

Loads both checkpoints into the **same Python process**, runs the exact same evaluation on both (one shared cache so every comparison is paired image-by-image), and reports head-to-head differences:

| § | Comparison | Question it answers |
|---|------------|--------------------|
| 1 | Paired eval cache | Single forward pass per backbone over the **same** eval split → `pc, pf, zf, logit, pred, lbl, l2, W, b` per model. Headline `top-1` is printed as a sanity check. |
| 2 | Headline metrics | Top-1, mean per-class accuracy, NLL, mean attribute-AUC, attribute Brier, attribute ECE — one row per backbone. |
| 3 | Concept fidelity | Per-attribute AUC histograms + paired scatter (one dot per attribute) + top-10 attributes where each backbone wins. |
| 4 | Oracle accuracy | **Class-prototype oracle** (matches §5 of 02/03): Actual vs Oracle vs Gap, with the naive GT-0/1 number reported alongside as a calibration-of-method check. |
| 5 | TTI curves overlaid | Class-prototype intervention sweep `r ∈ {0, 0.1, 0.25, 0.5, 0.75, 1}` × 3 seeds — does each model's accuracy rise smoothly toward its own oracle line? |
| 6 | Per-part causal necessity | 13 parts × 2 bars — for each body part, zero the **child `p_f` columns** of both models and compare the accuracy drop. Reveals whether the two backbones rely on the same parts. |
| 7 | Calibration | Reliability diagrams side-by-side. |
| 8 | Classifier weight structure | Sparsity (L1/L2), top-k mass curve, part-Herfindahl — whose species definitions are more concise? |
| 9 | Agreement matrix | both right / only-ResNet / only-ViT / both wrong + Cohen's κ + oracle-selection and confidence-tie-break ensemble ceilings + per-class breakdown of "only-X correct". |
| 10 | Verdict table | One row per backbone, every metric in one place, plus an "axes won" score. |

---

## Dataset

CUB-200-2011 (11,788 images, 200 species).

| Split | Size |
|-------|------|
| Train | ~5,394 |
| Val   | ~600 (10 % stratified hold-out from train, `random_state=42`) |
| Test  | 5,794 |

- **L1 (13 coarse parts):** `back, belly, bill, breast, crown, eye, forehead, leg, nape, tail, throat, underparts, wing` — built automatically by parsing the prefix of `attributes.txt` (no extra labelling needed).
- **L2 (312 fine attributes):** the standard CUB attributes, kept only when `certainty ≥ 3`.

Expected layout on Google Drive:

```
MyDrive/XAI-Project/DB/DB1/
├── CUB_200_2011/         ← images + the official txt files
├── attributes.txt
├── checkpoints/          ← best_model.pth, best_hcbm_vit_384.pth, …
└── pipeline/             ← data_pipeline_resnet.pkl, data_pipeline_vit.pkl, train history
```

Downloads:
- [CUB-200-2011 (1.2 GB)](https://data.caltech.edu/records/65de6-vp158)
- [Segmentation masks (39 MB)](https://data.caltech.edu/records/w9d68-gec53) (optional, only used for nicer plots)

---

## Environment

```bash
git clone https://github.com/erythm/hierarchical-cbm-cub.git
cd hierarchical-cbm-cub

python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux / macOS

pip install torch torchvision pillow matplotlib numpy scikit-learn pandas
```

The training notebooks are written for **Google Colab with an A100 / H100** (BF16 AMP, `torch.compile`, large batch). The explainability and comparison notebooks are light and run on any GPU (or even CPU, slowly). The Colab-only cells (`drive.mount`) are at the top of each notebook and easy to skip when running locally.

---

# Literature Review (Deliverable 1)

| Group | Papers |
|-------|--------|
| **G1 — CBM foundations**   | Koh et al. ICML 2020 [1], Hase et al. AAAI 2019 [6] |
| **G2 — Hierarchical CBMs** | Pittino et al. EAAI 2023 [2], Sun et al. (under review) [3] |
| **G3 — XAI context**       | Poeta et al. ACM CSUR 2025 [4] |
| **G4 — Supporting methods**| Lin et al. ICCV 2017 [7], Kendall et al. CVPR 2018 [8], Bertinetto et al. CVPR 2020 [5] |

### [1] Koh et al. — *Concept Bottleneck Models* (ICML 2020)

The original CBM idea: the model first predicts $\hat{c} = g(x)$ (a vector of human-named concepts), then $\hat{y} = h(\hat{c})$, with the strict rule that $h$ sees **only** $\hat{c}$. Three training modes are introduced: independent, sequential, joint — each with a different trade-off between concept quality and task accuracy. The paper also introduces *concept intervention* (replace a predicted concept with its true value, watch accuracy go up) — the standard CBM diagnostic. **Limit:** all concepts are treated as a flat unstructured list. There is no notion that `bill_shape::dagger` and `bill_color::black` both belong to *bill*.

### [6] Hase et al. — *Interpretable Image Recognition with Hierarchical Prototypes* (AAAI 2019)

Prototype-based recognition where the prototypes are organised in a class taxonomy (animal → bird → warbler). Different from H-CBM (the hierarchy is over **classes**, not over attributes), but it shows that putting hierarchy *inside* the architecture — instead of bolting it on afterwards — gives more consistent explanations.

### [2] Pittino et al. — *Hierarchical CBM for Vision* (EAAI 2023)

The closest prior work. They use a high-level concept predictor that conditions a low-level one, then a final classifier. Same shape as our pipeline. **Difference:** their hierarchy constraint is a **soft penalty** during training (a regulariser that says “please keep `p_low ≤ p_high[parent]`”). At inference time, nothing prevents a violation. We instead use a **hard multiplicative mask** — so violations are mathematically impossible.

### [3] Sun et al. — *Supervised Hierarchical CBM* (under review)

Trains the concept predictors first, then the classifier — a **sequential training schedule**. This avoids the well-known leakage problem of joint CBM training (where the classifier learns to exploit fine-grained noise in the predicted concepts and ignore their semantic meaning). Our 3-phase training schedule follows this idea.

### [4] Poeta et al. — *Concept-based XAI: A Survey* (ACM CSUR 2025)

A taxonomy paper for concept-based XAI: post-hoc vs by-design, user-defined vs data-driven concepts, global vs local, faithful vs unfaithful, complete vs incomplete. Crucially, the survey lists four open problems for CBMs: (i) annotation cost, (ii) concept leakage, (iii) flat concept sets, (iv) no hierarchy-aware evaluation. We attack (iii) and (iv) directly.

### [7] Lin et al. — *Focal Loss for Dense Object Detection* (ICCV 2017)

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Down-weights easy examples so the model is forced to focus on hard ones. We use Focal Loss (α=0.25, γ=2) for the L_fine head because most CUB attributes are positive for under 5 % of images and plain BCE just learns to predict zero everywhere.

### [8] Kendall et al. — *Multi-Task Learning Using Uncertainty to Weigh Losses* (CVPR 2018)

A principled way to weight several losses without manual grid search:

$$L = \sum_i \frac{1}{2 \sigma_i^2} L_i + \log \sigma_i$$

Cited here as the theoretical justification for the multi-loss form $L = \lambda_c L_c + \lambda_f L_f + \lambda_t L_t$. We currently use fixed $\lambda$ values; learning them is an obvious extension.

### [5] Bertinetto et al. — *Making Better Mistakes* (CVPR 2020)

Standard top-1 treats every wrong answer as equally bad. Bertinetto et al. score mistakes by **lowest common ancestor depth** in a class hierarchy: confusing two warblers is a mild mistake; confusing a warbler with a gull is severe. We use this idea as a *family-level* sanity check on CUB (mild = same family, severe = different family).

---

# Research Gaps (Deliverable 2)

| Gap | Problem | Prior best | Our solution | Type of guarantee |
|-----|---------|------------|--------------|-------------------|
| **G1** | Flat concept sets ignore hierarchies | Soft conditioning (Pittino [2]) | `attr_parent_idx` parsed from attribute names → 13 L1 groups, no extra labels | Architectural |
| **G2** | No coarse-to-fine constraint baked into the architecture | Soft consistency penalty in the loss (Pittino [2]) | Multiplicative gate **inside the forward pass**: $p_f[i] = \sigma(z_f[i]) \cdot (0.5 + 0.5\,p_c[\text{parent}(i)])$ | Architectural — `p_f[i]` is parent-conditioned by construction at every forward, no loss term required |
| **G3** | CBMs are not evaluated on hierarchy quality | Mistake severity for plain classifiers (Bertinetto [5]) | Concept AUC, oracle gap, TTI, per-part necessity, calibration, classifier-weight sparsity — same suite on **both** backbones in notebook 04 | Evaluation |
| **G4** | Classifier leakage bypasses the bottleneck | Joint training with consistency loss | Strict bottleneck (`Linear(312 → 200)`, no shortcut) + sequential 3-phase schedule | Architectural |

### G1 — flat → hierarchical (no extra labels)

We never asked an annotator for the 13 part labels. Instead we parsed the prefix of every CUB attribute name:

```
has_bill_shape::dagger      → bill
has_wing_color::brown       → wing
has_size                    → NO_PARENT (mask = 1)
```

Then we OR-aggregate: a part is “present” for an image if at least one of its attributes is present in the GT. Result: 13 binary L1 labels per image, **for free**.

### G2 — loss-side soft penalty → architectural multiplicative gate

In Pittino et al. the coarse-to-fine consistency lives only in the loss: at inference nothing prevents a child being more confident than its parent. We move the constraint **inside the forward pass** as `p_f[i] = σ(z_f[i]) · (0.5 + 0.5·p_c[parent(i)])`. The gate is in `[0.5, 1]` (not `[0, 1]`) on purpose — a hard `× p_c[parent]` collapses children to zero whenever `p_c ≈ 0.5` early in training and the fine head never recovers. The soft form keeps gradients alive while still enforcing a *parent-conditioned ceiling*: when `p_c[parent] → 0` every child is capped at `0.5·σ(z_f)`, and the same gate is used identically at training and inference, so the classifier never sees an out-of-distribution `p_f`.

### G3 — broader evaluation, on both backbones

Each branch’s explainability notebook reports concept AUC, oracle, TTI, per-part necessity, calibration and weight structure. Notebook 04 then runs the **same** evaluation on both models in the same memory and reports paired differences — including the agreement matrix and the “oracle ensemble” ceiling, which reveal *complementarity* between the two backbones.

### G4 — strict bottleneck + sequential phases

The classifier is `Linear(312 → 200)` with **no** side-channel from the backbone. Training is sequential (Phase 1 — heads first; Phase 2 — calibrate the classifier on **predicted** concepts; Phase 3 — joint fine-tune with concept-dominant loss weights). This combination removes both leakage paths: structural (no shortcut features) and temporal (concepts are stable before the classifier is allowed to use them).

---

## Checkpoints written to disk

| File | Written by | Contents |
|------|------------|----------|
| `data_pipeline_resnet.pkl` | `01_data.ipynb` | Splits + metadata for the ResNet branch (IMG_SIZE=336). |
| `data_pipeline_vit.pkl`    | `01_data.ipynb` | Same splits, IMG_SIZE=384, for the ViT branch. |
| `best_model.pth`           | `02 ... train`   | ResNet-50 H-CBM, best by `val_acc`. |
| `train_history.pkl`        | `02 ... train`   | Per-epoch losses + accuracies. |
| `best_hcbm_vit_384.pth`    | `03 ... train`   | ViT-B/16 H-CBM, best by `val_acc`. |
| `train_history_vit_384.pkl`| `03 ... train`   | Same, for the ViT branch. |

---

## References

| | Paper | Venue | Why we cite it |
|-|-------|-------|----------------|
| [1] | Koh, P.W. et al. — *Concept Bottleneck Models* | ICML 2020 | CBM foundation + concept intervention |
| [2] | Pittino, F. et al. — *Hierarchical CBM for Vision* | EAAI 2023 | Closest prior work — direct baseline |
| [3] | Sun, X. et al. — *Supervised Hierarchical CBM* | Under review | Sequential training schedule |
| [4] | Poeta, E. et al. — *Concept-based XAI: A Survey* | ACM CSUR 2025 | Survey — defines the open problems we attack |
| [5] | Bertinetto, L. et al. — *Making Better Mistakes* | CVPR 2020 | Hierarchy-aware mistake severity |
| [6] | Hase, P. et al. — *Interpretable Image Recognition with Hierarchical Prototypes* | AAAI 2019 | Hierarchical architecture (over classes) |
| [7] | Lin, T.Y. et al. — *Focal Loss for Dense Object Detection* | ICCV 2017 | Focal Loss for the L2 attribute imbalance |
| [8] | Kendall, A. et al. — *Multi-Task Learning Using Uncertainty* | CVPR 2018 | Theoretical basis for multi-loss weighting |
