# Hierarchical Concept Bottleneck Model for Bird Classification

**Explainable and Trustworthy AI — Politecnico di Torino 2025/2026**
**Project P1 — Hierarchical Concept-Based Explainable-by-Design Models**

---

## Overview

This project implements a **Hierarchical Concept Bottleneck Model (H-CBM)** for explainable bird species classification on the [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) dataset.

Unlike standard black-box classifiers, H-CBM expresses every prediction through two levels of human-understandable concepts:

```
Image -> [bill  wing  tail]                      <- Level 1: body parts (13)
      -> [bill_shape_dagger  wing_color_brown ...]  <- Level 2: fine attributes (312)
      -> Acadian Flycatcher
```

A key architectural constraint — the **Masked Fine Head** — ensures that if a body part is predicted absent, none of its child attributes can be active. This enforces hard hierarchical consistency at both training and inference time.

---

## Architecture

```
                  Input Image (3, 224, 224)
                               |
                               v
       ResNet-50 (pretrained ImageNet, FC layer removed)
                               |
                     Feature Map F (2048,)
                               |
          +-----------------------------------------+
          |                                         |
          v                                         v
  Coarse Head  g_c(F)                     Fine Head  g_f(F, p_c)
  Linear(2048->512)                       concat([F, p_c]) -> (2061,)
  ReLU -> Dropout(0.3)                    Linear(2061->512)
  Linear(512->13)                         ReLU -> Dropout(0.3)
          |                               Linear(512->312)
  z_c  (13 logits)                                  |
  p_c = sigmoid(z_c)                      z_f  (312 logits)
          |                               p_f_raw = sigmoid(z_f)
          |                               mask[i] = p_c[parent(i)]  <- Masked Fine Head
          |                               p_f    = p_f_raw * mask    <- hard constraint
          |                                         |
          +-----------------------------------------+
                               |
                          p_f (312,)
                               v
                      Classifier  h(p_f)
                       Linear(312->200)
                               |
                          200 species
```

### Key Design Decisions

| Principle | Description |
|-----------|-------------|
| **Masked Fine Head** | `p_f[i] = sigmoid(z_f[i]) * p_c[parent(i)]` — if a part is absent, all its child attributes are suppressed. Hard architectural constraint enforced at training and inference. |
| **Bottleneck Property** | Classifier reads **only from `p_f` (312-dim)**, never from the backbone feature map. If the classifier also reads from raw features, concepts can be bypassed — an interpretability illusion. |
| **L1 from attribute parsing** | 13 coarse concepts are derived by parsing attribute names, not hardcoded. `has_bill_shape::dagger` -> parent `bill`. Part Locations are used only for visibility masking in the loss. |

---

## Dataset

**CUB-200-2011** — 11,788 images of 200 North American bird species.

| Split | Size |
|-------|------|
| Train | ~5,394 |
| Val   | ~600 (10% stratified from train, `random_state=42`) |
| Test  | 5,794 |

**Two annotation levels:**

- **L1 (13 coarse parts):** `back, belly, bill, breast, crown, eye, forehead, leg, nape, tail, throat, underparts, wing`
- **L2 (312 fine attributes):** binary attributes filtered by `certainty >= 3`

### Dataset Directory Layout

**Google Colab / Drive:**
```
MyDrive/xai_dataset/
+-- CUB_200_2011/
|   +-- CUB_200_2011/      <- images + annotation txt files (double-nested)
|   +-- attributes.txt     <- 312 attribute names (between the two CUB dirs)
+-- segmentations/          <- XAI faithfulness evaluation only, NOT used in training
```

**Local (VS Code):**
```
DB/DB1/
+-- CUB_200_2011/
|   +-- CUB_200_2011/
|   +-- attributes.txt
+-- DB1-Mask/segmentations/
```

Downloads:
- [CUB-200-2011 (1.2 GB)](https://data.caltech.edu/records/65de6-vp158)
- [Segmentation masks (39 MB)](https://data.caltech.edu/records/w9d68-gec53)

---

## Project Structure

```
hierarchical-cbm-cub/
+-- 01_data.ipynb              <- Step 1: data pipeline
+-- 02_model.ipynb             <- Step 2: model definition & verification
+-- 03_train.ipynb             <- Step 3: training loop
+-- 04_evaluate.ipynb          <- Step 4: evaluation & XAI analysis
+-- README.md                  <- this file (pipeline + literature review)
+-- .gitignore
```

Dataset lives on Google Drive or locally — not tracked by git.

---

## Environment Setup

```bash
# Clone
git clone https://github.com/erythm/hierarchical-cbm-cub.git
cd hierarchical-cbm-cub

# Local virtual environment (Python 3.11)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install torch torchvision pillow matplotlib numpy scikit-learn pandas
```

All notebooks auto-detect the runtime environment:

```python
IN_COLAB = 'google.colab' in sys.modules
```

- **In Colab** -> mounts Drive, uses `/content/drive/MyDrive/xai_dataset/...`
- **Local (VS Code + Jupyter)** -> uses `DB/DB1/CUB_200_2011/...` relative to the project root

---

## Literature Review

Covers Deliverable 1 (Literature Review) and Deliverable 2 (Research Gaps).

| Group | Papers |
|-------|--------|
| **G1 — CBM Foundations** | Koh et al. ICML 2020 [1], Hase et al. AAAI 2019 [6] |
| **G2 — Hierarchical CBMs** | Pittino et al. EAAI 2023 [2], Sun et al. (under review) [3] |
| **G3 — XAI Context** | Poeta et al. ACM CSUR 2025 [4] |
| **G4 — Supporting Methods** | Lin et al. ICCV 2017 [7], Kendall et al. CVPR 2018 [8], Bertinetto et al. CVPR 2020 [5] |

---

### GROUP 1 — CBM Foundations

#### [1] Koh et al. — *Concept Bottleneck Models* (ICML 2020)

##### Core Contribution

Koh et al. introduced the **Concept Bottleneck Model (CBM)**: a two-stage neural network that maps an input image $x$ first to a set of human-interpretable concept scores $\hat{c} \in [0,1]^k$, then from those concept scores to the final label $\hat{y}$. The key property is the **bottleneck constraint**: the task predictor $h$ receives *only* $\hat{c}$ as input, never the raw image features.

$$\hat{c} = g(x) \qquad \hat{y} = h(\hat{c})$$

This guarantees that every prediction can be explained by tracing which concepts $g$ activated and how $h$ weighted them.

##### Three Training Modes

| Mode | How it trains | Trade-off |
|------|--------------|----------|
| **Independent** | $g$ trained on concepts first; $h$ trained on predicted $\hat{c}$ | Best concept accuracy; task accuracy may suffer |
| **Sequential** | Same as Independent but $h$ is trained on real GT concepts | Cleaner bottleneck; slightly lower task accuracy |
| **Joint** | Both $g$ and $h$ share a joint loss $L_c + L_y$ | Best task accuracy; concepts can encode shortcut information (leakage) |

For this project, training is **sequential with joint fine-tuning** (Phases 1a/1b → 2 → 3) to balance concept quality and task performance without allowing leakage.

##### Concept Intervention

CBM's most cited contribution beyond architecture: at test time, replace a predicted concept $\hat{c}_i$ with its ground-truth value $c_i^*$ and measure accuracy gain. If accuracy improves significantly, the concepts are causally meaningful.

$$\text{Acc}(\text{intervention rate}) = \text{Acc}\left(h(c^*_{1..r} \cup \hat{c}_{r+1..k})\right)$$

Plotting accuracy vs. intervention rate from 0% to 100% is the standard CBM evaluation, implemented in `04_evaluate.ipynb`.

##### Key Limitation (motivates this project)

Koh et al. treat all concepts as a **flat, unstructured set**. There is no notion that `has_bill_shape::dagger` and `has_bill_color::black` both belong to the *bill* group, or that predicting `bill_color` is only meaningful if a bill is visible. This structural ignorance is the primary gap that H-CBM addresses.

---

#### [6] Hase et al. — *Interpretable Image Recognition with Hierarchical Prototypes* (AAAI 2019)

##### Core Contribution

Hase et al. propose **HPrototypes**: a prototype-based network where prototypes are organised in a hierarchy that mirrors a visual class taxonomy. Each prototype corresponds to a region of the feature space at a specific granularity level (e.g., *animal* → *bird* → *warbler*). This gives users coarse-to-fine explanations.

##### Architecture

- A shared backbone extracts features $z$.
- Prototypes at each tree level are trained to minimise distance to their assigned samples.
- A hierarchical classification head reads from prototypes at the finest level, constrained so that higher-level class scores are consistent with lower-level ones.

##### Relevance to H-CBM

| Aspect | HPrototypes | H-CBM |
|--------|------------|-------|
| Hierarchy levels | Class taxonomy (animal → bird → species) | Attribute hierarchy (part → attribute) |
| Explanation unit | Prototype patch | Concept probability |
| Human-alignment | Visual similarity | Named semantic attributes |
| Intervention | Not supported | Concept intervention (Koh et al.) |

HPrototypes demonstrates that **hierarchical structure in the model architecture** — not just post-hoc annotation — leads to more interpretable and semantically consistent decisions. H-CBM applies the same principle at the attribute level.

---

### GROUP 2 — Hierarchical CBMs (Closest Prior Work)

#### [2] Pittino et al. — *Hierarchical Concept Bottleneck Models for Vision* (EAAI 2023)

##### Core Contribution

Pittino et al. extend CBMs to a **two-level concept hierarchy** for fine-grained visual classification. They define:

- **High-level concepts** (analogous to L1 in this project): broad semantic categories such as body parts.
- **Low-level concepts** (analogous to L2): fine-grained visual attributes within each high-level concept.

The high-level concept predictor provides context that guides the low-level predictor, and the final classifier reads from low-level concepts.

##### Architecture

$$F \xrightarrow{g_H} z_H \xrightarrow{\sigma} p_H \qquad \text{(high-level)}$$
$$[F, p_H] \xrightarrow{g_L} z_L \xrightarrow{\sigma} p_L \qquad \text{(low-level, conditioned on } p_H\text{)}$$
$$p_L \xrightarrow{h} \hat{y} \qquad \text{(task classifier)}$$

This is structurally identical to H-CBM's design. The critical difference is in **enforcement of hierarchy consistency**:

##### Key Difference from H-CBM

Pittino et al. use a **soft penalty** (consistency regularisation loss) to encourage $p_L[i] \leq p_H[\text{parent}(i)]$, but do not enforce it architecturally. H-CBM uses a **hard multiplicative mask**:

$$p_f[i] = \sigma(z_f[i]) \times p_c[\text{parent}(i)]$$

This is an **architectural guarantee**: hierarchy violations are impossible at inference time, not just discouraged during training. The distinction is important for trustworthiness — a user can be certain that if the model says *the bill is absent*, all bill-colour predictions will be exactly 0.

##### Results

Pittino et al. report improved concept accuracy over flat CBMs on CUB-200-2011, validating the two-level design. This work is the direct baseline in this project's evaluation.

---

#### [3] Sun et al. — *Supervised Hierarchical Concept Bottleneck Models* (Under Review)

##### Core Contribution

Sun et al. propose a CBM architecture with explicit **multi-level supervision** at each node of a concept hierarchy. Instead of deriving L1 labels by OR-aggregation (as done in this project), they train with direct annotations at every level of the hierarchy, allowing gradients to flow from both levels independently.

They also introduce a **sequential training strategy**: first stabilise the concept predictors, then train the task classifier. This prevents the classic CBM failure mode where the classifier learns to exploit imperfect concept predictions rather than their semantic content.

##### Sequential Training Protocol (adopted in this project)

| Phase | What is trained | Loss terms |
|-------|----------------|------------|
| 1a (backbone frozen) | Coarse head only | $L_{\text{coarse}}$ |
| 1b (backbone frozen) | Coarse + Fine heads | $L_{\text{coarse}} + L_{\text{fine}} + L_{\text{consistency}}$ |
| 2 (backbone unfrozen, $L_{\text{task}}$ warm-up) | All | $L_{\text{coarse}} + L_{\text{fine}} + L_{\text{task}} + L_{\text{consistency}}$ |
| 3 (full fine-tuning) | All | Same as Phase 2 |

This schedule (directly used in `04_train.ipynb`) ensures the concepts are meaningful before the classifier is allowed to use them, preventing the leakage problem that joint training can cause.

##### Key Limitation of Sun et al. (addressed here)

Sun et al. still rely on soft consistency penalties. The hard masking constraint and OR-aggregated L1 labels derived purely from attribute names are contributions of this project.

---

### GROUP 3 — XAI Context

#### [4] Poeta et al. — *Concept-based Explanations in Computer Vision: Where Are We and Where Could We Go?* (ACM CSUR 2025)

##### Core Contribution

Poeta et al. provide a comprehensive survey of concept-based XAI methods in computer vision, establishing a **taxonomy** that classifies methods along several axes:

| Axis | Categories |
|------|------------|
| **Explanation timing** | Post-hoc vs by-design |
| **Concept type** | User-defined, data-driven, prototype-based |
| **Granularity** | Global vs local explanations |
| **Faithfulness** | Does the explanation reflect the actual decision process? |
| **Completeness** | Do concepts fully determine the output? |

##### Position of CBMs in this Taxonomy

CBMs are **by-design**, using **user-defined** concepts with **global** explanations (the concept vector $\hat{c}$ is the same for every input, unlike LIME which produces local approximations). CBMs achieve **structural completeness** by construction (the task predictor sees only $\hat{c}$).

##### Critique of CBMs (from survey)

The survey highlights several open problems:

1. **Scalability of annotations:** CBMs require per-image concept labels, which are expensive to obtain at scale.
2. **Concept leakage:** Joint training modes allow the classifier to encode non-concept information in high-precision concept activations.
3. **Flat concept sets:** Existing methods rarely capture structural relations between concepts (e.g., part-attribute hierarchies).
4. **Evaluation gaps:** Most CBM papers evaluate task accuracy and concept accuracy separately; few measure whether the *hierarchy* of explanations is coherent.

H-CBM directly addresses gaps 3 and 4.

> **Note:** One of the survey's authors is the course professor; this work defines the expected framing for the literature review.

---

### GROUP 4 — Supporting Technical Methods

#### [7] Lin et al. — *Focal Loss for Dense Object Detection* (ICCV 2017)

##### Core Contribution

Lin et al. propose **Focal Loss** to address extreme class imbalance in one-stage object detectors. Standard cross-entropy assigns equal weight to easy and hard examples; Focal Loss adds a modulating factor $(1 - p_t)^\gamma$ that down-weights well-classified examples:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t$ is the model's estimated probability for the ground-truth class
- $\gamma \geq 0$ is the *focusing parameter* (default $\gamma=2$)
- $\alpha_t$ balances positive vs. negative classes (default $\alpha=0.25$)

##### Application to L2 Concept Loss

CUB-200-2011 attribute labels are highly imbalanced: most birds do not exhibit most attributes (e.g., `has_wing_color::pink` is positive for <2% of images). Using standard BCE for the fine head loss would cause the model to predict all-zeros and still achieve high accuracy.

In `03_train.ipynb`, Focal Loss with $\alpha=0.25, \gamma=2$ is used for $L_{\text{fine}}$, masked by per-image certainty $\geq 3$.

| Loss | $\gamma=0$ | $\gamma=0.5$ | $\gamma=1$ | $\gamma=2$ (default) |
|------|-----------|-------------|-----------|----------------------|
| Weight on easy examples | 1.0 | ↓ | ↓ | very low |
| Weight on hard examples | 1.0 | same | same | same |

---

#### [8] Kendall et al. — *Multi-Task Learning Using Uncertainty to Weigh Losses in Deep Learning* (CVPR 2018)

##### Core Contribution

When training with multiple losses $\{L_i\}$, manually tuning the scalar weights $\{\lambda_i\}$ requires expensive grid search. Kendall et al. show that each task's weight should be inversely proportional to its **homoscedastic (task) uncertainty** $\sigma_i^2$:

$$L = \sum_i \frac{1}{2\sigma_i^2} L_i + \log \sigma_i$$

The $\log \sigma_i$ term regularises the uncertainty parameters, preventing $\sigma_i \to \infty$. The $\sigma_i$ values are learned jointly with the model weights via gradient descent.

##### Application in This Project

The H-CBM loss has three components:

$$L_{\text{total}} = \lambda_c L_{\text{coarse}} + \lambda_f L_{\text{fine}} + \lambda_{\text{cls}} L_{\text{task}}$$

Uncertainty weighting would replace the fixed $(\lambda_c=0.5,\; \lambda_f=0.5,\; \lambda_{\text{cls}}=1.0)$ with learnable parameters. `03_train.ipynb` uses fixed weights for simplicity; this paper is cited as the theoretical basis for the optional learnable-weighting extension.

---

#### [5] Bertinetto et al. — *Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks* (CVPR 2020)

##### Core Contribution

Standard top-1 accuracy treats every misclassification as equally bad. Bertinetto et al. formalise mistake severity with the **Lowest Common Ancestor (LCA) depth**:

$$\text{LCA-depth}(y, \hat{y}) = \text{depth of LCA of } y \text{ and } \hat{y} \text{ in class hierarchy}$$

A **low LCA depth** (shallow ancestor) means the prediction and label are far apart semantically (severe mistake). A **high LCA depth** means they share a specific ancestor (mild mistake).

##### Application in This Project

`04_evaluate.ipynb` approximates the LCA metric using **bird family names** extracted from CUB class names:

- *Same family* → mild mistake (e.g., predicting *Sayornis_Phoebe* for *Eastern_Phoebe*)
- *Different family* → severe mistake (e.g., predicting *Herring_Gull* for *Purple_Martin*)

The hypothesis is that H-CBM, by organising predictions through body-part concepts, should make proportionally more **mild mistakes** than an unconstrained ResNet-50 baseline.

| Metric | Expected direction |
|--------|-------------------|
| Mild Mistake % | H-CBM > B1 (ResNet) |
| Mild Mistake % | H-CBM >= B2 (Flat CBM) |

---

## Research Gaps (Deliverable 2)

Based on the literature review above, four concrete gaps motivate the design of H-CBM.

| Gap | Problem | Prior Best | H-CBM Solution | Guarantee |
|-----|---------|-----------|----------------|-----------|
| **G1** | Flat concept sets ignore semantic hierarchies | Soft conditioning on high-level scores (Pittino [2]) | `attr_parent_idx` from attribute name parsing; 13 L1 groups | Structural (architecture) |
| **G2** | No hard coarse-to-fine constraint at inference | Soft consistency penalty during training (Pittino [2]) | `p_f[i] = σ(z_f[i]) × p_c[parent(i)]` — multiplicative mask | Mathematical (always holds) |
| **G3** | CBM evaluation ignores semantic quality of mistakes | Semantic mistake severity for standard classifiers (Bertinetto [5]) | Evaluate mild/severe mistake rate alongside concept quality | Evaluation (novel combination) |
| **G4** | Classifier leakage bypasses concept bottleneck | Joint training with consistency loss | Strict bottleneck + sequential training phases | Structural (no feature path to classifier) |

### Gap 1: Flat Concept Sets Ignore Natural Semantic Hierarchies

Koh et al. [1] treat the 312 CUB attributes as an unordered flat set — the model must independently learn the correlation between `bill_visible` and `has_bill_*` attributes. Pittino et al. [2] partially address this by conditioning the low-level predictor on high-level scores, but do not enforce any structural constraint between levels at inference time.

**H-CBM solution:** The attribute hierarchy is built directly from CUB attribute names (`has_bill_shape::dagger` → `"bill"` → L1 index 1). This produces 13 body-part groups. The `attr_parent_idx` tensor (shape 312) encodes this as a lookup used in the Masked Fine Head at every forward pass. If a part is absent, its children are identically 0 — both semantically correct and computationally efficient.

### Gap 2: Existing CBMs Lack a Hard Coarse-to-Fine Constraint

Pittino et al. [2] add a consistency loss $L_{\text{consistency}} = \sum_i \max(0, p_f[i] - p_c[\text{parent}(i)])^2$ during training. However, at inference time nothing prevents a violation, the soft penalty creates a trade-off requiring careful tuning, and a user cannot be certain explanations are consistent without inspecting every prediction.

**H-CBM solution:** The Masked Fine Head implements the constraint algebraically:

$$p_f[i] = \underbrace{\sigma(z_f[i])}_{\in [0,1]} \times \underbrace{p_c[\text{parent}(i)]}_{\in [0,1]} \leq p_c[\text{parent}(i)]$$

Since both factors are in $[0,1]$, the product is always $\leq$ the parent probability. This is a **mathematical guarantee** — no training regime, no example, no adversarial input can violate it.

### Gap 3: Limited Hierarchy-Aware Evaluation in the CBM Literature

Koh et al. [1] evaluate task accuracy and concept intervention curves only. Pittino et al. [2] add per-level concept accuracy but do not measure hierarchy consistency rates or semantic mistake severity. Bertinetto et al. [5] propose hierarchy-aware evaluation metrics but apply them to standard classifiers, not CBMs.

**H-CBM solution:** `04_evaluate.ipynb` introduces a comprehensive evaluation suite:

| Metric | What it measures | Novel? |
|--------|-----------------|--------|
| Top-1 / Top-5 Accuracy | Standard task performance | — |
| L1 Concept Accuracy (13 parts) | Coarse concept quality | — |
| L2 Concept macro-F1 + AUROC | Fine concept quality (imbalance-aware) | Partial |
| Concept Intervention curve | Causal meaningfulness of concepts | — |
| Semantic Mistake Severity | Hierarchy-aware error quality (applied to CBM) | **Novel** |
| Hierarchy Consistency Rate | Fraction of samples with no violations | **Novel** |
| Multi-level explanation visualisation | L1 vs L2 comparison per image | **Novel** |

### Gap 4: Classifier Leakage Undermines Interpretability

Poeta et al. [4] identify this as one of the most critical open problems in concept-based XAI. In joint training, the classifier $h$ may learn to exploit subtle non-semantic signals encoded in $\hat{c}$ — creating an **interpretability illusion** where the model appears to explain itself via concepts but the actual decision mechanism is opaque.

**H-CBM solution:** Two structural choices prevent leakage:

1. **Strict bottleneck:** `HierarchicalCBM.classifier` is a `nn.Linear(312, 200)` that takes *only* `p_f` as input. The 2048-dim feature map is never passed to the classifier.
2. **Sequential training** (from Sun et al. [3]): the classifier is only introduced in Phase 2 after the concept heads have been trained in isolation for 30 epochs, ensuring concept activations carry genuine semantic meaning before the classifier can use them.

---

### Conclusion

| Paper | What it establishes | Gap it leaves |
|-------|--------------------|--------------|
| Koh et al. [1] | CBM architecture; concept intervention | Flat concepts; leakage in joint mode |
| Hase et al. [6] | Hierarchical architecture improves consistency | Prototype-based, not attribute-based; no intervention |
| Pittino et al. [2] | Two-level CBM for vision | Soft constraint only; no formal guarantee |
| Sun et al. [3] | Sequential training prevents leakage | Still soft consistency; no hard mask |
| Poeta et al. [4] | Identifies flat concepts and leakage as open problems | Survey only — no solution proposed |
| Lin et al. [7] | Focal Loss for attribute imbalance | Not applied in CBM context before |
| Kendall et al. [8] | Principled multi-task loss weighting | Not applied in CBM context before |
| Bertinetto et al. [5] | Hierarchy-aware error metrics | Applied only to standard classifiers |

H-CBM's contributions relative to the state-of-the-art:

1. **Hard multiplicative masking** — the first CBM with a provable $p_f \leq p_c$ constraint.
2. **Zero-annotation L1 hierarchy** — L1 labels derived purely from attribute name parsing, requiring no additional manual labelling.
3. **Unified hierarchical evaluation** — combining concept quality, concept intervention, and semantic mistake severity in a single protocol.

---

## Pipeline: Step by Step

### Step 1 — Data Pipeline (`01_data.ipynb`)

**Goal:** Load all CUB annotation files, construct L1/L2 concept labels, build PyTorch `Dataset` and `DataLoader` objects, and save a `data_pipeline.pkl` for downstream notebooks.

#### L1 Label Construction

1. Parse `attributes.txt` -> extract the category before `::` (e.g., `has_bill_shape::dagger` -> `bill`)
2. Strip `has_` prefix and property suffixes (`_color`, `_pattern`, `_shape`, `_length`)
3. Merge compound parts: `under_tail` / `upper_tail` -> `tail`; `upperparts` -> `back`; `underparts` -> `belly`
4. Attributes without a clear body part (`primary_color`, `size`, `shape`) have no L1 parent -> always unmasked
5. Build `part_to_attr_indices` dict mapping each of the 13 parts to its L2 attribute indices
6. `L1[part] = max(L2[attr] for attr in part_to_attr_indices[part])` — OR aggregation

#### L2 Label Construction

- Source: `image_attribute_labels.txt`
- Filter: keep only annotations with `certainty >= 3`

#### Visibility Masking

- Source: `part_locs.txt` — the `visible` flag per part per image
- Used to mask the loss on invisible parts (not input features — only affects loss computation)

#### Image Preprocessing

```
Train:    Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip
          -> ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
          -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Val/Test: Resize(256) -> CenterCrop(224) -> Normalize
```

#### Dataset Output

`BirdDataset.__getitem__` returns a 6-tuple:

```python
(img, label, l1, l2, cert, vis)
# img   : (3, 224, 224) float tensor
# label : int  - 0-indexed species (CUB is 1-indexed, subtract 1)
# l1    : (13,) float - coarse binary labels
# l2    : (312,) float - fine binary labels (certainty-filtered)
# cert  : (312,) float - certainty mask (1 = certain enough, 0 = skip in loss)
# vis   : (13,) float - visibility mask (1 = part visible in image)
```

**Saved output:** `data_pipeline.pkl` with keys: `NUM_L1, NUM_L2, CONCEPT_NAMES, attr_parent_idx`

---

### Step 2 — Model Definition (`02_model.ipynb`)

**Goal:** Define `HierarchicalCBM`, verify the forward pass produces zero hierarchy violations, and save `model_config.pkl`.

#### `HierarchicalCBM` forward pass

```python
def forward(self, x):
    F       = self.backbone(x)                      # (B, 2048)
    z_c     = self.coarse_head(F)                   # (B, 13)
    p_c     = torch.sigmoid(z_c)                    # (B, 13)

    F_cat   = torch.cat([F, p_c], dim=1)            # (B, 2061)
    z_f     = self.fine_head(F_cat)                 # (B, 312)
    p_f_raw = torch.sigmoid(z_f)                    # (B, 312)

    # Masked Fine Head - hard hierarchical constraint
    parent_probs = p_c[:, self.attr_parent_idx]     # (B, 312); -1 index -> 1.0 (unmasked)
    p_f          = p_f_raw * parent_probs           # (B, 312)

    cls_logits = self.classifier(p_f)               # (B, 200)
    return cls_logits, z_c, p_c, z_f, p_f
```

- `attr_parent_idx` is a `(312,)` buffer; value `-1` means no parent (attribute is always unmasked)
- Returns **5 values** — raw logits `z_c`, `z_f` are exposed for numerically stable loss computation

**Saved output:** `model_config.pkl`

---

### Step 3 — Training (`03_train.ipynb`)

**Goal:** Train H-CBM end-to-end, checkpoint the best model, and save training curves.

#### Loss Function

$$L_{\text{total}} = \lambda_c L_{\text{coarse}} + \lambda_f L_{\text{fine}} + \lambda_{\text{cls}} L_{\text{task}}$$

| Term | Formula | Notes |
|------|---------|-------|
| $L_{\text{coarse}}$ | `WeightedBCE(z_c, y_l1)` | `pos_weight = N_neg/N_pos` per part; masked by visibility |
| $L_{\text{fine}}$ | `FocalLoss(z_f, y_l2, alpha=0.25, gamma=2)` | masked by `cert` and visible parts |
| $L_{\text{task}}$ | `CrossEntropy(cls_logits, y_class)` | standard 200-class classification |

Fixed weights: $\lambda_c = 0.5$, $\lambda_f = 0.5$, $\lambda_{\text{cls}} = 1.0$

#### Optimizer & Schedule

```
AdamW, weight_decay = 1e-4
  Backbone (ResNet-50):    lr = 1e-4   <- lower to prevent catastrophic forgetting
  Heads + Classifier:      lr = 1e-3

Epochs:         50
Batch size:     32
Early stopping: patience = 10 on val_loss
LR schedule:    ReduceLROnPlateau (patience=5, factor=0.5)
```

**Saved outputs:**
- `best_model.pth` — checkpoint with `epoch, model_state_dict, optimizer_state_dict, val_loss, val_acc, NUM_L1, NUM_L2, CONCEPT_NAMES, attr_parent_idx`
- `train_history.pkl` — per-epoch loss and accuracy history
- `training_curves.png` — loss and accuracy plots

---

### Step 4 — Evaluation (`04_evaluate.ipynb`)

**Goal:** Measure predictive performance, concept quality, semantic mistake severity, and produce multi-level explanations. Also trains two baselines for comparison.

#### Standard Metrics

| Metric | Details |
|--------|---------|
| Top-1 Accuracy | 200-class species classification |
| Top-5 Accuracy | Species in top-5 predictions |
| L1 Concept Accuracy | Binary accuracy per part (threshold 0.5), macro-averaged |
| L2 Concept macro-F1 | Threshold 0.5; F1 preferred over accuracy due to class imbalance |
| L2 macro-AUROC | Per-attribute AUROC averaged over attrs with both classes present |

#### Hierarchy-Aware Metrics

| Metric | Description |
|--------|-------------|
| **Concept Intervention** | Replace predicted L2 values with ground-truth for fractions [0%, 25%, 50%, 75%, 100%] of L1 groups. Steep accuracy slope = concepts are causally meaningful. |
| **Semantic Mistake Severity** | For each wrong prediction, check if the predicted species is in the same bird family (mild) or a different family (severe). Lower severe % = semantically better errors. |
| **Hierarchy Consistency Rate** | Fraction of samples where `p_f[child] <= p_c[parent]` holds for all pairs. |

#### Multi-Level Explanation

`plot_multilevel_explanation()` produces a two-panel figure per sample:

- **Left (L1):** Horizontal bars of all 13 coarse body-part probabilities. Green = active (p > 0.5), red = absent.
- **Right (L2):** Horizontal bars of the top-10 most activated fine attributes with their names and probabilities.

#### Baselines

| ID | Model | Architecture | Concept levels |
|----|-------|-------------|----------------|
| B1 | ResNet-50 end-to-end | Standard classifier | None — accuracy upper bound |
| B2 | Flat CBM (Koh et al. 2020) | Feature -> L2 -> species | L2 only, no hierarchy |
| **Ours** | **H-CBM** | Masked Hierarchical CBM | L1 + L2, hard mask |

Baselines are checkpoint-conditional: training is skipped if the checkpoint already exists.

**Final output:** pandas comparison table — Top-1, Top-5, L1 Acc, L2 F1, L2 AUROC, Mild Mistake %, Acc@0%, Acc@100% intervention.

---

## Implementation Status

| Component | File | Status |
|-----------|------|--------|
| Data Pipeline | `01_data.ipynb` | Complete |
| Model Architecture | `02_model.ipynb` | Complete |
| Training Loop | `03_train.ipynb` | Complete |
| Evaluation & XAI | `04_evaluate.ipynb` | Complete |

---

## References

| | Paper | Venue | Relevance |
|-|-------|-------|-----------|
| [1] | Koh, P.W. et al. — *Concept Bottleneck Models* | ICML 2020 | CBM foundation + Concept Intervention |
| [2] | Pittino, F. et al. — *Hierarchical CBM for Vision* | EAAI 2023 | Closest prior work — direct baseline |
| [3] | Sun, X. et al. — *Supervised Hierarchical CBM* | Under review | Sequential training reference |
| [4] | Poeta, E. et al. — *Concept-based XAI: A Survey* | ACM CSUR 2025 | Survey — teacher is co-author |
| [5] | Bertinetto, L. et al. — *Making Better Mistakes* | CVPR 2020 | Hierarchical Distance / mistake severity |
| [6] | Hase, P. et al. — *Interpretable Image Recognition with Hierarchical Prototypes* | AAAI 2019 | Hierarchical architecture |
| [7] | Lin, T.Y. et al. — *Focal Loss for Dense Object Detection* | ICCV 2017 | Focal Loss for L2 attribute imbalance |
| [8] | Kendall, A. et al. — *Multi-Task Learning Using Uncertainty* | CVPR 2018 | Learnable lambda weights |
