# Improvement Plan — Paraphrase Semantic Analysis

## Current Status (Phase 2 Complete)

| Benchmark | Score |
|:----------|:-----:|
| MRPC Accuracy | **80.88%** |
| MRPC F1 | **0.87** |
| 50-Example Test | **84% (42/50)** |

**Architecture**: Bi-Encoder (`all-mpnet-base-v2`, fine-tuned) + Cross-Encoder (`stsb-distilroberta-base`) with weighted ensemble scoring (35% Bi / 65% Cross, threshold 0.70).

### Known Limitations

| Failure Type | Count | Examples | Root Cause |
|:-------------|:-----:|:---------|:-----------|
| Idiom / Figurative language | 1 | "raining cats and dogs" ↔ "pouring outside" | Model lacks idiom knowledge |
| Implicit vocabulary reasoning | 2 | "trilingual" ↔ "speaks three languages" | Rare synonym gap |
| Causal / Implicit reasoning | 2 | "too sweet" ↔ "too much sugar" | Bi-Encoder can't reason across sentences |
| Word-order swap (subject/object) | 2 | "dog bit man" ↔ "man bit dog" | Bi-Encoder is order-insensitive |
| Structural similarity trap | 1 | "I am hungry" ↔ "I am thirsty" | Same structure, different meaning |

---

## Phase 3: Potential Future Improvements

### Option A: Better Loss Function (Low Effort, Medium Impact)

**What**: Replace `ContrastiveLoss` with `CosineSimilarityLoss` during training.

**Why**: The current training uses ContrastiveLoss which only distinguishes 0/1 (paraphrase or not). CosineSimilarityLoss learns graded similarity from STS-B (0.0 to 1.0), producing better-calibrated scores that work more naturally with our weighted ensemble.

**Expected improvement**: +2-4% on MRPC, better calibration on borderline pairs.

**Files to modify**:
- `backend/training/train_v2.py` — change loss function
- No changes to inference pipeline

```python
# In train_v2.py, replace:
train_loss = losses.ContrastiveLoss(model=model)
# With:
train_loss = losses.CosineSimilarityLoss(model=model)
```

---

### Option B: Knowledge Distillation from Large Teacher (Medium Effort, High Impact)

**What**: Use a large Cross-Encoder as a "teacher" to generate soft labels for training.

**Why**: Instead of training on binary 0/1 labels, the student model learns nuanced similarity scores from a model that understands context deeply (e.g., `cross-encoder/stsb-roberta-large`). This helps with implicit reasoning cases like "too sweet" ↔ "too much sugar".

**Expected improvement**: +5-8% on MRPC, significant improvement on vocabulary gaps.

**Files to modify/create**:
- `backend/training/train_v3_distillation.py` — new training script

```python
from sentence_transformers import SentenceTransformer, losses

# 1. Load teacher (large model, used only during training)
teacher_model = CrossEncoder('cross-encoder/stsb-roberta-large')

# 2. Generate soft labels for all training pairs
for t1, t2 in training_pairs:
    soft_label = teacher_model.predict([t1, t2]) / 5.0  # normalize to 0-1
    # Replace binary label with teacher's continuous score

# 3. Train student with CosineSimilarityLoss on soft labels
student = SentenceTransformer('all-mpnet-base-v2')
train_loss = losses.CosineSimilarityLoss(model=student)
```

**Compute**: ~30 min on Mac M-series (teacher inference on ~50k pairs + 1 epoch student training).

---

### Option C: Add Idiom/Figurative Training Data (Low Effort, Targeted Impact)

**What**: Add idiom paraphrase pairs to the training data.

**Why**: The model has never seen idioms during training (QQP/MRPC/MNLI don't contain many). Adding even 200-500 idiom→literal pairs would help.

**Data source**: Manually curated or from datasets like PPDB (Paraphrase Database) or IMPLI (Idiomatic Paraphrase Dataset).

**Example training pairs**:
```
"It's raining cats and dogs"  ↔  "It is raining very heavily"       → 1.0
"Break a leg"                 ↔  "Good luck"                        → 1.0
"Piece of cake"               ↔  "Very easy"                        → 1.0
"He kicked the bucket"        ↔  "He died"                          → 1.0
"Let the cat out of the bag"  ↔  "Revealed a secret"                → 1.0
```

**Files to modify**:
- `backend/training/train_v2.py` — add idiom dataset loading section

---

### Option D: Cross-Encoder Only Mode (High Effort, Highest Impact)

**What**: For maximum accuracy, use the Cross-Encoder alone (no Bi-Encoder) at inference time.

**Why**: Cross-Encoders see both sentences together and can reason about word order, negation, and implicit meaning. This would fix the word-swap problem entirely.

**Tradeoff**: ~10x slower inference (200ms → 2s per pair) because Cross-Encoders can't pre-compute embeddings.

**Implementation**:
```python
def predict_similarity_cross_only(text1, text2):
    score = _cross_model.predict([text1, text2])
    score = max(0.0, min(1.0, float(score)))
    is_paraphrase = score > 0.70
    return score, is_paraphrase
```

**When to use**: If inference speed doesn't matter (batch processing, offline analysis).

---

## Recommended Priority

| Priority | Option | Effort | Impact | Risk |
|:--------:|:------:|:------:|:------:|:----:|
| 1 | A (Better Loss) | 🟢 Low | Medium | Low |
| 2 | C (Idiom Data) | 🟢 Low | Targeted | Low |
| 3 | B (Distillation) | 🟡 Medium | High | Medium |
| 4 | D (Cross-Encoder Only) | 🟡 Medium | Highest | Speed tradeoff |
