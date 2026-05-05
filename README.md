# 🔍 DeepLens
### Explainable Deepfake Detection

Detecting AI-generated images isn't enough — DeepLens tells you **why** an image is fake, in plain language.

**[Try the demo notebook →](notebooks/03_explanation_generation_evaluation.ipynb)**

## What is DeepLens?

A two-stage pipeline that combines computer vision and NLP to produce auditable deepfake detections.

```
Image → ViT-Base/16 → "FAKE 92%" + attention map
                              ↓ (convert to spatial language)
              "focused on upper-left and center"
                              ↓
        Qwen2.5-VL-7B → natural language explanation
```

Most deepfake detectors are black boxes — they output a verdict but never explain *why*. DeepLens flags an image, identifies which regions the detector focused on, and uses a vision-language model to describe the specific artifacts it sees in those regions.

**Example output:**
> **Prediction: FAKE (87.2% confidence)**
> *Attention focus: upper-left and center regions of the image*
>
> The skin texture of the monkey's face looks somewhat unnatural. There are areas where the texture seems overly smooth or lacks the fine details one would expect from a real animal. The lighting is quite even, with no harsh shadows or highlights that would naturally occur in a real photograph. The blending between the fur and the banana is not entirely seamless...

## Performance

### Detection (ViT-Base/16, fine-tuned on 25k images)

| Metric | Value |
|---|---|
| Test Accuracy | **0.9284** |
| F1 Score | 0.9288 |
| AUROC | 0.9824 |
| Precision | 0.9262 |
| Recall | 0.9314 |

### Explanation Quality (LLM-as-Judge: Qwen2.5-7B, 100 images × 4 prompts)

| Prompt Strategy | Specificity | Plausibility | Grounding | **Overall** |
|---|---|---|---|---|
| Zero-shot | 3.40 | 4.40 | 2.74 | 3.513 |
| Chain-of-Thought | 3.74 | 4.75 | 3.28 | 3.923 |
| Few-shot taxonomy | 2.95 | 4.06 | 2.43 | 3.147 |
| **Attention-grounded** ✓ | **3.94** | **4.85** | **4.09** | **4.293** |

**+36% over the few-shot baseline. +68% on the grounding dimension specifically** — direct evidence that injecting the detector's spatial attention as language helps the VLM produce spatially-anchored explanations.

## Quick Start

```bash
git clone https://github.com/AliAI11/DeepLens.git
cd DeepLens/notebooks
```

Run the notebooks in order on Colab Pro (A100 recommended):

1. `01_dataset_exploration_preprocessing.ipynb` — download NTIRE shard, build 80/10/10 splits
2. `02_train_vit_detector.ipynb` — fine-tune ViT-Base/16, extract attention maps
3. `03_explanation_generation_evaluation.ipynb` — generate explanations across 4 prompt strategies, run LLM-as-judge eval, run interactive demo

Total runtime end-to-end: ~2 hours on A100.

## Technical Details

**Detector:** `google/vit-base-patch16-224` fine-tuned for binary classification with bf16 mixed precision, AdamW, linear warmup + decay, 5 epochs.

**Dataset:** [NTIRE-2026 Robust AI-Generated Image Detection](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train) (MSU). 25,000 images sampled balanced (12.5k real / 12.5k fake), split 80/10/10. *Note: the official validation set is unlabeled (competition leaderboard), so we carved a held-out test split from the labeled training data.*

**Explainer:** [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), zero-shot in bf16. Attention maps are converted to spatial natural language ("the upper-left and middle-center regions") and injected into the prompt.

**Evaluator:** [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as LLM-as-judge, scoring 1–5 on three dimensions: specificity, plausibility, grounding.

**Prompt strategies compared:** zero-shot, chain-of-thought, few-shot artifact taxonomy, and attention-grounded (the novel contribution).

## Limitations

- Single-judge evaluation — cross-judge validation across model families left for future work
- LLM judge cannot see the image, so it scores how plausible explanations *sound*, not factual correctness against the image
- 14×14 attention resolution limits fine-grained spatial grounding
- No evaluation on unseen generators (cross-generator generalization)

## Repository Structure

```
DeepLens/
├── notebooks/
│   ├── 01_dataset_exploration_preprocessing.ipynb
│   ├── 02_train_vit_detector.ipynb
│   └── 03_explanation_generation_evaluation.ipynb
├── data/
│   ├── splits.csv
│   ├── test_predictions.csv
│   ├── explanations.csv
│   ├── judge_qwen.csv
│   └── results_qwen.csv
├── proposal/
│   └── NLP_ProjectProposal.pdf
└── presentation/
    └── DeepLens_slides.pdf
```

Built as a final project for **Virginia Tech CS 5624 – Natural Language Processing (Spring 2026)** by [Kean Jaldin Guzman](mailto:keanjg28@vt.edu) and [Afeef Ali](mailto:aali11@vt.edu).
