# SVTRv2X: Training, Validation, and Evaluation Pipeline Deep-Dive

## 1. Executive Summary

When you execute:
```bash
python tools/train_rec.py \
  -c configs/rec/svtrv2/svtrv2x_tiny_mixed_latin_indic.yml \
  -o Global.checkpoints=...
```

You are running a **hybrid Training + Online Validation (`train_eval`) pipeline** for the **SVTRv2X Tiny** scene text recognition model.

### In Short:
* **Is it training?** **Yes.** It is actively updating ~13.9 million neural network weights across 3 epochs using backward backpropagation and the AdamW optimizer.
* **Is it testing?** **Yes, periodically.** Every **2,000 steps**, training pauses briefly to run inference on a validation test set without computing gradients.
* **What is it finding out?** It determines whether the network is learning generalizable visual representations for both **Latin and Indic text**, whether accuracy is improving, and automatically persists the highest-scoring model (`best.pth`).
* **What is the final outcome?** A compact, high-throughput OCR model checkpoint that can recognize words across English and 12 Indian languages in complex, real-world scene text (curved, blurry, low-resolution, irregular).

---

## 2. What Pipeline is It Running?

The pipeline connects three core components:

```
[Training Data: 60% Latin + 40% Indic]
               │
               ▼
[Data Augmentations: PARSeqAug (distort, blur, jitter)]
               │
               ▼
[SVTRv2X Backbone: Conv + Local/Global Attention + MoE Experts]
               │
       ┌───────┴───────────────────────┐
       ▼                               ▼
[CTC Decoder (Fast Alignment)]  [GTC / SMTR Decoder (Autoregressive)]
       │                               │
       └───────┬───────────────────────┘
               ▼
   [Multi-Task Loss Calculation]
               │
   [Backward Pass + Optimizer Step]
               │ (Every 2,000 steps)
               ▼
   [Online Evaluation on Test Set] ──> Updates `best.pth`
```

### A. Data Mixing & Sampling (Input)
* **Dataset Mix**:
  * **60% Latin**: Drawn from `Union14M-L` across 5 difficulty levels (`easy`, `normal`, `medium`, `hard`, `challenging`).
  * **40% Indic**: Drawn from `IndicSTR12` across 12 Indian language scripts (Devanagari, Bengali, Telugu, Tamil, Gujarati, Malayalam, Kannada, Odia, Gurmukhi, etc.).
* **Sampling Engine (`MixedScriptRatioSampler`)**:
  * Guarantees that every mini-batch of 64 images preserves the exact 60:40 script ratio.
  * Buckets images by aspect ratio (`128x32`) to minimize padding distortion.

### B. Network Architecture (`SVTRv2X Tiny`)
* **Feature Extraction**: 3 hierarchical stages with dimension widths `[64, 128, 256]` and depths `[3, 6, 3]`.
* **Hybrid Mixers**: Uses 3x3 local convolutions for early stroke features, followed by Windowed Local Attention and Global Self-Attention for character contextual modeling.
* **Mixture of Experts (MoE)**: 4 parallel expert networks with dynamic top-1 routing, allowing individual experts to specialize in distinct script topologies (e.g. Latin letters vs. Indic complex conjuncts).
* **Self-Distillation Module (SDM)**: Aligns intermediate feature representations to transfer high-level semantic knowledge directly down into earlier stages.
* **Dual Decoders**:
  1. **CTC Decoder (`RCTCDecoder`)**: Independent frame-by-frame character classifier.
  2. **GTC Decoder (`SMTRDecoder`)**: Sub-string masked transformer decoder that models character sequences and bi-directional context.

### C. Multi-Task Optimization (Loss Function)
During every step, the total loss is computed as:
$$\mathcal{L}_{\text{total}} = 0.1 \cdot \mathcal{L}_{\text{CTC}} + 1.0 \cdot \mathcal{L}_{\text{GTC}} + 0.5 \cdot \mathcal{L}_{\text{SDM}} + 0.01 \cdot \mathcal{L}_{\text{MoE\_Load}}$$
* **$\mathcal{L}_{\text{CTC}}$**: Ensures CTC alignment convergence.
* **$\mathcal{L}_{\text{GTC}}$**: Cross-entropy over predicted character sub-strings.
* **$\mathcal{L}_{\text{SDM}}$**: Minimizes distance between intermediate stage projections.
* **$\mathcal{L}_{\text{MoE\_Load}}$**: Balances the workload across all 4 experts so no single expert starves or dominates.

---

## 3. Metrics: What Do They Mean?

Every 50 training steps, and during every 2,000-step validation interval, the pipeline logs performance metrics:

| Metric Name | Full Name | Explanation |
| :--- | :--- | :--- |
| **`acc`** | Exact Word Match Accuracy | The percentage of words where **100% of characters** match ground truth exactly. (If a 10-letter word has 9 letters right and 1 wrong, `acc = 0`). |
| **`norm_edit_dis`** | Normalized Edit Distance (NED) | Measures fine-grained character-level accuracy using Levenshtein distance: $\text{NED} = 1 - \frac{\text{Distance}(y, \hat{y})}{\max(\|y\|, \|\hat{y}\|)}$. If 9 of 10 letters are correct, $\text{NED} = 0.90$. |
| **`gtc_acc`** | GTC Branch Word Accuracy | Exact word accuracy produced specifically by the autoregressive SMTR transformer decoder. |
| **`gtc_norm_edit_dis`** | GTC Branch NED | Normalized Edit Distance produced specifically by the SMTR transformer decoder. |
| **`num_samples`** | Evaluation Sample Count | The number of images evaluated in that validation cycle (e.g. 933 images). |
| **`fps`** | Inference Throughput | Frames per second processed during evaluation (measures runtime inference speed). |

---

## 4. Unified Measure vs. Section-by-Section Breakdown

### During Training (Online Validation):
* **It gives a unified aggregate metric.**
* In `configs/rec/svtrv2/svtrv2x_tiny_mixed_latin_indic.yml`, the validation section evaluates on two benchmark sets combined:
  * `evaluation/CUTE80` (curved text images)
  * `evaluation/SVTP` (perspective-distorted street-view text)
  * Total: **933 images**.
* **Why unified during training?** 
  Running an exhaustive per-dataset, per-language evaluation on tens of thousands of test images every 2,000 steps would slow down training drastically. The 933-image unified subset acts as a fast benchmark to monitor if the model is improving without introducing overhead.

### Post-Training (Standalone Benchmark Evaluation):
Once training completes (or using any checkpoint like `best.pth`), you can run evaluation broken down into **individual sections**:

1. **Section-by-Section Benchmark Script (`tools/eval_rec_all_en.py`)**:
   Outputs a breakdown across individual benchmarks:
   * **Standard Benchmarks**: IIIT5K, SVT, IC13, IC15, SVTP, CUTE80.
   * **Challenging Benchmarks**: Curve, Multi-Oriented, Artistic, Contextless, Salient, Low-Resolution.
2. **Dedicated Indic Evaluation**:
   By pointing the evaluation config to individual language splits (e.g., Hindi test, Tamil test, Bengali test), it outputs separate accuracy and NED tables for each language and script.

---

## 5. What is the Final Outcome?

1. **Trained Checkpoint Artifacts**:
   * **`best.pth`**: The best checkpoint achieving peak accuracy across the 3 epochs (e.g., $\ge 68\%+$).
   * **`latest.pth`**: The most recent checkpoint (saved every 500 steps) enabling immediate resumption if Colab disconnects.
2. **Deployable OCR Model**:
   * A unified model capable of recognizing mixed Latin and Indic words without needing separate models for different scripts.
   * Compatible with ONNX export (`tools/toonnx.py`) for real-time mobile/CPU/GPU deployment.

