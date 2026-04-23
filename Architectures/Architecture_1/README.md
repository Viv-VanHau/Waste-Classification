## Architecture 1: Single-Stream CNN Baseline

- **Role:** Primary CNN baseline for the study

- **Design Paradigm:** Monolithic, single-stream architecture

- **Core Task:** Simultaneous extraction and resolution of macro-level features (material classification) and micro-level features (quality grading)

### Model Configuration & Setup

- **Backbone:** MobileNetV2

- **Pipeline:** Input (224 x 224 x 3) → MobileNetV2 Feature Extractor → Dense Layer → 10-Class Softmax Output

- **Inference Setup:** Batch size strictly constrained to 1 → Simulates real-world edge deployment where waste items are processed sequentially (frame-by-frame) on a conveyor belt

*Architecture 1 dual-evaluation metrics*
<img width="1886" height="936" alt="image" src="https://github.com/user-attachments/assets/8a60ad2d-b17b-40e9-aaa1-0862074f35c5" />

### System Performance Metrics

- **Average Latency:** 107.92 ms / image

- **Frame Rate:** 9.27 FPS

- **Operational Throughput:** ~556 images / min

Performance was evaluated under two conditions to isolate the specific impact of the quality grading task:

| Evaluation Scope                  | Target Classes | Accuracy |
|----------------------------------|---------------|----------|
| Full Task (Material + Grading)   | 10 Classes    | 77.10%   |
| Pure Material Classification     | 8 Classes     | 89.70%   |
| **Performance Gap (Δ)**          | —             | **+12.60%** |

### Key Findings

**1. Strong Material Identification Capability**

The CNN effectively captures global structural features across distinct categories:

- Battery: Achieved an exceptional F1-score of 0.9375

- Paper/Cardboard: Reached a near-perfect Recall of 0.9900

→ Standard CNNs are highly reliable and efficient for coarse-grained pure material recognition (~90% accuracy)

**2. Critical Failure in Quality Grading**

The 12.60% absolute accuracy drop exposes a fundamental weakness in handling fine-grained intra-class differences

- Metric Collapse: Metal Grade A suffered a catastrophic Recall drop to just 0.0900

- Model Bias: The network exhibits severe prediction bias, consistently over-predicting Grade B while under-detecting Grade A

### Root Causes for Grading Failure

**Class Imbalance Resilience:** While class weighting techniques mitigated some bias, the extreme structural data deficit (Grade B: >7,000 samples vs. Grade A: ~200 samples) proved too severe for a standard CNN feature extractor → The model statistically skewed toward the dominant class.

**Semantic Complexity vs. CNN Limitations:** Quality grading requires strict, multi-conditional logic. A "Metal Grade A" (e.g., an aluminum can) must be fully open, uncrushed, and clean. A single defect degrades it to Grade B. CNNs rely on local spatial features (edges/textures) and lack the contextual awareness for this "all-or-nothing" visual reasoning

**Feature Interference:** Forcing a single-stream network to learn both broad material boundaries and subtle intra-class surface variations simultaneously causes representation dilution and a loss of discriminative power

### Architectural Implications

**Task Decoupling:** Classification and quality grading require distinct levels of feature extraction and cannot be optimally solved within a single monolithic model.

**Justification for Future Architectures:** The explicit failure of the baseline CNN at the grading level mathematically validates the necessity of transitioning to multi-stage pipelines and integrating Vision-Language Models (VLMs) to handle complex semantic reasoning in subsequent architectures.
