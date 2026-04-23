## Architecture 2: Dual-Stage Sequential CNN

- **Role:** First iteration of a multi-stage pipeline, designed to decouple base material filtering from quality grading

- **Design Paradigm:** Hierarchical, conditional routing pipeline

- **Core Task:** Utilize a lightweight primary model to identify base materials, and conditionally route complex categories (metal, plastic) to a secondary model for grading verification

### Model Configuration & Setup

- **Stage 1 Backbone:** MobileNetV2 (8-Class Base Material Filter)

- **Stage 2 Backbone:** MobileNetV2 (10-Class Classification & Grading)

- **Pipeline:** Input (224 x 224 x 3) → Stage 1 Predicts → IF 'metal' or 'plastic': Route to Stage 2 for final decision → ELSE: Output Stage 1 decision

- **Inference Setup:** Batch size strictly constrained to 1 → Simulates real-world edge deployment where waste items are processed sequentially (frame-by-frame) on a conveyor belt

*Architecture 2 dual-evaluation metrics*
<img width="1840" height="961" alt="image" src="https://github.com/user-attachments/assets/2f803caa-0cb3-4b0f-b4c7-b942377efce8" />

### System Performance Metrics

- **Average Latency:** 136.65 ms / image

- **Frame Rate:** 7.32 FPS

- **Operational Throughput:** ~439 images / min

- **Stage 2 Utilization:** 375 / 1000 samples processed by the secondary model

- **Routing Efficiency:** 18 classification errors successfully rescued by Stage 2

Performance was evaluated under two conditions to isolate the specific impact of the quality grading task:

| Evaluation Scope                  | Target Classes | Accuracy |
|----------------------------------|---------------|----------|
| Full Task (Material + Grading)   | 10 Classes    | 78.00%   |
| Pure Material Classification     | 8 Classes     | 90.50%   |
| **Performance Gap (Δ)**          | —             | **+12.50%** |

### Key Findings

**1. Marginal Improvement in Pure Material Recognition**

The task decoupling strategy slightly improved broad categorization:

- Pure Classification Accuracy increased to 90.50% (vs. Baseline 89.70%)

- Stage 2 successfully acted as a secondary filter, "rescuing" 18 misclassified items

→ Separating macro-level filtering from micro-level grading shows conceptual promise

**2. Persistent Failure in Quality Grading**

Despite the multi-stage approach, the 12.50% performance gap remains almost identical to Architecture 1:

- Metal Grade A Recall dropped even further to 0.0800 (from baseline 0.0900)

- While Stage 2 rescued some items, it failed entirely to resolve the fine-grained intra-class confusion between Grade A and Grade B

### Root Causes for Grading Failure

**Architectural Redundancy & Feature Waste:** Stage 2 utilizes a full 10-class CNN, blindly recalculating low-level features (edges/textures) from scratch instead of focusing solely on the A/B grading logic. This redundancy costs 30+ ms of latency per image without improving grading precision

**Cascading Error Propagation:** The rigid routing mechanism relies entirely on Stage 1's hard class prediction. If Stage 1 misclassifies a crushed metal can as 'glass', the system bypasses Stage 2 entirely, permanently locking in the error. It lacks an uncertainty assessment

**CNN Representational Limits:** Reusing MobileNetV2 for Stage 2 confirms that the grading bottleneck is not merely a task-combination issue, but a fundamental limitation of the CNN architecture itself. Local receptive fields cannot capture the strict, "all-or-nothing" semantic logic required to verify Grade A conditions

Architectural Implications
Need for Confidence-Based Routing: Hard-coded conditional routing based solely on predicted class is structurally flawed. The pipeline must route inputs based on the primary model's statistical uncertainty (confidence scores) to prevent cascading errors.

Justification for Vision Transformers (VLMs): Chaining standard CNNs merely compounds computational cost (FPS dropped from 9.27 to 7.32) without solving the surface evaluation problem. True quality grading mathematically requires the global attention and semantic reasoning capabilities of a Transformer/VLM in the secondary stage.



