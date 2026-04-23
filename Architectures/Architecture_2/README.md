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

## Architecture 2 Extensions

To fully exhaust the potential of hierarchical pipelines before transitioning to Transformer-based models, we conducted three controlled experiments (Tests 1-3) built upon the Architecture 2 baseline. The objective is to evaluate how various software-level routing logic and confidence heuristics impact the accuracy-latency trade-off. In this architectural research context, both performance gains and regressions provide critical insights into system boundaries.

## Architecture 2 - Test 1: Confidence-Based Fallback Mechanism

**Motivation for Test 1:** The base Architecture 2 suffered heavily from "Cascading Errors" due to its rigid routing. If Stage 1 incorrectly classified a crushed metal can as 'glass', the item bypassed the grading stage entirely, permanently locking in the error. Test 1 attempts to patch this vulnerability by introducing a dynamic, uncertainty-aware fallback mechanism.

- **Role:** Experimental iteration extending Architecture 2 with dynamic, uncertainty-aware thresholding

- **Design Paradigm:** Heuristic-driven conditional routing

- **Core Task:** Mitigate cascading errors from the primary model by utilizing prediction confidence as a safety net before finalizing outputs

### Inference Logic & Routing Setup

**Stage 1 (Primary):** MobileNetV2 (8-class) outputs a base material prediction alongside a statistical confidence score

**Trigger A (Mandatory Grading):** If Stage 1 predicts metal or plastic $\rightarrow$ Route to Stage 2 for A/B quality assessment

**Trigger B (Uncertainty Fallback):** If Stage 1 predicts another material, BUT confidence is < 0.90 → Route to Stage 2 as a safety verification

**Fast-Path (Bypass):** If Stage 1 predicts another material AND confidence is >= 0.90 → Bypass Stage 2 and finalize output

### System Performance Metrics

*Architecture 2 Test 1*
<img width="430" height="500" alt="image" src="https://github.com/user-attachments/assets/fdbf259f-c59f-4528-9a03-76f55b2f64f0" />

- Average Latency: 160.06 ms / image

- Frame Rate: 6.25 FPS

- Stage 2 Utilization: 453 / 1000 samples (45.3% of the dataset) -- Grading Calls: 375 -- Fallback Calls: 78

- System Recovery: 10 classification errors successfully prevented by the fallback mechanism.

- Overall Accuracy: 77.50%

### Key Findings

**1. Proof of Concept for Dynamic Routing**

The logic successfully intercepted and corrected cascading errors that the rigid Base Architecture 2 missed:

- 78 statistically uncertain predictions were caught and routed to the secondary model

- 10 of these were successfully "rescued" (Stage 1 base prediction was entirely incorrect, but Stage 2 overrode it with the perfect truth)

→ Confidence thresholding is a mathematically sound software strategy to prevent rigid pipeline failures

**2. Severe Computational Penalty**

The safety net comes at a heavy processing cost:

- Frame rate plummeted to 6.25 FPS (down from 9.27 FPS in the Single-Stream Baseline)

- Forcing nearly half the dataset (45.3%) through two separate MobileNetV2 networks creates massive architectural redundancy

→ Using a heavy 10-class CNN just to double-check uncertain predictions is computationally inefficient and pushes the system below optimal real-time operational thresholds

**3. The Grading Bottleneck Remains Unsolved**

Despite the smarter software logic, the core classification flaw remains untouched:

- Metal Grade A Recall remains severely suppressed at 0.0900 (9%)

- Overall pipeline accuracy only saw a negligible improvement (77.50% vs 77.10% baseline)

→ Advanced routing logic cannot fix a fundamentally incapable feature extractor. Even when routed correctly, the Stage 2 CNN still cannot resolve the fine-grained surface defects required for A/B grading.

Test Implication: The confidence < 0.90 heuristic proves that multi-stage pipelines require uncertainty assessment to function reliably.
