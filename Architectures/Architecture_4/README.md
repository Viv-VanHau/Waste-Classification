## Architecture 4: Precision Grading Pipeline (Decoupled VLM) [Stage 1 (Model 1 - CNN 8 Classes) → Stage 2 (Model 4 - VLM Grading)]

**Motivation for Architecture 4:**

The exhaustive optimization of Architecture 3 established a hard operational ceiling of 80.60%

The limitation was fundamentally structural: forcing neural networks (whether CNN or VLM) to simultaneously learn macro-level material boundaries and micro-level quality variations dilutes their representational capacity

The hybrid system suffered from conflicting objectives, requiring complex, computationally heavy probability calibration just to maintain stability. Architecture 4 proposes a radical structural shift: Task Decoupling

By stripping the Stage 1 CNN of its grading authority (reverting to an 8-Class material filter) and restricting the Stage 2 VLM exclusively to surface inspection (a 4-Class grading), this pipeline aims to eliminate destructive overrides

- **Design Paradigm:** Decoupled Task / Hard-Conditional Routing

- **Core Task:** Isolate base material triage from quality assessment, utilizing specialized models for their respective mathematical strengths (CNN for rapid structural triage, VLM for complex semantic grading)

### Inference Logic & Routing Setup

- **Stage 1 (Macro-Triage):** A lightweight MobileNetV2 trained exclusively on 8 pure material classes (ignoring grades entirely)

- **Routing Trigger:** Deterministic logic. If Stage 1 predicts metal or plastic, the item is routed to Stage 2. All other materials bypass Stage 2

- **Stage 2 (Micro-Grading):** A Vision Transformer (TrashVLM) trained strictly on 4 classes (metal_Grade_A, metal_Grade_B, plastic_Grade_A, plastic_Grade_B)

- **Final Decision:** The VLM's prediction acts as the absolute final output for targeted materials

### System Performance Metrics

*Architecture 4 dual-performance metric*
<img width="1813" height="943" alt="image" src="https://github.com/user-attachments/assets/4a45eb84-fa28-41fc-b679-c0b12be9bcbb" />

- **Average Latency:** 121.69 ms / image

- **Frame Rate:** 8.22 FPS

- **VLM Utilization:** 375 / 1000 samples routed strictly for grading.

- **Pure Material Accuracy (8-Class):** 92.70%

- **Overall Pipeline Accuracy (10-Class):** 90.50%

### Key Findings

The decoupled architecture achieved a monumental leap to 90.50% overall accuracy, decisively breaking the 80% stagnation zone of unified models: 

- By removing grading from Stage 1, the CNN achieved an exceptional 92.70% pure material accuracy.

- Without the task of identifying base materials, the VLM operated with surgical precision, completely eliminating the "destructive override" phenomenon observed in previous hybrid iterations

Restricting the Vision Transformer to a 4-Class specialist environment unlocked its full attention mechanism, yielding unprecedented results in minority class detection:

- Metal Grade A: Recall surged to 0.7500, with Precision hitting an astounding 0.9868 (F1: 0.8523).

- Plastic Grade A: Recall reached 0.8900, with perfect Precision of 1.0000 (F1: 0.9418).

→ The VLM no longer wastes attention heads trying to differentiate an aluminum can from a glass bottle. 100% of its computational power is directed toward detecting surface dents and contamination, successfully mastering the multi-conditional logic of Grade A.

Despite utilizing a heavy Transformer, the deterministic routing logic maintained a highly stable 8.22 FPS. Because the VLM is only invoked exactly when necessary (37.5% of the time), and complex probability fusion matrices are eliminated, the system achieves near-real-time edge capability without sacrificing accuracy.

## Architecture 4 - Test 1: Calibrated Decoupled Grading

**Motivation for Test 1:**

The Decoupled VLM in baseline Architecture 4 achieved a monumental 90.50% accuracy.

However, a granular review of the confusion matrix indicated that the Vision Transformer still exhibited a slight statistical bias toward Grade B materials, a direct consequence of the imbalanced training distribution (e.g., thousands of Grade B samples versus hundreds of Grade A) 

While the specialized 4-Class structure naturally mitigated most of this bias, Metal Grade A Recall plateaued at 0.7500

To theoretically optimize the economic value of the recycling pipeline, where missing a premium Grade A material is more costly than downgrading a Grade B, Test 1 implements a highly targeted Probability Calibration on the VLM's output logits

- **Design Paradigm:** Targeted Output Probability Multiplier

- **Core Task:** Apply a minimal synthetic boost to the mathematical probability of the rarest class (metal_Grade_A) to counteract residual training bias, without triggering cascading errors across other classes

### Inference Logic & Routing Setup

- **Stage 1 (Macro-Triage):** Unchanged. MobileNetV2 (8-Class) acts as the deterministic router

- **Stage 2 (Micro-Grading):** TrashVLM (4-Class) generates the raw probability array for targeted items (metal or plastic)

- **Calibration Multiplier:** Before executing the argmax function, the VLM's output array is intercepted. The index corresponding to metal_Grade_A is multiplied by 1.25 (a 25% targeted boost). All other classes retain a multiplier of 1.0

- **Final Decision:** The system recalculates the optimal decision based on the calibrated probability distribution

### System Performance Metrics

*Architecture 4 Test 1*
<img width="450" height="470" alt="image" src="https://github.com/user-attachments/assets/0dfa5bbf-bf16-448c-93ed-3e117d8f7b0d" />

- **Average Latency** 119.84 ms / image

- **Frame Rate:** 8.34 FPS

- **VLM Utilization:** 375 / 1000 samples

- **Calibration Shifts:** 2 critical predictions were altered exclusively due to the multiplier

- **Overall Pipeline Accuracy:** 90.50% (Identical to Base Architecture 4)

### Key Findings

While the overall system accuracy remained statistically identical at 90.50%, the targeted calibration achieved its specific micro-objective:

- Metal Grade A Recall: Improved from 0.7500 to 0.7600

- Metal Grade A F1-Score: Peaked at 0.8539

The specific calibration shifts successfully increased borderline Grade A items that the raw VLM had conservatively downgraded to Grade B.

Unlike the unstable 10-Class hybrid (Architecture 3), where calibration caused unpredictable shifts across entirely unrelated materials, the decoupled pipeline is highly robust. Applying a 25% boost to a metal grade did not artificially inflate or corrupt the plastic grades or the base materials. The task isolation acts as a proof to the architecture's stability
