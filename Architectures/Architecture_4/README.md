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
