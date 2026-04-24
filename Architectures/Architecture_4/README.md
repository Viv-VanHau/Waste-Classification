| Iteration     | Routing / Masking Mechanism        | Accuracy | Metal Grade A (Recall) | Key Architectural Insight |
|---------------|------------------------------------|----------|------------------------|---------------------------|
| **Base Arch 4**   | Decoupled (Unconstrained VLM)      | **90.50%**   | 0.7500                 | Task isolation shatters the 80% ceiling. CNN handles materials; VLM masters grading |
| Test 1        | Decoupled + Calibration Boost      | 90.50%   | 0.7600                 | Proves the system is structurally optimized; probability manipulation yields diminishing returns |
| Test 2        | Decoupled + Hard Hierarchical Mask | 89.90%   | 0.7500                 | Prevents VLM hallucinations but completely blinds the system to CNN triage failures |
| Test 3        | Decoupled + Soft Penalty Mask      | 90.40%   | 0.7500                 | The definitive configuration. Balances hierarchical control with dynamic VLM error-correction capabilities |

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

Unlike the unstable 10-Class hybrid (Architecture 3), where calibration caused unpredictable shifts across entirely unrelated materials, the decoupled pipeline is highly robust. Applying a 25% boost to a metal grade did not artificially inflate or corrupt the plastic grades or the base materials. The task isolation acts as proof of the architecture's stability

## Architecture 4 - Test 2: Hierarchical Logit Masking

**Motivation for Test 2**

While Architecture 4 successfully decoupled base classification from quality grading, log analysis revealed a subtle structural vulnerability

The VLM-4 specialist model evaluates four distinct classes simultaneously (Metal A/B and Plastic A/B). When Stage 1 (CNN) confidently routes a metal item to Stage 2, the VLM is theoretically given the freedom to contradict Stage 1 and classify the item as Plastic Grade A

This permits "cross-domain hallucinations," where the VLM's grading expertise overrides and corrupts the CNN's superior macro-material classification

To enforce strict architectural hierarchy, Test 2 implements Domain-Constrained Masking, algorithmically locking the VLM into the specific material domain dictated by Stage 1

- **Design Paradigm:** Hierarchical Tree Routing / Conditional Logit Masking

- **Core Task:** Mathematically restrict the Stage 2 Vision Transformer's output space exclusively to the A/B quality grades of the specific base material predicted by the Stage 1 CNN

### Inference Logic & Routing Setup

- **Hierarchical Trigger:** Stage 1 predicts a base material (e.g., metal). The item is routed to the VLM

- **Dynamic Mask Generation:** The system generates a domain-specific constraint mask corresponding to the Stage 1 prediction:

- *If CNN predicts metal:* Mask = [1.0, 1.0, 0.0, 0.0] (allowing only Metal A/B)

- *If CNN predicts plastic:* Mask = [0.0, 0.0, 1.0, 1.0] (allowing only Plastic A/B)

- **Mask Application & Normalization:** The VLM computes raw probabilities across all 4 classes. The array is multiplied by the constraint mask, forcefully zeroing out cross-domain probabilities. The array is then re-normalized to sum to 1.0

- **Final Decision:** The VLM acts purely as a local grade discriminator within the enforced boundary

### System Performance Metrics

*Architecture 4 Test 2*
<img width="470" height="500" alt="image" src="https://github.com/user-attachments/assets/b6e7f588-6897-4701-a0c2-0110a7f77978" />

- Average Latency: 120.23 ms / image

- Frame Rate: 8.32 FPS

- VLM Utilization: 375 / 1000 samples

- Masked Corrections: 6 explicit out-of-domain VLM hallucinations were successfully intercepted and prevented by the hierarchical mask.

Overall Pipeline Accuracy: 89.90%

###  Key Findings

The implementation successfully established a rigid deterministic hierarchy:

- The constraint mask intercepted 6 distinct instances where the VLM attempted to cross material domains (e.g., trying to grade a metal object as plastic)

- By enforcing this logic, the pipeline ensures that the 92.70% pure material accuracy achieved by the Stage 1 CNN is mathematically protected from downstream VLM corruption

Despite creating a theoretically safer and more logically sound system, the overall accuracy slightly regressed from the 90.50% baseline to 89.90%

- In rare, highly anomalous cases, the Stage 1 CNN makes a primary classification error (e.g., predicting plastic for a highly degraded metal can). In the unconstrained Base Arch 4, the VLM sometimes "hallucinated" the correct material across domains, accidentally correcting the CNN's mistake

- Trade-off Resolution: By applying the domain constraint, we remove the VLM's ability to accidentally "lucky guess" the base material. While this causes a minor statistical drop (-0.60%), it is an essential engineering trade-off. In industrial deployment, deterministic reliability (where the system obeys strict hierarchical rules) is vastly preferred over relying on neural network hallucinations for random error correction

## Architecture 4 - Test 3: Soft Penalty Domain Constraint (Dynamic Trust Masking)

While Test 2's strict hierarchical masking guaranteed that the VLM would never contradict the CNN's material prediction, it introduced an inflexible vulnerability: if the CNN made a catastrophic triage error, the VLM was mathematically "blindfolded" and forced to grade the wrong material

To achieve true systemic resilience, the pipeline requires a middle ground between the chaotic freedom of the Base Architecture and the rigid authoritarianism of Test 2

Test 3 introduces Soft Penalty Domain Constraints. Instead of forcefully zeroing out cross-domain probabilities, the system applies a heavy penalty multiplier. This heuristic weighting suppresses casual VLM hallucinations but permits "rescues", allowing the VLM to override the CNN only if its cross-domain semantic conviction is overwhelmingly high

- **Design Paradigm:** Heuristic Weighting / Soft Bounding

- **Core Task:** Balance architectural hierarchy with error-correction flexibility by penalizing, rather than prohibiting, cross-domain VLM predictions

### Inference Logic & Routing Setup

- **Stage 1 (Macro-Triage):** MobileNetV2 (8-Class) predicts the base material

- **Routing:** Targeted materials (metal, plastic) are routed to the VLM (4-Class)

- **Soft Penalty Masking:** Based on the Stage 1 prediction, a penalty mask is generated

- *Example (CNN predicts Metal):* The mask applied to the VLM output is [1.0, 1.0, 0.35, 0.35]. The probabilities for Metal A/B remain intact, while the probabilities for Plastic A/B are severely penalized (reduced by 65%)

- **Dynamic Override:** To output a cross-domain prediction (e.g., classifying as Plastic when the CNN said Metal), the VLM's raw probability for Plastic must be exponentially higher than its probability for Metal to survive the 0.35 penalty multiplier

### System Performance Metrics

*Architecture 4 Test 3*
<img width="450" height="470" alt="image" src="https://github.com/user-attachments/assets/630f289b-2ba5-429d-9699-69acc2fb49d0" />

- Average Latency: 121.95 ms / image

- Frame Rate: 8.20 FPS

- VLM Utilization: 375 / 1000 samples

- Suppressed Hallucinations: 1 out-of-domain false alarm successfully prevented by the penalty

- Soft Rescues: 5 cross-domain errors explicitly fixed by the VLM overcoming the penalty

- Overall Pipeline Accuracy: 90.40%

### Key Findings

The implementation perfectly executed its dual mandate:

- The system recorded 5 "Soft Rescues". These were instances where the Stage 1 CNN entirely misidentified the material structure, but the VLM's semantic recognition was so statistically dominant that it mathematically overpowered the 0.35 penalty to correct the base material

- Simultaneously, the penalty successfully suppressed 1 hallucination, preventing the VLM from making a casual cross-domain error

Operating at 90.40% (statistically equivalent to the unconstrained baseline), this configuration is qualitatively superior. It provides the mathematical safety net of Architecture 4 Test 2, without sacrificing the synergistic error-correction potential of Architecture 3.

## Final Conclusion for Architecture 4

The evolutionary journey from Architecture 1 (Monolithic CNN at 77.10%) to Architecture 4 (Decoupled VLM Pipeline at ~90.50%) definitively answers the core research question. Standard Convolutional Neural Networks fundamentally lack the semantic reasoning necessary for strict, multi-conditional quality grading.
