## Architecture 3: Confidence-Routed Hybrid System [Stage 1 (Model 1 - CNN 8 Classes) → Stage 2 (Model 2 - CNN 10 Classes)]

- **Role:** The first hybrid iteration integrating a Vision Transformer (ViT) to specifically address the fine-grained grading bottleneck identified in the purely CNN-based architectures

- **Design Paradigm:** Uncertainty-driven conditional routing across heterogeneous frameworks (CNN to Transformer)

- **Core Task:** Utilize a high-speed CNN for primary triage, routing only statistically uncertain "Hard Cases" to a computationally heavier, semantically aware VLM

### Model Configuration & Setup

- **Stage 1 (Triage):** MobileNetV2 (10-Class)

- **Stage 2 (Refinement):** TrashVLM (ViT-base-patch16-224 fine-tuned via LoRA)

- **Routing Threshold:** Confidence < 0.85

- **Pipeline:** Input → CNN Predicts. IF CNN Confidence < 0.85 → Route to VLM for override. ELSE → Accept CNN prediction

### System Performance Metrics

*Architecture 3 dual-evaluation metrics*
<img width="1830" height="979" alt="image" src="https://github.com/user-attachments/assets/288264c7-3d75-47a9-b637-f181e9684c79" />

- Average Latency: 125.24 ms / image

- Frame Rate: 7.98 FPS

- VLM Utilization: 286 / 1000 samples (28.6% of the dataset)

- VLM Rescued Count: 72 classification errors successfully fixed by the ViT

- Overall Accuracy: 75.80% (10-Class) | 88.70% (Pure Material 8-Class)

This architecture yielded highly polarized results, proving the fundamental capability of Transformers while exposing a critical flaw in the routing mechanism:

For the first time in this research, the system demonstrated a genuine structural capability to process multi-conditional quality rules

Metal Grade A (The Ultimate Test): Recall surged to 0.5500 (compared to the CNN baseline of 0.0800 and the heavily calibrated CNN peak of 0.3100). F1-score jumped to 0.5729.

→ The Vision Transformer's global self-attention mechanism successfully captured the localized semantic defects (dents, dirt) that local CNN receptive fields missed. When the VLM is allowed to see the image, it successfully differentiates Grade A from Grade B.

Despite the VLM's superior grading ability, the overall system accuracy regressed to 75.80% (lower than the optimized CNN baseline of 82.20%):

- Root Cause: The Stage 1 CNN (10-Class) acts as a highly biased, "arrogant gatekeeper." Because the CNN was forced to learn both material and grade in one step, it developed a severe statistical bias toward Grade B

- The Trap: The CNN frequently predicts "Grade B" with extreme confidence (e.g., Confidence = 0.95+), easily clearing the 0.85 threshold

→ Because the CNN is overconfident in its mistakes, the pipeline bypasses Stage 2. The VLM only received 28.6% of the dataset. It cannot rescue Grade A images if the CNN confidently hoards them

## Architecture 3 - Sensitivity Analysis

To fully address the paradox observed in the baseline hybrid system, we must investigate whether the performance bottleneck can be resolved by manipulating the routing parameters. 

Hypothetically, if the Stage 1 CNN is overconfident in its grading errors, raising the confidence threshold should forcefully strip the CNN of its authority, pushing a higher volume of "hard cases" to the VLM. 

The data reveals a counter-intuitive trajectory as the threshold increases:

*Architecture 3 Sensitivity Analysis*
<img width="800" height="210" alt="image" src="https://github.com/user-attachments/assets/fed81fca-a48b-43f7-a510-0c42bec20c76" />

The empirical data uncovers a critical architectural flaw that threshold optimization alone cannot fix.

**1. The Inverse Accuracy Correlation**

Logically, routing more data to a highly advanced Vision Transformer should increase accuracy. The data proves the exact opposite:

- As the threshold rises to 0.95, VLM utilization doubles (reaching 38.9%), and successful rescues nearly triple (reaching 100)

- However, the overall 10-Class Accuracy actually drops from its peak of 75.80% down to 74.60%

- More critically, the pure material (8-Class) accuracy bleeds steadily from 89.50% down to 87.60%

**2. Identifying the Root Cause**
   
By forcing the threshold higher, the system intercepts CNN predictions that were statistically uncertain but factually correct.

- The VLM is highly specialized in complex surface grading (identifying dents and dirt), but it is functionally inferior to the CNN at broad, macro-level material recognition

- When forced to re-evaluate base materials, the VLM overwrites correct CNN material predictions with incorrect ones. It rescues 100 grading cases but destroys even more pure material cases in the process, resulting in a net negative impact on the system

This sensitivity sweep mathematically proves that you cannot optimize a flawed pipeline simply by shifting its gating parameter. Finding a "sweet spot" is impossible because the Stage 1 router and the Stage 2 solver have fundamentally conflicting strengths.

## Architecture 3 Extensions

To fully exhaust the potential of hierarchical pipelines before transitioning to Transformer-based models, we conducted three controlled experiments (Tests 1-3) built upon the Architecture 2 baseline. The objective is to evaluate how various software-level routing logic and confidence heuristics impact the accuracy-latency trade-off. In this architectural research context, both performance gains and regressions provide critical insights into system boundaries.

## Architecture 3 - Test 1: Soft Fusion Fallback

**Motivation for Test 1:** The sensitivity sweep in the baseline Architecture 3 revealed a critical flaw: hard-routing is a double-edged sword. 

As the VLM was granted more authority (threshold = 0.95), it successfully rescued 100 grading errors but simultaneously destroyed even more correct base material predictions, causing overall accuracy to drop. 

The VLM acts as an excellent "Grader" but a poor "Material Filter." Test 1 introduces Soft Fusion to act as an algorithmic shock absorber. Instead of allowing the VLM to completely overwrite the CNN (Hard Override), the system forces a mathematical consultation between both models, weighting the CNN's superior material intuition against the VLM's superior surface inspection.

- **Role:** An optimized hybrid iteration utilizing weighted probability combination to prevent the VLM from corrupting macro-level material predictions

- **Design Paradigm:** Heuristic-Triggered Soft Voting Ensemble

- **Core Task:** Retaining a percentage of the CNN's original confidence distribution when consulting the Vision Transformer on hard cases

### Inference Logic & Routing Setup

Stage 1 (Primary): MobileNetV2 outputs a 10-class probability array (P_{CNN})

- **Routing Trigger:** Confidence_{CNN} < 0.85

- **Fast-Path:** If Confidence >= 0.85, the pipeline outputs the CNN prediction directly

- **Soft Fusion Sub-routine:** If routed to Stage 2, the VLM generates its own 10-class probability array (P_{VLM})

- **Weighted Integration:** The final decision is mathematically fused using a 40/60 split favoring the VLM's grading expertise while anchoring to the CNN's material baseline: P_{final} = (0.4 x P_{CNN}) + (0.6 x P_{VLM})

### System Performance Metrics

**Architecture 3 Test 1**
<img width="420" height="450" alt="image" src="https://github.com/user-attachments/assets/0e021fd5-770d-4c4a-b052-f5370317c87f" />

- Average Latency: 118.25 ms / image

- Frame Rate: 8.46 FPS

- VLM Utilization: 286 / 1000 samples

- Fusion Rescues: 69 classification errors successfully prevented by the fused probabilities

- Overall Accuracy: 78.60% (An absolute improvement of +2.80% over Base Architecture 3)

### Key Findings

**1. Successful Mitigation of False Rescues**

The Soft Fusion logic performed exactly as hypothesized, stabilizing the pipeline's overall accuracy:

- The systemic accuracy surged to 78.60%, fully recovering the performance lost to the VLM's destructive overrides in the baseline hybrid

- The 40% CNN weight effectively acted as an anchor, preventing the VLM from making wildly incorrect out-of-domain predictions (e.g., misclassifying a highly confident metal can as glass)

**2. Preservation of VLM Grading Superiority**

Crucially, the fusion mechanism did not dilute the VLM's primary contribution:

- Metal Grade A Recall: Maintained an impressive 0.5600 (compared to the CNN-only ceiling of 0.3100)

- The VLM successfully applied its global self-attention to surface defects, pulling up the minority class performance while relying on the CNN weight to keep the base material accurate.

This test mathematically proves that heterogeneous models (CNNs and Transformers) possess distinct, complementary feature spaces. Hard routing creates conflicting objectives, whereas probability fusion harmonizes them.
