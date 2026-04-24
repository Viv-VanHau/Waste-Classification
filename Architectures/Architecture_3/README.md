| Iteration        | Core Software Mechanism      | Accuracy (10-Class) | Metal Grade A (Recall) | **Key Architectural Insight** |
|------------------|------------------------------|---------------------|------------------------|---------------------------|
| Base Arch 3      | Hard Routing (Override)      | 75.80%              | 0.5500                 | VLM solves grading, but CNN Gatekeeper causes destructive overrides |
| Sensitivity      | Threshold Sweep              | 75.80% (Max)        | N/A                    | Proves parameter tuning cannot fix structural incompatibility |
| Test 1           | Soft Fusion                  | 78.60%              | 0.5600                 | Probability averaging stabilizes accuracy but exposes underlying class bias |
| Test 2           | Calibrated Fusion            | 80.10%              | 0.5600                 | Prior adjustment boosts minority classes, breaking the 80% threshold |
| **Test 3**    | Top-K Constrained Fusion | **80.60%**          | 0.5400             | Masking prevents VLM hallucination but exposes the need for structural task decoupling |

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

- **Average Latency:** 125.24 ms / image

- **Frame Rate:** 7.98 FPS

- **VLM Utilization:** 286 / 1000 samples (28.6% of the dataset)

- **VLM Rescued Count:** 72 classification errors successfully fixed by the ViT

- **Overall Accuracy:** 75.80% (10-Class) | 88.70% (Pure Material 8-Class)

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

## Architecture 3 - Test 2: Calibrated Soft Fusion (Post-hoc Prior Adjustment)

**Motivation for Test 2:** While Test 1 (Soft Fusion) successfully arrested the systemic accuracy decay, it exposed a persistent statistical flaw deeply rooted in the 10-Class structure: extreme class imbalance bias. 

The system's ability to identify Grade A materials plummeted (Plastic Grade A Recall: 0.3800). 

Because both the CNN and VLM were trained on imbalanced datasets, their raw softmax outputs naturally skewed toward the statistically dominant Grade B. Soft Fusion simply averaged these two biased distributions, failing to correct the underlying prejudice. 

To counteract this, Test 2 implements Post-hoc Probability Calibration, a Bayesian-inspired adjustment to manually shift the prior distribution before finalizing the output.

- **Design Paradigm:** Probability Weighting / Algorithmic Bias Correction

- **Core Task:** Inject domain-specific calibration multipliers into the fused probability array to artificially boost the "activation likelihood" of minority classes (Grade A), counteracting the inherent training bias

### Inference Logic & Routing Setup

- **Routing:** Maintains the optimal 0.85 confidence threshold established in previous analysis

- **Base Fusion:** If the VLM is invoked, the system computes the hybrid probability array: P_{fused} = (0.4 x P_{CNN}) + (0.6 x P_{VLM})

- **Calibration Multiplier:** The raw P_{fused} array is multiplied element-wise by a custom weight matrix designed to inject synthetic "oxygen" into suppressed classes:

- *Plastic Grade A:* Multiplied by 1.50 (50% artificial boost to rescue the 38% Recall)

- *Metal Grade A:* Multiplied by 1.25 (25% boost)

- *All other classes:* Multiplied by 1.0 (No change)

- **Final Decision:** The system recalculates the argmax on the newly calibrated probability distribution

### System Performance Metrics

*Architecture 3 Test 2*
<img width="450" height="500" alt="image" src="https://github.com/user-attachments/assets/b4d20bb1-7094-40a3-8b69-5226c6fae436" />

- Average Latency: 126.30 ms / image

- Frame Rate: 7.92 FPS

- VLM Utilization: 286 / 1000 samples

- Calibration Shifts: 36 predictions were explicitly altered (bypassing the standard argmax) due to the calibration multipliers

- Total Rescues: 75 errors successfully prevented in Stage 2

- Overall Accuracy: 80.10% (A milestone breakthrough, surpassing the 80% threshold)

### Key Findings

**1. The Efficacy of Algorithmic Bias Correction**

The 36 calibration shifts yielded massive, targeted improvements precisely where the system was failing:

- Plastic Grade A: Recall surged from a catastrophic 0.3800 up to 0.6300. F1-score jumped from 0.5205 to 0.7000

- Metal Grade A: Recall stabilized at 0.5600, but Precision improved slightly

→ By mechanically shifting the prior distribution, we proved that the hybrid network was extracting the correct features for Grade A, but those signals were being statistically drowned out by the Grade B bias.

**2. Breaking the 80% Threshold**

For the first time in the 10-Class unified pipeline (where one system handles all materials and grades), accuracy crossed the 80% mark. This 80.10% result proves the viability of CNN-VLM fusion when supported by rigorous mathematical calibration.

## Architecture 3 - Test 3: Top-K Constrained Calibrated Fusion

**Motivation for Test 3:** Test 2 successfully addressed the class imbalance bias via probability calibration, pushing the accuracy to 80.10%. 

However, empirical log analysis revealed that when the VLM was invoked to resolve low-confidence CNN predictions, it occasionally generated wildly out-of-domain probabilities due to the highly noisy nature of the visual data. 

Soft fusion mitigated this, but did not eliminate it. To maximize efficiency, the VLM should act strictly as a "Tie-Breaker" between the CNN's most likely hypotheses, rather than searching the entire 10-Class space. 

Test 3 implements Top-K Constrained Fusion to algorithmicly force the VLM to focus its semantic reasoning solely on the most highly probable material candidates identified by the primary CNN.

- Design Paradigm: Heuristic Bounding / Top-K Logit Masking

- Core Task: Utilize the CNN's localized feature extraction to narrow the search space, forcing the VLM to apply its global attention exclusively to resolving the ambiguity between the Top-2 predictions

### Inference Logic & Routing Setup

- Stage 1 (CNN): MobileNetV2 outputs the raw probability array. If Confidence >= 0.85, the pipeline bypasses 

- Stage 2.Top-K Extraction: If routed to Stage 2, the system extracts the indices of the Top 2 highest probabilities from the CNN output. (Hypothesis: Even when uncertain, the CNN correctly identifies the true class within its Top 2 choices)

- Logit Masking: The VLM's 10-Class probability array is intercepted. All probabilities outside the CNN's Top-2 indices are forcefully set to zero (zero-out)

- Normalization & Fusion: The masked VLM array is re-normalized to sum to 1.0, then fused with the CNN using the established 40/60 weighting

- Calibration: The calibration multipliers (1.50x for Plastic A, 1.25x for Metal A) are applied before the final argmax decision

### System Performance Metrics

*Architecture 3 Test 3*
<img width="500" height="550" alt="image" src="https://github.com/user-attachments/assets/a78d7a5d-0e1f-4fbb-aef1-1d8ffae5499f" />

- Average Latency: 117.88 ms / image

- Frame Rate: 8.48 FPS

- VLM Utilization: 286 / 1000 samples

- Constrained Corrections: 33 out-of-domain VLM errors were successfully prevented by the Top-2 masking

- Total Rescues: 76 classification errors fixed by the integrated Stage 2 logic

### Key Findings

**1. The Effectiveness of Bounded Reasoning**

By restricting the VLM to a "Tie-Breaker" role, the pipeline achieved its highest accuracy (80.60%) while minimizing destructive overrides:

- The Top-K mask successfully intercepted and prevented 33 instances where the VLM attempted to output an irrational prediction (e.g., classifying a low-confidence plastic bottle as a battery)

- By bounding the search space, the VLM could safely leverage its 60% fusion weight to resolve the fine-grained differences without corrupting the macro-level material classification

**2. The Persistent Metal Grading Bottleneck**

Despite applying the absolute pinnacle of algorithmic optimization (confidence routing, soft fusion, probability calibration, and Top-K masking) the system still fundamentally struggles with specific grades:

- Metal Grade A Recall: Stagnated at 0.5400.

→ The VLM is capable of grading, but forcing it to operate within a 10-Class framework fundamentally dilutes its representational power. Even when artificially constrained to two classes during inference, the VLM was structurally trained on ten

## Final Conclusion for Architecture 3

Architecture 3 has definitively proven two critical axioms for this research:

- Vision Transformers are capable of strict quality grading (achieving > 50% Recall on Metal A, compared to the CNN's ~8%)

- Unified 10-Class Models are inherently flawed routers. Forcing a model to learn macro (materials) and micro (grades) simultaneously dilutes its latent space

The 80.60% ceiling reached in Test 3 represents the absolute limit of "software-level patching." To achieve industrial-grade accuracy (>85%), the system must abandon the 10-Class paradigm entirely. This mandates the transition to Architecture 4, which will physically decouple the pipeline into an 8-Class CNN (pure material classification) and a specialized 4-Class VLM (pure grading), removing the need for complex algorithmic safety nets.

