## Architecture 3: Confidence-Routed Hybrid System [ Stage 1 (Model 1 - CNN 8 Classes) → Stage 2 (Model 2 - CNN 10 Classes) ]

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

