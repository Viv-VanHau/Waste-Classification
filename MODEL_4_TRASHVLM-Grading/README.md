TRASHVLM-Grading is a high-precision Vision Transformer (ViT) model dedicated solely to the hierarchical grading of high-value recyclables: Metal and Plastic. By isolating these categories from the general waste stream, this model performs a deep-dive quality assessment to distinguish between Grade A (Premium/Intact) and Grade B (Standard/Contaminated) materials.

## 1. Technical Methodology

### Data Layer Extraction
Instead of training on the full 10-class dataset, Model 4 utilizes a Targeted Layer Extraction strategy. We isolated the four most economically volatile classes from the UWCD (Unified Waste Classification Dataset):

- metal_Grade_A, metal_Grade_B

- plastic_Grade_A, plastic_grade_B

### State-of-the-Art Architecture: ViT + QLoRA

To achieve superior grading accuracy, we transitioned from CNNs to a Vision Transformer (ViT) backbone:

- **Base Model:** google/vit-base-patch16-224-in21k

- **Parameter-Efficient Fine-Tuning (PEFT):** We implemented QLoRA (4-bit Quantization + Low-Rank Adaptation)

- **Quantization:** 4-bit NormalFloat (NF4) with Double Quantization via bitsandbytes

- **LoRA Config:** Rank (r) = 32 and Alpha = 64, targeting query and value matrices

Benefits: This allows for full-scale Transformer fine-tuning on consumer-grade GPUs (T4) while maintaining high-fidelity feature extraction

### Automated Cost-Sensitive Learning

Given the extreme imbalance (e.g., Plastic Grade A having significantly fewer samples), we implemented an Automated Class Weighting pipeline:

- The system dynamically calculates weights using the 'balanced' heuristic

- These weights are injected into a Custom WeightedTrainer, ensuring that misclassifications of rare Grade A materials are penalized more heavily than majority classes

### Training Configuration & Pipeline

- **Ecosystem:** Built on the Hugging Face stack (transformers, peft, datasets, accelerate)

- **Data Pipeline:** Utilized imagefolder for high-speed streaming and with_transform for on-the-fly normalization and resizing to 224x224

- **Optimizer:** AdamW with a learning rate of 1e-4

- **Hardware Optimization:** FP16 Mixed Precision enabled for faster computation

- **Objective:** Maximizing F1-Macro Score to ensure unbiased performance across all four quality grades

## 2. Ouput

*Fig 1. MODEL 4 - LEARNING CURVES*
<img width="1589" height="590" alt="image" src="https://github.com/user-attachments/assets/fa6f1960-01ad-45b3-b1fc-da105bacdf25" />

Loss Convergence: The training loss exhibits a classic exponential decay, dropping from 5.0 to <0.5 within the first 4 epochs. The Validation Loss remains exceptionally stable and flat (around 0.25), indicating that the 4-bit QLoRA quantization provided a perfect balance between model capacity and regularization.

Metric Evolution: Validation Accuracy plateaus almost immediately at ~95% from Epoch 2. However, the F1-Macro (yellow line) continues to climb steadily from 0.72 to 0.86. This confirms that the model continued to improve its understanding of the minority classes (Grade A) even after the majority classes (Grade B) had already converged.

*Fig 2. MODEL 4 - CONFUSION MATRIX*
<img width="650" height="600" alt="image" src="https://github.com/user-attachments/assets/65aa2678-1e7b-439b-80a4-a5b75dbd5335" />

*Fig 3. MODEL 4 - Performance Evaluation*
<img width="650" height="280" alt="image" src="https://github.com/user-attachments/assets/f6ebee24-0671-4511-92ef-62ec948aaf5f" />

Model 4 (TrashVLM-Grading) serves as the Specialized Quality Auditor in our hierarchy. It is responsible for the final A/B grading of Metal and Plastic streams

- **Global Accuracy:** 96.61%

- **Macro Average F1-Score:** 84.05%

- **Weighted Average F1-Score:** 96.53%

The model shows high reliability in distinguishing between material types and quality grades:

- **Metal Grade B:**	Near-perfect identification of standard metal

- **Plastic Grade B:** Robust detection of standard plastic recyclables

- **Metal Grade A:**	Successfully captured premium metal despite small sample size

- **Plastic Grade A:** The most challenging class (extreme scarcity), yet managed a >50% recall

The primary source of error is "Down-grading" (Grade A items predicted as Grade B)

- 25 samples of Metal Grade A were classified as Grade B
- 21 samples of Plastic Grade A were classified as Grade B

*Reasoning:* Under harsh underwater lighting, subtle deformations or light scattering can make an "Intact" (Grade A) object appear "Crushed" or "Contaminated" (Grade B). However, "Up-grading" errors (Grade B predicted as A) are extremely low, ensuring the purity of the premium output stream.

The high Macro Average F1-score (0.84) on a dataset with such extreme imbalance (1546 vs. 45 samples) is a direct result of our Weighted Cross-Entropy Loss:

- **Recall Priority:** By "buffing" the weights of Grade A samples, we forced the ViT's self-attention mechanism to focus on structural integrity and transparency

- **QLoRA Efficiency:** Fine-tuning with Rank (r=32) allowed the model to capture the complex textural nuances required for grading without the risk of overfitting common in full fine-tuning
