## 🛠 Quick Start
- Find the model 'Backbone_Stage1_Final_9429.h5' at: 
- Hardware: Google Colab (T4 GPU enabled)
- Execution: Run the provided train.py. The best-performing model will be saved automatically as Backbone_Stage1_Final_9429.h5

## **1. Model Architecture**
The architecture utilizes Transfer Learning with a highly optimized custom head designed for underwater imagery:

- **Backbone:** MobileNetV2 (Pre-trained on ImageNet). This provides a lightweight yet powerful feature extraction base, significantly reducing the parameter count compared to deeper networks like ResNet or VGG

- **Hybrid Pooling Head:** 
A combination of Global Average Pooling (GAP) and Global Max Pooling (GMP)
Rationale: GAP captures the overall spatial structure of the waste object, while GMP focuses on the most prominent features (such as metallic reflections or textile textures). Concatenating these two provides a richer feature representation for material identification

- **L2 Regularization (0.001):** Applied to the dense layers to penalize large weights and prevent overfitting
- *Dropout (0.5):* Randomly deactivates 50% of neurons during training to ensure the model learns robust features rather than memorizing the training data.
- *Batch Normalization:* Used to stabilize the learning process and accelerate convergence

## **2. Training Strategy**
The training process follows a rigorous Two-Phase approach:
### Phase 1: Head Warm-up
- Status: Backbone frozen; only the custom classification head is trained
- Learning Rate: 1e-3 (Adam Optimizer)
- Objective: To allow the newly initialized dense layers to adapt to the dataset features without distorting the pre-trained ImageNet weights
  
### Phase 2: Deep Fine-tuning
- Status: The top 30 layers of MobileNetV2 are unfrozen for fine-tuning
- Learning Rate: 5e-5
- Objective: To subtly adjust the high-level features of the backbone to better represent the unique characteristics of environments (e.g., light refraction, turbidity, and color distortion)

## 3. Advanced Optimizations
- **Mixed Precision Training (float16):** Implemented to utilize Tensor Cores on Colab T4 GPUs. This results in 2x-3x faster training and significantly lower VRAM consumption without loss in precision

- **Label Smoothing (0.1):** A regularization technique that prevents the model from becoming "overconfident" in its predictions. This improves generalization and helps the model handle noisy or ambiguous images more effectively

- **ReduceLROnPlateau:** Using ReduceLROnPlateau, the learning rate is halved whenever the validation accuracy plateaus, ensuring the model reaches the global minimum of the loss function

## 4. Output:

*Fig 1. Accuracy & Loss Learning Curves*

<img width="1590" height="733" alt="image" src="https://github.com/user-attachments/assets/618ab40a-fb35-45a0-b9d3-cdbcccd535e4" />



**1. Warm-up Phase Efficiency (Epochs 1-5):** During the initial frozen backbone phase, the model achieved a stable baseline accuracy of approximately 83-84%. This confirms that the custom Hybrid Pooling head (GAP + GMP) effectively integrated with the pre-trained features of MobileNetV2 from the onset

**2. Fine-tuning Breakthrough (Epoch 6 Onwards):** A significant "performance leap" is observed at Epoch 6, coinciding with the unfreezing of the top 30 layers. This architectural transition triggered a sharp decline in Loss and a steep increase in Accuracy, validating that the high-level features of MobileNetV2 successfully adapted to the waste domain

**3. Convergence and Generalization:**

- Peak Performance: The model reached its optimal state at Epoch 19, achieving a Validation Accuracy of 94.29% and a Validation Loss of 0.327
- Overfitting Control: Although a gap exists between training and validation metrics, the validation curve remains stable without divergence. This demonstrates the effectiveness of 0.5 Dropout and Label Smoothing (0.1) in maintaining high generalization capabilities despite the complexity of the dataset

**4. Conclusion:** Model 1 training was highly successful. The implementation of Early Stopping at Epoch 23 prevented model degradation, ensuring that the final weights represent the best possible balance between bias and variance

*Fig 2. MODEL 1 - Confusion Matrix (Validation Set)*

<img width="1295" height="984" alt="image" src="https://github.com/user-attachments/assets/0b502102-00b9-48aa-95e0-7ec897787824" />


*Fig 3. MODEL 1 - Performance Evaluation*

<img width="885" height="511" alt="image" src="https://github.com/user-attachments/assets/ae805490-7835-4f87-9070-f3b9ff3aabda" />

The model achieves a Global Accuracy of 95.11% on a balanced dataset of 12,800 images, demonstrating robust generalization across 8 material categories

*Battery and Trash* achieved near-perfect F1-scores (>99%). This indicates that the MobileNetV2 backbone successfully extracted unique morphological and textural features for these categories, even under suboptimal underwater lighting

*Textiles vs. Paper_cardboard:* This represents the primary source of error. 207 textile samples were misclassified as paper_cardboard

Reasoning: Submerged textiles often lose their structural definition and exhibit folding patterns similar to saturated cardboard. This leads to a lower Recall (78%) for the Textiles category

*Plastic vs. Glass:* There is a noticeable confusion where 61 plastic samples were predicted as glass.

Reasoning: The transparent and reflective properties of both materials create similar visual signatures, challenging the model's ability to distinguish between polymer and silica-based surfaces

*Paper_cardboard Precision (84%):* While the model successfully captured almost all paper samples (99% Recall), it suffered from high False Positives, frequently mislabeling textiles and plastics as paper

The high Macro Average F1-score (0.95) confirms that the model is not biased towards specific classes. The integration of Global Max Pooling was likely instrumental in capturing the subtle specular highlights needed to differentiate metals and glass from more matte materials like organic waste

*Fig 4: MODEL 1 - Random Successful Predictions*

<img width="2124" height="1838" alt="image" src="https://github.com/user-attachments/assets/51db4566-c05e-4e95-a4ff-731850ea1542" />


Figure 4 showcases the model's performance on unseen data. Most samples achieve a confidence score of 99-100%, indicating that the MobileNetV2 backbone has learned high-quality features for common waste objects like bottles, batteries, and metal cans

*Fig 5. MODEL 1 - Error Diagnosis*

<img width="2052" height="1634" alt="image" src="https://github.com/user-attachments/assets/bc54730c-b4a1-4432-b590-7e427d6c4cd3" />


Figure 5 highlights the 'Error Diagnosis' phase. By analyzing these 626 misclassified samples, we observe that the model primarily struggles with Textiles and Plastic when they appear in complex shapes or have textures similar to Paper/Cardboard. This insight is crucial for future model iterations and dataset balancing
