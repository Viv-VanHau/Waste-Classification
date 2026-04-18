## 🛠 Quick Start
- Hardware: Google Colab (T4 GPU enabled)
- Execution: Run the provided train.py. The best-performing model will be saved automatically as 'Stage2_10Classes_Best.h5'

This model involves a more detailed classification of the UWCD Dataset. The original Metal and Plastic categories have been sub-divided into *Grade A and Grade B*, resulting in a 10-class problem. This model addresses the significant challenge of extreme data imbalance through cost-sensitive learning

## 1. Dataset Engineering

### Manual Data Refinement
A manual audit was conducted to remove mislabeled images in the Metal and Plastic categories. This ensures that the model learns from high-quality, representative samples, despite the reduction in sample size for minority classes

### Class Distribution & Imbalance

The dataset exhibits a severe "Long-Tail" distribution:

<img width="408" height="380" alt="image" src="https://github.com/user-attachments/assets/ebd62c61-ac94-4d1b-997d-226704efb5cf" />

This reflects the rarity of clean metal/plastic in reality.

### Data Pipeline & Reproducibility

- **Random Shuffling:** Utilized the split-folders library to perform a deterministic 80/20 Train/Val split with seed=42. This ensures that the evaluation is consistent and reproducible
- **Augmentation Strategy:** To compensate for the lack of data in Grade A categories, the ImageDataGenerator was configured with:
- **Spatial Transformations:** Rotation (20°), Width/Height Shifts (0.2), and Shear (0.2)
- **Morphological Transformations:** Zoom (0.2) and Horizontal Flips
- **Normalization:** Rescaling pixel values to [0, 1]

### Class Weights          

To prevent the model from ignoring minority classes, we implemented Class Weighting. By using the 'balanced' heuristic from sklearn, we assigned a higher "penalty" to misclassifications of rare classes:

<img width="480" height="200" alt="image" src="https://github.com/user-attachments/assets/37ee7824-7ea3-4ba8-87ba-e2ada0de0f75" />

The loss function treats one mistake in Plastic Grade A as equivalent to ~32 mistakes in a majority class.

## 2. Two-Phase Training

Instead of training from ImageNet weights, Model 2 performs Sequential Transfer Learning. The model loads the pre-trained weights from Model 1 (8-class, 94.29% accuracy)

The original 8-class output layer was removed. A new Dense layer (10 neurons) with Softmax activation was integrated

### Phase 1: Knowledge Inheritance & Head Adaptation

- **Backbone Status:** Frozen (trainable = False)
- **Learning Rate:** 1e-3 (Adam Optimizer)
- **Mechanism:** By freezing the backbone and using a relatively high learning rate of 1e-3, focus solely on training the new 10-neuron classification head. This ensures the robust feature extraction layers remain intact while the new head learns the boundaries between the 10 material classes

### Phase 2: Cost-Sensitive Learning

- **Technique:** Class Weight Optimization
- **Safeguards:** 'ModelCheckpoint' and 'EarlyStopping' (patience=5) are active to monitor val_accuracy, ensuring the model achieves the best balance between majority-class precision and minority-class recall without overfitting
- These weights amplify the gradients during backpropagation for rare samples. This effectively "buffs" the model’s attention, forcing the optimizer to prioritize the subtle textures and visual nuances that distinguish Grade A materials from Grade B, even with limited data

## 3. A/B Grading

This model implemented a granular classification for Metal and Plastic. Below is the specific criteria used for manual data labeling:

<img width="870" height="206" alt="image" src="https://github.com/user-attachments/assets/5a08947e-7e5a-411a-8286-7ad93e653f86" />

### Rationale: Why Focus Only on Metal & Plastic Grading?

The decision to refine grading specifically for Metal and Plastic within Reverse Logistics model is driven by two strategic pillars:

**1. High Economic Variance:**

- Metal and Plastic exhibit the highest price volatility based on purity
  
- Grade A materials can be sold at a 20-40% premium compared to mixed batches due to lower processing costs for recyclers
  
- For materials like Paper or Glass, the cost of fine-sorting often exceeds the marginal increase in market value, making complex grading economically unviable for those streams

**2. Robotic Handling & Industrial Safety:** Metal and Plastic carry the highest operational risks in automated facilities:
- Pressure Hazards: A capped plastic bottle or a sealed metal tin can cause structural damage to industrial shredders/compactors due to internal pressure build-up
  
- Mechanical Safety: Deformed or jagged metal (Grade B) poses a severe risk to conveyor belts and robotic grippers. By isolating Grade A (Intact), ensuring maximum throughput for automated robotic picking

## 4. Output

*Fig 1. MODEL 2 - Confusion Matrix*
<img width="937" height="25" alt="image" src="https://github.com/user-attachments/assets/28497422-db2c-4b06-ac99-91658f20201a" />
<img width="937" height="730" alt="image" src="https://github.com/user-attachments/assets/cb9e0584-0e62-4abd-ba50-083a31bfe12d" />

*Fig 2. MODEL 2 -*
<img width="580" height="350" alt="image" src="https://github.com/user-attachments/assets/89a9dd3e-974a-45a9-8e86-cec2940b50f7" />


- Global Accuracy: 94.61%
- Weighted Average F1-Score: 95.07%
- Macro Average Recall: 86.97%

Without the implemented class weights (Plastic Grade A buffed by 26.02x), the model would likely have ignored these classes entirely (0% Recall). Achieving ~41% recall on only 49 validation samples proves that the model has learned the specific "Grade A" features (transparency, integrity) rather than just bias

The Confusion Matrix reveals "Semantic Overlap" between grades of the same material:

- **Metal A vs. Metal B:** 40 samples of Metal Grade A were misclassified as Grade B. More importantly, 197 samples of Grade B were predicted as Grade A

Reasoning: Subtle rust or minor deformations in Grade B can sometimes be overlooked by the model, leading to "false premium" labels

- **Plastic A vs. Plastic B:** 77 samples of Plastic Grade B were misclassified as Grade A

Reasoning: High-quality standard plastic (Grade B) can visually mimic the transparency and structural integrity of PET Grade A under certain underwater lighting conditions

Majority classes like Battery (99.5% F1), Organic (98.3% F1), and Textiles (97.2% F1) remain unaffected by the granular split of other classes. This confirms that the MobileNetV2 backbone is highly robust and the added 10-class complexity did not cause "feature interference" across different material types

*Fig 3. MODEL 2 - TRAINING HISTORY*
<img width="1489" height="635" alt="image" src="https://github.com/user-attachments/assets/009a7691-8838-4d87-b9ce-76078d4aef19" />

During these first 5 epochs, the training and validation curves showed a steady upward trend. The new 10-class head quickly learned to differentiate between the 8 original categories while beginning to grasp the subtle differences between Grade A and Grade B for Metal and Plastic.

At Epoch 9, the model reached its absolute peak performance with a Validation Accuracy of ~94.6% and its lowest Validation Loss of ~0.16. The gap where Validation Accuracy > Training Accuracy remained consistent. This is a positive indicator that the heavy Data Augmentation and Dropout (0.5) applied during training were successfully forcing the model to learn robust, generalized features rather than memorizing the training samples.

After Epoch 9, the Validation Accuracy began to fluctuate and exhibited a slight downward trend, while the Validation Loss started to creep back up. Since the Validation Accuracy did not improve beyond the peak reached at Epoch 9 for 5 consecutive epochs (from Epoch 10 to 14), the training was automatically terminated. This intervention was crucial to prevent overfitting. Since the minority classes (Plastic/Metal Grade A) are heavily weighted (up to 26x), continuing training would have forced the model to become biased toward specific noise in those small samples, sacrificing the overall stability of the majority classes.
