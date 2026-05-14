# DISMISSED

## 🛠 Quick Start
- Hardware: Google Colab (T4 GPU enabled)
- Execution: Run the provided train.py. The best-performing model will be saved automatically as 'TrashVLM_LoRA_Output.h5'

Model 3 is designed to handle the "Hard Cases" that the Model 2 CNN (10-class model) struggled to classify. Instead of training on the entire dataset, we focus specifically on Misclassified Samples (Plastic and Metal errors) to understand and mitigate the root causes of failure using Explainable AI (XAI) and Heuristic Analysis.

## 1. Pre-processing & Error Diagnosis

### Stage 1: Error Mining

Utilized the best weights from Model 2 (Stage2_10Classes_Best.h5) to perform inference on the validation set. By comparing Ground Truth against Predictions, the model isolated 314 critical errors (183 Plastic, 131 Metal) for deeper inspection

### Stage 2: XAI Diagnosis (Grad-CAM)

To understand "Why the model failed," we implemented Gradient-weighted Class Activation Mapping (Grad-CAM)

- **Mechanism:** extracted gradients from the last convolutional layer of the MobileNetV2 backbone
- **Purpose:** By visualizing the "Heatmaps" of misclassified images, I identified whether the model was focusing on the actual object or was distracted by background noise and reflections

### Stage 3: Automated Heuristic Root Cause Analysis

Developed an automated diagnostic engine using OpenCV to categorize errors into four fundamental taxonomies based on Grad-CAM heatmaps:

- **Label Interference:** High edge density within the activation mask (Canny Edge Detection), indicating complex shapes or overlapping objects

- **Specular Reflection:** High pixel intensity (Brightness thresholding) within the "hot" zones, common in underwater metallic/plastic surfaces

- **Background Bias:** Measuring the Euclidean distance between the heatmap's centroid and the image center. If the model focuses on the corners, it is biased by the background

- **Boundary Ambiguity:** Cases where the model fails to find a clear focal point or the object boundaries are too turbid/blurry

### Stage 4: Strategic Oversampling (Targeted Buffing)

To prepare the dataset for the subsequent Vision-Language Model (VLM) training, I implemented a targeted oversampling strategy to balance the "Hard Case" distribution, ensuring at least 50 samples per minority error class

### Stage 5: Semantic Dataset Structuring for VLM Injection
After the diagnostic phase, the "Hard Cases" are physically reorganized into a hierarchical directory structure. This structure is designed to facilitate Targeted Fine-Tuning for Model 3 (VLM)

The dataset is split into a multi-level hierarchy: Material Type → Error Root Cause

- Plastic: Label_Interference, Specular_Reflection, Background_Bias, Boundary_Ambiguity

- Metal: Label_Interference, Specular_Reflection, Background_Bias, Boundary_Ambiguity

To assist in error-specific analysis during VLM training, each file is renamed using a strictly formatted metadata string:

- Format: True-{Label}___Pred-{Label}___Conf-{Score}_{Original_Name}

- Example: True-plastic_Grade_A___Pred-paper_cardboard___Conf-98.6_img_102.jpg

By structuring the data this way, we enable Model 3 to perform "Failure-Aware Learning." The VLM can now process images not just as raw pixels, but as specific examples of CNN failures. This structured repository at Stage6_VLM_Training_Data serves as the primary knowledge base for the subsequent VLM fine-tuning process, aiming to resolve the "Hard Cases" that traditional CNNs cannot handle

## 2. Training Techniques

### 1. Multi-Stream Data Pipeline & CLAHE

Model 3 uses an Error-Aware Pipeline

- **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Integrated specifically for images tagged with *Specular_Reflection*

- **Mechanism:** By converting images to LAB color space and applying CLAHE to the L-channel (Lightness), the model effectively neutralizes underwater glare and light scattering, restoring hidden textures on metallic and plastic surfaces

### 2. Adaptive Transform Strategy

We implemented a dynamic augmentation strategy where the transformation type is determined by the Error Root Cause identified in the previous stage:

- **Specular Reflection:** Applied CLAHE + Standard Augmentation

- **Label Interference & Background Bias:** Applied 'Heavy Crop Transforms (scale 0.5 - 0.8)' and 'Color Jitter'. This forces the Transformer to focus on local object patches rather than being distracted by complex backgrounds or overlapping labels

- **Standard Errors:** Applied 'Random Resized Crops' and 'Horizontal Flips'

### 3. Risk-Weighted Loss (WeightedTrainer)

To align with industrial safety standards, we implemented a custom WeightedTrainer with a Security Coefficient (w) for high-risk materials:

- Battery (w=20.0): Highest priority due to chemical leakage risks

- Organic (w=15.0): High priority to prevent contamination of recyclable batches

- Plastic_Grade_A (w=12.0): Priority based on high economic recovery value

The model is penalized up to 20 times more for missing a Battery than a standard waste item

### 4. Adaptive QLoRA (PEFT)

To train a heavy Transformer model on consumer-grade hardware (GPU T4), we utilized Parameter-Efficient Fine-Tuning (PEFT):

- **4-bit Quantization (NF4):** Uses BitsAndBytesConfig to compress the ViT-Base model, reducing VRAM usage by over 70% while maintaining 16-bit compute precision

- **LoRA (Low-Rank Adaptation):**

**Rank (r)=16, Alpha=64:** Injected trainable low-rank matrices into the query and value layers of the Attention mechanism

**Trainable Modules:** Only the LoRA adapters and the final classifier head are updated, preserving the foundational "visual knowledge" of the pre-trained ViT

### 5. Training Configuration

- **Framework:** Integrated via 'Hugging Face Ecosystem' (Transformers, PEFT, Datasets, and BitsAndBytes)

- **Base Architecture:** google/vit-base-patch16-224-in21k

- **Optimization:** AdamW with Weight Decay (0.05)

- **Learning Rate:** 1e-4 with a Warmup Ratio (0.2) to stabilize early training on hard cases

- **Epochs:** 70 (Iterative refinement)

- **Batch Management:** Effective batch size of 32 (8 per device × 4 gradient accumulation steps)

## 3. Pre-processing Output

The following table summarizes the automated diagnostic results for the detected misclassifications:

*Table 1. Root Cause Distribution*
<img width="600" height="150" alt="image" src="https://github.com/user-attachments/assets/b6f7009c-263f-4d7d-a5d0-a4cd5785a214" />

*Figure 1. Root Cause Distribution Plotting*
<img width="1300" height="750" alt="image" src="https://github.com/user-attachments/assets/8069dfab-e782-4d9a-857c-095fb942972f" />

*Figure 2. Highly Confident Prediction Cases*
<img width="2308" height="1283" alt="image" src="https://github.com/user-attachments/assets/a09d3019-a698-4179-9b02-d4dc6dd97f5a" />

*Figure 3. Grad-CAM for error analysis - Metal Cases*
<img width="1834" height="1329" alt="image" src="https://github.com/user-attachments/assets/3bacd419-6df0-46a3-bf3e-ed11878611c4" />

Before proceeding to Model 3 training, we performed a "Targeted Buffing" to ensure data sufficiency for the Vision-Language Model:
<img width="550" height="120" alt="image" src="https://github.com/user-attachments/assets/6952a409-949d-4d58-92de-f48b107389dc" />

## 4. Training Output

*Fig 1. Textures before and after CLAHE*
<img width="650" height="250" alt="image" src="https://github.com/user-attachments/assets/33d8ddcf-5469-4baf-8ad0-cd89c3b75abf" />
<img width="650" height="250" alt="image" src="https://github.com/user-attachments/assets/6482dcc0-5a51-42e1-a118-2fd367a4677d" />

*Fig 2. MODEL 3 - Training Curves*
<img width="850" height="950" alt="image" src="https://github.com/user-attachments/assets/d7ebdb6b-ab6b-4b68-a633-4b6f06423364" />

The learning process for the VLM exhibits a highly stable and progressive convergence:

- **Loss Convergence:** The Training Loss started at a high baseline (~9.25) due to the complexity of the hard cases but dropped sharply, stabilizing below 1.0. The Validation Loss remained consistently low and smooth, indicating that the Low-Rank Adaptation (LoRA) effectively prevented "Catastrophic Forgetting" while fine-tuning on a small, niche dataset

- **Accuracy Climb:** Starting from a near-zero baseline, the model reached a performance plateau of 83% at Epoch 44 (marked by the dashed line). The steady climb proves that the Adaptive CLAHE and Heavy-Crop transforms successfully guided the Transformer's attention to the relevant material textures

*Fig 3. MODEL 3 - Confusion Matrix*
<img width="550" height="580" alt="image" src="https://github.com/user-attachments/assets/b5043da7-3ee3-481b-8fca-1835c85e5fde" />

*Fig 4. Model 3 - Performance Evaluation*
<img width="550" height="250" alt="image" src="https://github.com/user-attachments/assets/1d994b00-13da-4ce9-bdc0-0d6c76457575" />

The "Hard Examples Focus" matrix reveals the model's granular decision-making:

- Metal Grade A: Outstanding Performance. The VLM excelled at identifying premium metal, even those previously confused by the CNN
- Plastic Grade B:	Highly Robust. High precision confirms the model can distinguish standard plastic despite background noise
- Plastic Grade A:	Moderate success. Scarcity (6 samples) remains the primary challenge for this category
- Metal Grade B: The model struggled here due to extremely low support (4 samples) and high visual similarity to Trash

While CNNs focus on local textures, the ViT’s Self-Attention mechanism allowed it to look at the "big picture", connecting disparate specular highlights to identify a material's true nature. By neutralizing reflections via CLAHE, we provided the VLM with a "cleaner" signal for the most difficult metallic and plastic surfaces. The high weights assigned to Metal and Plastic Grade A forced the model to prioritize these high-value streams, resulting in significantly higher Recall for premium recyclables.
