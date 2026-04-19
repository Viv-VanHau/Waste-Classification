Model 3 is designed to handle the "Hard Cases" that the Model 2 CNN (10-class model) struggled to classify. Instead of training on the entire dataset, we focus specifically on Misclassified Samples (Plastic and Metal errors) to understand and mitigate the root causes of failure using Explainable AI (XAI) and Heuristic Analysis.

## 1. Pre-processing & Error Diagnosis

### Stage 1: Error Mining

Utilized the best weights from Model 2 (Stage2_10Classes_Best.h5) to perform inference on the validation set. By comparing Ground Truth against Predictions, the model isolated 314 critical errors (183 Plastic, 131 Metal) for deeper inspection

### Stage 2: XAI Diagnosis (Grad-CAM)

To understand "Why the model failed," I implemented Gradient-weighted Class Activation Mapping (Grad-CAM)

- **Mechanism:** extracted gradients from the last convolutional layer of the MobileNetV2 backbone
- **Purpose:** By visualizing the "Heatmaps" of misclassified images, I identified whether the model was focusing on the actual object or was distracted by background noise and reflections

### Stage 3: Automated Heuristic Root Cause Analysis

Developed an automated diagnostic engine using OpenCV to categorize errors into four fundamental taxonomies based on Grad-CAM heatmaps:

- **Label Interference:** High edge density within the activation mask (Canny Edge Detection), indicating complex shapes or overlapping objects

- **Specular Reflection:** High pixel intensity (Brightness thresholding) within the "hot" zones, common in underwater metallic/plastic surfaces

- **Background Bias:** Measuring the Euclidean distance between the heatmap's centroid and the image center. If the model focuses on the corners, it is biased by the background

- **Boundary Ambiguity:** Cases where the model fails to find a clear focal point or the object boundaries are too turbid/blurry

### Stage 4: Strategic Oversampling (Targeted Buffing)

To prepare the dataset for the subsequent Vision-Language Model (VLM) training, I implemented a targeted oversampling strategy to balance the "Hard Case" distribution, ensuring at least 50 samples per minority error class.

## 2. Pre-processing Output

The following table summarizes the automated diagnostic results for the detected misclassifications:

*Table 1. Root Cause Distribution*
<img width="600" height="150" alt="image" src="https://github.com/user-attachments/assets/b6f7009c-263f-4d7d-a5d0-a4cd5785a214" />

*Figure 1. Root Cause Distribution Plotting*
<img width="1300" height="750" alt="image" src="https://github.com/user-attachments/assets/8069dfab-e782-4d9a-857c-095fb942972f" />

*Figure 2. Highly Confident Prediction Cases*
<img width="2308" height="1283" alt="image" src="https://github.com/user-attachments/assets/a09d3019-a698-4179-9b02-d4dc6dd97f5a" />

*Figure 3. Grad-CAM for error analysis - Metal Cases*
<img width="1834" height="1329" alt="image" src="https://github.com/user-attachments/assets/3bacd419-6df0-46a3-bf3e-ed11878611c4" />



