## A Comparative Study of Multi-Stage Lightweight CNN and Vision Transformer Model Approaches for Waste Classification and Grading

This project explores 5 deep learning models and 9 distinct system architectures to achieve high-precision waste classification and quality grading using the UWCD (Unified Waste Classification Dataset).

## 📂 Project Resources

All models and datasets are hosted on Google Drive. Please note the different formats based on the architecture:

- **Model 1 & 2 (CNN - MobileNetV2):** Provided as single .h5 files (Includes both architecture and weights)

- **Model 3, 4, & 5 (VLM - Vision Language Model):** Provided as Integrated Folders. These folders contain LoRA Adapters (compatible with Hugging Face peft library)

To use them, you must download the entire folder for each model

👉 https://drive.google.com/drive/folders/1OjmOsKOakRivIvJmrcQtkmwNkQH3bOsR?usp=drive_link

## 📚 Key References

If you find this work useful for your research, please refer to the foundational papers:

**Application of MobileNetV2 to waste classification**

- L. Yong, L. Ma, D. Sun, and L. Du
- Published: March 16, 2023
- DOI: 10.1371/journal.pone.0282336

**TrashVLM: Lightweight and Efficiently Fine-Tuned Vision-Language Models for Waste Classification**

- T. Trang, H. V. Pham, S. D. Vu, T. M. Le, H. M. Tran, and S. V. T. Dao
- 2025 International Conference on Advanced Technologies for Communications (ATC)
- DOI: 10.1109/ATC67618.2025.11268574

## 💠 System Components (5 Core Models)

This repository is organized into 5 primary modules, each representing a specific stage of research:

**MODEL 1:** CNN (MobileNetV2) - Material Classification

**MODEL 2:** CNN (MobileNetV2) - 10-Class Classification (Material + A/B Grading for Plastic & Metal)

**MODEL 3:** TrashVLM-HardCases - Vision Transformer (ViT) specialized in resolving CNN failures

**MODEL 4:** TrashVLM-Grading - 4-Class Material Assessment (Metal & Plastic A/B)

**MODEL 5:** TrashVLM-10Classes - 10-Class Vision Transformer (Material + A/B Grading for Plastic & Metal)

## 🔗 9 Research Architectures

We evaluated 9 architectural iterations to navigate the trade-offs between inference latency (FPS) and classification robustness (Accuracy):

### Architecture 1: Single-Stream CNN Baseline

- Component: Model 2 (CNN 10 Classes)

- Description: A standard end-to-end convolutional neural network serving as the control baseline for integrated material classification and quality grading

### Architecture 2: Hierarchical CNN Pipeline

- Workflow: Stage 1 (Model 1 - CNN 8 Classes) → Stage 2 (Model 2 - CNN 10 Classes)

- Description: A two-stage sequential approach designed to mitigate feature interference by separating primary material identification from granular grading

### Architecture 3: Confidence-based Hybrid Routing

- Workflow: Stage 1 (Model 2 - CNN 10 Classes) → Stage 2 (Model 3 - TrashVLM Hardcases)

- Logic: Employs Conditional Inference. If Stage 1 confidence is < 0.85, the "Hard Case" is routed to the VLM for diagnostic resolution

### Architecture 4: Decoupled VLM Grading

- Workflow: Stage 1 (Model 1 - CNN 8 Classes) → Stage 2 (Model 4 - VLM Grading)

- Description: A heterogeneous pipeline where a lightweight CNN handles general classification, while a specialized VLM focuses on grading (Metal/Plastic)

### Architecture 5: Monolithic Vision Transformer (ViT)

- Component: Model 5 (VLM 10 Classes)

- Description: An end-to-end Transformer-based model leveraging global self-attention mechanisms to process all categories in a single inference pass

### Architecture 6: Cross-Framework Integration

- Workflow: Stage 1 (Model 1 - CNN 8 Classes) → Stage 2 (Model 5 - VLM 10 Classes)

- Description: A hierarchical system combining the computational efficiency of CNNs for initial filtering with the robust parameter space of VLMs for final verification

### Architecture 7: Sequential Hybrid Verification

- Workflow: Stage 1 (Model 2 - CNN 10 Classes) → Stage 2 (Model 5 - VLM 10 Classes)

- Description: A complete dual-stage pipeline where every input undergoes a "double-check" process

### Architecture 8: Sequential Transformer Refinement

- Workflow: Stage 1 (Model 5 - VLM 10-Class) → Stage 2 (Model 4 - VLM 4-Class Grading)

- Description: A dual-stage Transformer pipeline utilizing a primary VLM for broad categorization and a secondary VLM for high-precision refinement of complex subclasses

### Architecture 9: Weighted Softmax Ensemble

- Workflow: Weighted Voting [Model 2 - CNN 10 Classes + Model 5 - VLM 10 classes] → Model 4 - VLM 4-Class Grading (Tie-breaker)

- Description: A multi-model ensemble system. Merges probabilistic outputs from all architectures using weighted voting to eliminate misclassification noise. This architecture aims to reach the theoretical performance ceiling by minimizing individual model bias

## 🛠️ Technical Stack

- **Frameworks:** TensorFlow/Keras, PyTorch, Hugging Face Transformers

- **Optimization:** QLoRA (4-bit Quantization), Mixed Precision Training, Class Weight Balancing

- **Backbones:** MobileNetV2, ViT-Base-Patch16-224

- **XAI:** Grad-CAM for error root-cause diagnosis
