# Waste Classification & Quality Grading

This project explores 5 deep learning models and 6 distinct system architectures to achieve high-precision waste classification and quality grading using the UWCD (Unified Waste Classification Dataset).

## 📂 Project Resources

All models and datasets are hosted on Google Drive. Please note the different formats based on the architecture:

- **Model 1 & 2 (CNN - MobileNetV2):** Provided as single .h5 files (Includes both architecture and weights)

- **Model 3, 4, & 5 (VLM - Vision Transformer):** Provided as Integrated Folders
These folders contain LoRA Adapters (compatible with Hugging Face peft library)

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

## 💠 System Components (The 5 Core Models)

This repository is organized into 5 primary modules, each representing a specific stage of research:

**MODEL 1:** CNN (MobileNetV2) - Material Classification

**MODEL 2:** CNN (MobileNetV2) - 10-Class Classification (Material + A/B Grading for Plastic & Metal)

**MODEL 3:** TrashVLM-HardCases - Vision Transformer (ViT) specialized in resolving CNN failures

**MODEL 4:** TrashVLM-Grading - 4-Class Material Assessment (Metal & Plastic A/B)

**MODEL 5:** TrashVLM-10Classes - 10-Class Vision Transformer (Material + A/B Grading for Plastic & Metal)

## 🔗 The 6 Research Architectures

We evaluated 6 different architectural approaches to find the optimal balance between speed, safety, and economic recovery value:

### Architecture 1: Single-Stream CNN Baseline

- Component: Model 2 (CNN 10 Classes)

- Goal: Establish a baseline for all-in-one classification and grading using a mobile-optimized CNN architecture

### Architecture 2: Dual-Stage Sequential CNN

- Workflow: Stage 1 (Model 1 - CNN 8 Classes) → Stage 2 (Model 2 - CNN 10 Classes)

- Goal: Implement a tiered filtering system where basic material identification precedes granular quality grading to reduce feature interference

### Architecture 3: Confidence-Routed Hybrid System

- Workflow: Stage 1 (Model 2 - CNN 10 Classes) → Stage 2 (Model 3 - TrashVLM Hardcases)

- Logic: Employs Conditional Inference. If Stage 1 confidence is < 0.85, the "Hard Case" is routed to the VLM for expert-level diagnostic resolution

### Architecture 4: Precision Grading Pipeline

- Workflow: Stage 1 (Model 1 - CNN 8 Classes) → Stage 2 (Model 4 - TrashVLM Grading)

- Goal: Identifying basic materials first, then using a specialized VLM to ensure maximum precision for high-value Metal and Plastic grades

### Architecture 5: End-to-End Vision Transformer (ViT)

- Component: Model 5 (TrashVLM 10 Classes)

- Goal: Testing the power of Self-Attention as a single-stage solution for both material classification and grading

### Architecture 6: Cross-Architectural Hierarchical Pipeline

- Components: Stage 1 (Model 1 - CNN 8 Classé) → Stage 2 (Model 5 - VLM 10 Classes)

- Goal: Leverage the speed of CNNs for primary sorting and the massive parameter space of VLMs for secondary validation, ensuring the highest possible system-wide robustness and error recovery

## 🛠️ Technical Stack

- **Frameworks:** TensorFlow/Keras, PyTorch, Hugging Face Transformers

- **Optimization:** QLoRA (4-bit Quantization), Mixed Precision Training, Class Weight Balancing

- **Backbones:** MobileNetV2, ViT-Base-Patch16-224

- **XAI:** Grad-CAM for error root-cause diagnosis

***Note:** This project is part of a research thesis on intelligent Reverse Logistics and automated waste sorting systems*
