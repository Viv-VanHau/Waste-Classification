## A Comparative Study of Multi-Stage Lightweight CNN and Vision Transformer Model Approaches for Waste Classification and Grading

This project explores 5 deep learning models and 9 distinct system architectures to achieve high-precision waste classification and quality grading using the UWCD (Unified Waste Classification Dataset).

## 📂 Project Resources

All models and datasets are hosted on Google Drive. Please note the different formats based on the architecture:

- **Model 1 & 2 (CNN - MobileNetV2):** Provided as single .h5 files (Includes both architecture and weights)

- **Model (3), 4 & 5 (ViT - Vision Transformer):** Provided as Integrated Folders. These folders contain LoRA Adapters (compatible with Hugging Face peft library)

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

## 💠 System Components (Core Models)

This repository is organized into 5 primary modules, each representing a specific stage of research:

**MODEL 1:** CNN (MobileNetV2) - Material Classification

**MODEL 2:** CNN (MobileNetV2) - 10-Class Classification (Material + A/B Grading for Plastic & Metal)

**MODEL 3:** TrashVLM-HardCases - Vision Transformer (ViT) specialized in resolving CNN failures (**dismissed**)

**MODEL 4:** TrashVLM-Grading - 4-Class Material Assessment (Metal & Plastic A/B)

**MODEL 5:** TrashVLM-10Classes - 10-Class Vision Transformer (Material + A/B Grading for Plastic & Metal)

## 🔗 Research Architectures

We evaluated 9 architectural iterations to navigate the trade-offs between inference latency (FPS) and classification robustness (Accuracy):

<img width="1252" height="1034" alt="image" src="https://github.com/user-attachments/assets/a003931f-415a-4c5c-9b8a-f12deb353716" />
