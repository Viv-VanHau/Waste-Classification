TRASHVLM-10Classes represents the pinnacle of our classification hierarchy. It leverages the high-capacity Vision Transformer (ViT) architecture to perform end-to-end classification and quality grading (Grade A/B) across the entire UWCD Dataset spectrum.

## 1. Technical Architecture

To handle the complexity of 10 simultaneous categories, including the subtle features required for quality grading, we implemented a state-of-the-art Transformer pipeline:

- **Backbone:** google/vit-base-patch16-224-in21k (Vision Transformer)

- **Parameter-Efficient Fine-Tuning (PEFT):**

- **QLoRA (4-bit Quantization):** Utilizing bitsandbytes (NF4) to compress the massive ViT model, allowing high-performance training within the VRAM limits of a T4 GPU

- **LoRA Configuration:** Rank (r) = 32 and Alpha = 64. By targeting the query and value projection matrices in the self-attention blocks, the model learns complex inter-class relationships without the risk of catastrophic forgetting

### Advanced Imbalance Management

Handling 10 classes with varying sample sizes (from 245 to 8,000 images) requires a robust cost-sensitive approach:

- **Automated Weight Calibration:** The system utilizes sklearn to dynamically calculate Balanced Class Weights. This ensures that minority classes like plastic_Grade_A and metal_Grade_A receive proportional "buffing" in the loss function

- **Custom WeightedTrainer:** We injected these weights into a specialized Trainer subclass. This forces the Cross-Entropy loss function to penalize errors on rare, high-value items significantly more than majority classes

### Training Configuration & Pipeline

- **Framework:** Fully integrated with the Hugging Face ecosystem (transformers, peft, datasets)

- **Efficient Data Loading:** Used the imagefolder API and with_transform for non-blocking, on-the-fly image preprocessing (Normalization, Resizing)

- **Optimizer:** AdamW with a learning rate of 1e-4 and FP16 Mixed Precision enabled for maximum throughput

- **Metric Priority:** The model is optimized for F1-Macro, ensuring a balanced performance across all 10 grades rather than just chasing raw accuracy

## 2. Output

*Fig 1. MODEL 5 - Learning Curves*
<img width="1590" height="590" alt="image" src="https://github.com/user-attachments/assets/4ab3bfb9-9473-4e0d-8c4a-02eb3aa83c3d" />

Loss Behavior: The training loss shows a rapid and healthy descent. Interestingly, the Validation Loss remains consistently low and stable (below 0.2), even as the training loss continues to decrease. This "Generalization Gap" confirms that our Dropout (0.1) and LoRA regularization effectively prevented the model from memorizing noise.

Steady Convergence: Unlike the CNN models which exhibited fluctuations, the ViT performance metrics (Accuracy & F1-Macro) show a smooth, logarithmic climb. The model reaches its "diminishing returns" point at Epoch 5, maintaining peak stability through to Epoch 9.

*Fig 2. MODEL 5 - Confusion Matrix*
<img width="850" height="850" alt="image" src="https://github.com/user-attachments/assets/bec9597d-b5c5-45c3-ae41-d9040e605fdf" />

*Fig 3. MODEL 5 - Performance Evaluation*
<img width="650" height="380" alt="image" src="https://github.com/user-attachments/assets/e1110226-159e-471b-b64f-45496f719d82" />

By replacing the traditional CNN backbone with a Vision Transformer (ViT) and applying QLoRA Fine-tuning, we have achieved the highest performance across the entire 10-class hierarchy: 

- **Global Accuracy:** 97.03% (An improvement of ~2.4% over the Stage 2 CNN)

- **Macro Average F1-Score:** 0.9246

- **Weighted Average F1-Score:** 0.9700

The 10-class green heatmap demonstrates near-perfect class separation:

- Battery: Near-perfect safety detection
- Textiles: Exceptional texture recognition
- Organic Waste:	High purity for contamination control
- Metal Grade B: Reliable industrial-scale sorting

The most significant achievement of Model 5 is its ability to handle the "Long-Tail" minority classes:

- **Metal Grade A:** Achieved a Recall of 77.04% and an F1-score of 0.803. This is a massive improvement over the CNN version, proving the Transformer's ability to detect structural integrity in metal

- **Plastic Grade A:** Reached a Recall of 53.57%. While still the most challenging class due to its extreme scarcity (only 56 samples), the Precision (85.7%) is outstanding, ensuring that when the model identifies "Premium Plastic," it is almost certainly correct

Model 2 (CNN) struggled with global context. Model 5’s Self-Attention mechanism allows it to relate different parts of an image (e.g., a cap, a label, and a reflection) to make a much more informed "Grade A" vs "Grade B" decision. The ViT architecture proved more resilient to light scattering and turbidity. The high precision in the Glass (0.97 F1) and Plastic (0.94 F1) categories shows it can "see through" the visual distortions that often confused the CNN. The use of QLoRA (r=32) provided the model with enough "trainable capacity" to learn the nuances of 10 classes without losing the general visual knowledge of the pre-trained ViT.
