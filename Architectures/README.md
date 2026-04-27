### Constrained Batch_Size = 1 (Conveyor Belt environment)

| Architecture | Best Configuration | Accuracy (8-Class) | Accuracy (10-Class) | Frames per Second |
|-------------|--------------------|-------------------------|-----------------------------|-----------------|
| Arch 1: CNN Baseline | N/A | 89.70% | 77.10% | 9.27 FPS |
| Arch 2: (CNN 8 Classes) → (CNN 10 Classes) | Test 3 | 90.50% | 82.20% | 6.97 FPS |
| Arch 3: (CNN 10 Classes) → (ViT Hardcases) | Test 3 | 88.70% | 80.60% | 8.48 FPS |
| Arch 4: (CNN 8 Classes) → (ViT Grading) | Test 1 | 92.70% | 90.50% | 8.34 FPS |
| Arch 5: ViT Baseline | N/A | 97.80% | 92.70% | 42.88 FPS |
| Arch 6: (CNN 8 Classes) → (ViT 10 Classes) | Test 3 | 93.20% | 91.60% | 7.95 FPS |
| Arch 7: (CNN 10 Classes) → (ViT 10 Classes) | Test 1 | 97.80%| 93.00% | 6.61 FPS |
| **Arch 8: (ViT 10-Class) → (ViT Grading)** | Test 4 | 97.90% | 95.70% | 25.95 FPS |
| Arch 9: [CNN 10 Classes + ViT 10 classes] → (VLM Grading) | Test 4 | 98.00% | 96.50% | 6.83 FPS |
