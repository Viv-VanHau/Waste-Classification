### Constrained Batch_Size = 1 (Conveyor Belt environment: End-to-End System Latency)

| Architecture | Best Configuration | Accuracy (8-Class) | Accuracy (10-Class) | Frames per Second |
|-------------|--------------------|-------------------------|-----------------------------|-----------------|
| Arch 1: CNN Baseline | N/A | 89.70% | 77.10% | 9.27 FPS |
| Arch 2: (CNN 8 Classes) → (CNN 10 Classes) | Test 3 | 92.10% | 82.20% | 6.97 FPS |
| Arch 3: (CNN 10 Classes) → (ViT Hardcases) | Test 3 | 90.90% | 80.60% | 8.48 FPS |
| Arch 4: (CNN 8 Classes) → (ViT Grading) | Test 1 | 92.70% | 90.50% | 8.34 FPS |
| Arch 5: ViT Baseline | N/A | 97.80% | 92.70% | 42.88 FPS |
| Arch 6: (CNN 8 Classes) → (ViT 10 Classes) | Test 3 | 96.40% | 91.60% | 7.95 FPS |
| Arch 7: (CNN 10 Classes) → (ViT 10 Classes) | Test 1 | 97.90%| 93.00% | 6.61 FPS |
| **Arch 8: (ViT 10-Class) → (ViT Grading)** | Test 4 | 97.90% | 95.70% | 25.95 FPS |
| Arch 9: [CNN 10 Classes + ViT 10 classes] → (VLM Grading) | Test 4 | 98.00% | 96.50% | 6.83 FPS |

### Constrained Batch_Size = 1: End-to-End Latency vs. Pure GPU Throughput
| Architecture | End-to-End Latency | Pure GPU Throughput | Hardware Speedup Factor |
|---|---:|---:|---|
| Arch 1: CNN Baseline | 9.27 | 37.72 | ~4.0x |
| Arch 2: Test 3 | 6.97 | 46.55 | ~6.6x |
| Arch 3: Test 3 | 8.48 | 36.99 | ~4.3x |
| Arch 4: Test 1 | 8.34 | 34.43 | ~4.1x |
| Arch 5: ViT Baseline | 42.88 | 35.20 | Synchronized Penalty |
| Arch 6: Test 3 | 7.95 | 42.03 | ~5.2x |
| Arch 7: Test 1 | 6.61 | 31.22 | ~4.7x |
| Arch 8: Test 4 | 25.95 | 31.71 | ~1.2x |
| Arch 9: Test 4 | 6.83 | 25.63 | ~3.7x |

Hardware Scalability and Throughput Analysis (Pure GPU Throughput)
| Architecture | Acc-8-Class | Acc-10-Class | BS = 1 | BS = 8 | BS = 16 | BS = 32 | BS = 64 | Peak FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Arch 1 | 89.70% | 77.10% | 37.72 | 215.49 | 422.63 | 437.52 | 386.69 | Batch 32 (437.5 FPS) |
| Arch 2 | 92.10% | 82.20% | 46.55 | 26.78 | 10.83 | 11.27 | 12.51 | Batch 1 (46.5 FPS) |
| Arch 3 | 90.90% | 80.60% | 36.99 | 131.92 | 160.95 | 164.72 | 152.05 | Batch 32 (164.7 FPS) |
| Arch 4 | 92.70% | 90.50% | 34.43 | 116.53 | 127.67 | 132.30 | 118.68 | Batch 32 (132.3 FPS) |
| Arch 5 | 97.80% | 92.70% | 35.20 | 94.07 | 82.57 | 78.47 | 74.42 | Batch 8 (94.1 FPS) |
| Arch 6 | 96.40% | 91.60% | 42.03 | 126.17 | 127.89 | 131.15 | 127.40 | Batch 32 (131.1 FPS) |
| Arch 7 | 97.90% | 93.00% | 31.22 | 74.98 | 71.24 | 69.77 | 65.30 | Batch 8 (75.0 FPS) |
| Arch 8 | 97.90% | 95.70% | 31.71 | 60.38 | 57.66 | 55.42 | 51.80 | Batch 8 (60.4 FPS) |
| Arch 9 | 98.00% | 96.50% | 25.63 | 51.95 | 50.14 | 50.24 | 47.94 | Batch 8 (52.0 FPS) |
