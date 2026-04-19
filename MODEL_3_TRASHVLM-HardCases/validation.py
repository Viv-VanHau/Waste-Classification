import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

predictions = trainer.predict(val_ds)
val_preds = np.argmax(predictions.predictions, axis=1)
val_labels = predictions.label_ids

present_indices = sorted(list(set(val_labels) | set(val_preds)))
present_names = [class_labels[i] for i in present_indices]

acc = (val_preds == val_labels).mean() * 100
print(f"\nAccuracy: {acc:.2f}%")

cm = confusion_matrix(val_labels, val_preds, labels=present_indices)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 linewidths=1, linecolor='white',
                 cbar_kws={"shrink": .8},
                 xticklabels=present_names,
                 yticklabels=present_names)

plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
plt.yticks(rotation=0, fontsize=11, fontweight='bold')

plt.title('TrashVLM Confusion Matrix\n(Hard Examples Focus)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold', labelpad=10)
plt.ylabel('True Label', fontsize=13, fontweight='bold', labelpad=10)

cm_path = os.path.join(output_model_dir, 'MODEL_3.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()

print("CLASSIFICATION REPORT")
print(classification_report(val_labels, val_preds,
                            labels=present_indices,
                            target_names=present_names,
                            zero_division=0))
