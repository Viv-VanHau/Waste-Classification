import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

best_model = load_model('/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5')

true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

predictions = best_model.predict(val_generator)
predicted_classes = np.argmax(predictions, axis=1)

print("CLASSIFICATION REPORT")
print(classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4))

print("\nPlotting Confusion Matrix...")
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(14, 11))

ax = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                 xticklabels=class_labels, yticklabels=class_labels,
                 annot_kws={"size": 12, "weight": "bold"},
                 linewidths=.5, linecolor='gray')

plt.title('Confusion Matrix - Stage 2 (Hierarchical Classification)',
          fontsize=18, pad=25, fontweight='bold', color='#333333')
plt.ylabel('True Labels', fontsize=14, fontweight='bold', color='#555555')
plt.xlabel('Predicted Labels', fontsize=14, fontweight='bold', color='#555555')

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()

cm_path = '/content/drive/MyDrive/Thesis_Project/Confusion_Matrix_Model2.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
