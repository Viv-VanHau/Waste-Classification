import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

final_model_path = '/content/drive/MyDrive/Thesis_Project/Backbone_Stage1_Final_9429.h5'
model = tf.keras.models.load_model(final_model_path)
print(f"Loaded: {final_model_path}")

eval_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

val_gen_eval = eval_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

Y_pred = model.predict(val_gen_eval, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_gen_eval.classes
class_names = list(val_gen_eval.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 10))
sns.set_theme(style="white")

sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 11, "weight": "bold", "family": "serif"})

plt.title('MODEL 1: Confusion Matrix', fontsize=16, fontweight='bold', pad=20, family='serif')
plt.ylabel('Actual Category', fontsize=12, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=12, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('ARLO_Confusion_Matrix_Final.png', dpi=300)
plt.show()

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

report_df.to_csv('Model1_Report_Full.csv', index=True)

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
