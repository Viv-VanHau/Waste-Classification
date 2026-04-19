!pip install -q -U bitsandbytes>=0.46.1 torchao>=0.16.0 transformers peft datasets accelerate scikit-learn

import os
import glob
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

zip_path = '/content/drive/MyDrive/Thesis_Project/UWCD_10Classes_Final.zip'
extract_dir = '/content/UWCD_10_Classes_Temp'

if not os.path.exists(extract_dir):
    print("Extracting UWCD...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

base_data_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if len(dirs) == 10:
        base_data_dir = root
        break

dataset = load_dataset("imagefolder", data_dir=base_data_dir)
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

val_dataset = dataset['test']
labels = val_dataset.features["label"].names
final_classes = labels

output_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
reports_dir = '/content/drive/MyDrive/Thesis_Project/Outputs_Stage8'
os.makedirs(reports_dir, exist_ok=True)

checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
if not checkpoints:
    raise FileNotFoundError("Found None")
latest_ckpt = max(checkpoints, key=os.path.getmtime)
print(f"Processing: {os.path.basename(latest_ckpt)}")

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
base_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10,
    ignore_mismatched_sizes=True
)
vlm_model = PeftModel.from_pretrained(base_model, latest_ckpt)

def transforms(examples):
    pixel_values = [processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0] for img in examples["image"]]
    return {"pixel_values": pixel_values, "labels": examples["label"]}

val_dataset = val_dataset.with_transform(transforms)

eval_args = TrainingArguments(
    output_dir="/tmp/eval",
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    report_to="none"
)
trainer = Trainer(model=vlm_model, args=eval_args)

print(f"\nTesting on {len(val_dataset)} images...")
predictions, true_labels, _ = trainer.predict(val_dataset)
y_pred = np.argmax(predictions, axis=1)
y_true = true_labels

print(classification_report(y_true, y_pred, target_names=final_classes, digits=4))

cm = confusion_matrix(y_true, y_pred)

green_colors = ["#ffffff", "#e8f5e9", "#a5d6a7", "#4caf50", "#1b5e20"]
custom_purple_cmap = LinearSegmentedColormap.from_list("white_green", green_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
sns.heatmap(cm, annot=True, fmt='d', cmap=custom_purple_cmap,
            xticklabels=final_classes, yticklabels=final_classes,
            annot_kws={"size": 12, "weight": "bold", "family": "serif"},
            linewidths=0, cbar=True)

plt.title('VLM 10 Classes (Model 5)',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#4a148c')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()

cm_path = os.path.join(reports_dir, 'Model5_Validation_Matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"Done: {cm_path}")
plt.show()
