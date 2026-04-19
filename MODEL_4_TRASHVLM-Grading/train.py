# ===============================
# 1. A/B GRADING LAYER EXTRACTION
# ===============================
import os
import zipfile
import shutil
import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight

zip_path = '/content/drive/MyDrive/Thesis_Project/UWCD_10Classes_Final.zip'
raw_extract_dir = '/content/UWCD_Raw_Temp'
vlm_data_dir = '/content/TrashVLM_4Classes_Data'

target_classes = ['metal_Grade_A', 'metal_Grade_B', 'plastic_Grade_A', 'plastic_grade_B']

if not os.path.exists(raw_extract_dir):
    print("Extracting UWCD...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_extract_dir)

base_data_dir = raw_extract_dir
for root, dirs, files in os.walk(raw_extract_dir):
    if len(dirs) == 10:
        base_data_dir = root
        break

print("Extracting A/B Metal Plastic...")
if os.path.exists(vlm_data_dir):
    shutil.rmtree(vlm_data_dir)
os.makedirs(vlm_data_dir)

total_images = 0
for cls in target_classes:
    src_dir = os.path.join(base_data_dir, cls)
    dst_dir = os.path.join(vlm_data_dir, cls)
    shutil.copytree(src_dir, dst_dir)
    num_files = len(os.listdir(dst_dir))
    total_images += num_files
    print(f"{cls:<18} : {num_files} images")

# Hugging Face Dataset (Automate splitting 80/20)
dataset = load_dataset("imagefolder", data_dir=vlm_data_dir)

dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['test'] # Đổi tên test thành validation
})

labels = dataset["train"].features["label"].names
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}

print(f"Mapping label: {id2label}")

# Class Weight Automate Calculation
print("\nCalculating weight for imbalance class...")
train_labels = dataset['train']['label']
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

for i, w in enumerate(class_weights):
    print(f"Weight {labels[i]:<18} : {w:.4f}")
print("Done, ready for training Model 4")

# =====================================
# 2. TRAINING TRASH-VLM (GRADING MODEL)
# =====================================
!pip install -q -U bitsandbytes>=0.46.1 transformers peft datasets accelerate

import os
import torch
import torch.nn as nn
from transformers import (AutoImageProcessor, ViTForImageClassification,
                          TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,    llm_int8_skip_modules=["classifier"]  
)

model_id = "google/vit-base-patch16-224-in21k"

processor = AutoImageProcessor.from_pretrained(model_id)
processor.image_mean = [0.5, 0.5, 0.5]
processor.image_std = [0.5, 0.5, 0.5]
processor.size = {"height": 224, "width": 224}

model = ViTForImageClassification.from_pretrained(
    model_id,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    quantization_config=bnb_config,
    ignore_mismatched_sizes=True 
)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"]
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

def transforms(examples):
    pixel_values = [
        processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        for img in examples["image"]
    ]
    return {"pixel_values": pixel_values, "labels": examples["label"]}

train_dataset = dataset["train"].with_transform(transforms)
val_dataset = dataset["validation"].with_transform(transforms)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    return {"accuracy": acc, "f1_macro": f1, "recall": recall}

# CUSTOM TRAINER
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = torch.tensor(class_weights, dtype=torch.float32).to(model.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels).float(), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

output_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_Grading_Model'
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    num_train_epochs=10,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    optim="adamw_torch",
    fp16=True,                       
    report_to="none",
    remove_unused_columns=False
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\nFINE-TUNING (10 EPOCHS)...")
trainer.train()

print("\nDone...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Model was saved at: {output_dir}")
