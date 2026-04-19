!pip install -q -U bitsandbytes>=0.46.1 torchao>=0.16.0 transformers peft datasets accelerate scikit-learn

import os
import zipfile
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (AutoImageProcessor, ViTForImageClassification,
                          TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

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
dataset = DatasetDict({'train': dataset['train'], 'validation': dataset['test']})

labels = dataset["train"].features["label"].names
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}

print("\nWeight calculating...")
train_labels = dataset['train']['label']
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["classifier"]
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
    r=32, lora_alpha=64, target_modules=["query", "value"],
    lora_dropout=0.1, bias="none", modules_to_save=["classifier"]
)
model = get_peft_model(model, config)

def transforms(examples):
    pixel_values = [processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0] for img in examples["image"]]
    return {"pixel_values": pixel_values, "labels": examples["label"]}

train_dataset = dataset["train"].with_transform(transforms)
val_dataset = dataset["validation"].with_transform(transforms)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    _, _, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = torch.tensor(class_weights, dtype=torch.float32).to(model.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels).float(), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

output_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    num_train_epochs=10,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    optim="adamw_torch",
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)

trainer = WeightedTrainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\nFINE-TUNING...")
trainer.train()

print("\nDone ehehehehehehe...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Model was saved at: {output_dir}")
