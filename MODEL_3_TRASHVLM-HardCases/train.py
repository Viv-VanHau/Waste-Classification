!pip install -q transformers peft accelerate datasets
!pip install -U bitsandbytes>=0.46.1 accelerate peft transformers datasets evaluate

# ============================================
# 1.MULTI-STREAM DATA PIPELINE & CLAHE PREVIEW
# ============================================
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset

class_labels = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
                'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
                'plastic_grade_B', 'textiles', 'trash']

dataset_dir = '/content/drive/MyDrive/Thesis_Project/Outputs_Stage3/Stage6_VLM_Training_Data'
valid_extensions = ('.jpg', '.jpeg', '.png')

image_paths = []
print("Finding images in Drive...")
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(root, file))

if not image_paths:
    raise ValueError(f"Found none, or restart session and run again")

print(f"Found {len(image_paths)} Hard Examples...")

data_dict = {"image_path": [], "label": [], "error_type": []}

for path in image_paths:
    try:
        parts = path.split(os.sep)
        error_type = parts[-2]

        filename = parts[-1]
        true_label_str = filename.split('___')[0].replace('True-', '')
        label_idx = class_labels.index(true_label_str)

        data_dict["image_path"].append(path)
        data_dict["label"].append(label_idx)
        data_dict["error_type"].append(error_type)
    except Exception as e:
        continue

hf_dataset = Dataset.from_dict(data_dict)
split_ds = hf_dataset.train_test_split(test_size=0.2, seed=42)

print(f"Pipeline is ready ehe! Train: {len(split_ds['train'])} images | Val: {len(split_ds['test'])} images")
print("Locating errors in Train Set:")
for err in set(data_dict["error_type"]):
    count = data_dict["error_type"].count(err)
    print(f" {err}: {count} iamges")

# CLAHE
def apply_clahe_to_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return img_rgb, final_img

print("\nTesting...")
metal_files = [p for p in image_paths if 'Specular_Reflection' in p]
if metal_files:
    test_img_path = random.choice(metal_files)
    orig, fixed = apply_clahe_to_image(test_img_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(orig); plt.title("Original (Specular Reflection)", fontweight='bold')
    plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(fixed); plt.title("After CLAHE Fix", fontweight='bold', color='green')
    plt.axis('off')
    plt.show()

# =================
# 2. ADAPTIVE QLORA
# =================
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision import transforms
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from evaluate import load

def apply_clahe(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    return Image.fromarray(cv2.cvtColor(limg, cv2.COLOR_LAB2RGB))

# Weight importance based on risky level
weights_array = [1.0] * 10
weights_array[0] = 20.0  # Battery
weights_array[4] = 15.0  # Organic
weights_array[6] = 12.0  # Plastic_A
weights_array[3] = 10.0  # Metal_B
custom_weights = torch.tensor(weights_array, dtype=torch.float32).to("cuda")

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=custom_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ADAPTIVE TRANSFORMS
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)

_base_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

_heavy_crop_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 0.8)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

_val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

def adaptive_transform_images(examples, is_train=True):
    processed_images = []
    for path, err_type in zip(examples["image_path"], examples["error_type"]):
        if not is_train:
            img = Image.open(path).convert("RGB")
            processed_images.append(_val_transforms(img))
            continue

        if 'Specular_Reflection' in err_type:
            img = apply_clahe(path) or Image.open(path).convert("RGB")
            processed_images.append(_base_transforms(img))
        elif err_type in ['Label_Interference', 'Background_Bias']:
            img = Image.open(path).convert("RGB")
            processed_images.append(_heavy_crop_transforms(img))
        else:
            img = Image.open(path).convert("RGB")
            processed_images.append(_base_transforms(img))
    examples["pixel_values"] = processed_images
    return examples

train_ds = split_ds["train"].with_transform(lambda x: adaptive_transform_images(x, is_train=True))
val_ds = split_ds["test"].with_transform(lambda x: adaptive_transform_images(x, is_train=False))

# Model loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["classifier"] 
)

model = ViTForImageClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    num_labels=10,
    id2label={i: l for i, l in enumerate(class_labels)},
    label2id={l: i for i, l in enumerate(class_labels)},
    ignore_mismatched_sizes=True
)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["query", "value"],
    modules_to_save=["classifier"], 
    lora_dropout=0.2,
    bias="none"
)
lora_model = get_peft_model(model, config)

acc_metric = load("accuracy")
def compute_metrics(p):
    return {"accuracy": acc_metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["accuracy"]}

def ultra_collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

output_model_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_LoRA_Output'

training_args = TrainingArguments(
    output_dir=output_model_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.05,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=70,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    warmup_ratio=0.2,
    logging_steps=5,
    logging_first_step=True,
    report_to="none",
    remove_unused_columns=False
)

trainer = WeightedTrainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=ultra_collate_fn,
)

print("\n70 epoch running...")
trainer.train()
trainer.save_model(output_model_dir)
print(f"Done training yeah, model was saved at: {output_model_dir}")
