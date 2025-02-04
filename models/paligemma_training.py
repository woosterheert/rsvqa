import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, TrainingArguments, Trainer
from datasets import load_from_disk

def collate_fn(examples):
    texts = ["answer " + example["question"] for example in examples]
    labels= [example['answer'] for example in examples]
    images = [example["image"] for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224", torch_dtype=torch.bfloat16).to(device)
processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")
args=TrainingArguments(
            num_train_epochs=1,
            remove_unused_columns=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            push_to_hub=True,
            save_total_limit=1,
            bf16=True,
            dataloader_pin_memory=False,
            output_dir="output2"
        )

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True

train_ds = load_from_disk('/home/wouter/data/paligemma_train')
val_ds = load_from_disk('/home/wouter/data/paligemma_val')

trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        args=args
        )
trainer.train()