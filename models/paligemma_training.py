import sys
sys.path.insert(1, '/home/wouter/rsvqa')

import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, TrainingArguments, Trainer
import pandas as pd
import rasterio
import tqdm
import os
from PIL import Image
from rasterio.plot import reshape_as_image
from datasets import Dataset

def collate_fn(examples):
    texts = ["answer " + example["question"] for example in examples]
    labels= [example['answer'] for example in examples]
    images = [example["image"] for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens

def data_gen(df, data_dir):
    for idx, row in tqdm.tqdm(df.iterrows()):

        path = os.path.join(data_dir, row.tile_name, row.patch_name)
        with rasterio.open(path) as src:
            image = src.read(out_shape=(src.count, 224, 224), resampling=rasterio.enums.Resampling.bilinear)
        
        yield {
            'image': Image.fromarray(reshape_as_image(image)).convert("RGB"),
            'question': row.question,
            'answer': row.answer
        }



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

df = pd.read_csv('/home/wouter/data/questions_and_answers_binary_new.csv', index_col=0)

df_train = df.query("split == 'train'")
df_train_pos = df_train.query('binary_answer==1').sample(100)  
df_train_neg = df_train.query('binary_answer==0').sample(100)
df_train_balanced = pd.concat([df_train_pos, df_train_neg]).sample(frac=1)

df_val = df.query("split == 'validation'")
df_val_pos = df_val.query('binary_answer==1').sample(100)  
df_val_neg = df_val.query('binary_answer==0').sample(100)
df_val_balanced = pd.concat([df_val_pos, df_val_neg]).sample(frac=1)

train_ds = Dataset.from_generator(data_gen, gen_kwargs={"df": df_train_balanced, "data_dir": '/home/wouter/data/rgb_data'})
val_ds = Dataset.from_generator(data_gen, gen_kwargs={"df": df_val_balanced, "data_dir": '/home/wouter/data/rgb_data'})

trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        args=args
        )
trainer.train()