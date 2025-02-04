import os
import rasterio
import numpy as np
import torch
from torchvision import transforms
from rasterio.plot import reshape_as_image
from PIL import Image
from transformers import BertTokenizer
import pandas as pd
import yaml
import tqdm
import uuid


class RSVQADataProcessor:
    def __init__(self, df, train_args, data_dir, dir_out, tokenizer, model_type):
        self.df = df
        self.means = np.array(train_args["data_mean"]).reshape(-1, 1, 1)
        self.stds = np.array(train_args["data_std"]).reshape(-1, 1, 1)   
        self.data_dir = data_dir
        self.dir_out = dir_out
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.rgb_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def process(self):
        print('start processing')
        for idx, row in tqdm.tqdm(self.df.iterrows()):

            path = os.path.join(self.data_dir, row.tile_name, row.patch_name)
            with rasterio.open(path) as src:
                img = src.read(out_shape=(src.count, 224, 224), resampling=rasterio.enums.Resampling.bilinear)
            
            if self.model_type == '6d':
                normalized_img = self.preprocess_6d(img)
            elif self.model_type == 'rgb':
                normalized_img = self.preprocess_rgb(img)
            
            tokens = self.tokenizer(row.question, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
            label = torch.tensor(np.expand_dims(np.array(row.binary_answer), axis=0), dtype=torch.float32)

            unique_id = str(uuid.uuid4())
            save_path = os.path.join(self.dir_out, f"{unique_id}.pt")
            torch.save({"image": normalized_img, 
                        "input_ids": input_ids, 
                        "attention_mask": attention_mask,
                        "label": label}, save_path)


    def preprocess_6d(self, image):
        normalized = image.copy()
        normalized = np.expand_dims(((image - self.means) / self.stds), axis=1)
        normalized = torch.from_numpy(normalized).to(torch.float32)
        return normalized
    
    def preprocess_rgb(self, image):
        img = reshape_as_image(image.copy())
        pil_img = Image.fromarray(img).convert("RGB")
        normalized = self.rgb_transform(pil_img)
        return normalized


if __name__ == "__main__" :
    
    df = pd.read_csv('/home/wouter/data/questions_and_answers_binary_new.csv', index_col=0)
    
    df_train = df.query("split == 'train'")
    df_train_pos = df_train.query('binary_answer==1').sample(25000)  
    df_train_neg = df_train.query('binary_answer==0').sample(25000)
    df_train_balanced = pd.concat([df_train_pos, df_train_neg]).sample(frac=1)

    df_val = df.query("split == 'validation'")
    df_val_pos = df_val.query('binary_answer==1').sample(500)  
    df_val_neg = df_val.query('binary_answer==0').sample(500)
    df_val_balanced = pd.concat([df_val_pos, df_val_neg]).sample(frac=1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open("/home/wouter/data/config.yaml") as f:
        model_config = yaml.safe_load(f)
    train_args = model_config["train_params"]

    train_proc = RSVQADataProcessor(df_train_balanced, train_args, '/home/wouter/data/rgb_data', '/home/wouter/data/resnet_bert/training_large', tokenizer, "rgb")
    train_proc.process()
    train_proc = RSVQADataProcessor(df_train_balanced, train_args, '/home/wouter/data/6d_data', '/home/wouter/data/prithvi_bert/training_large', tokenizer, "6d")
    train_proc.process()
    train_proc = RSVQADataProcessor(df_val_balanced, train_args, '/home/wouter/data/rgb_data', '/home/wouter/data/resnet_bert/validation_large', tokenizer, "rgb")
    train_proc.process()
    train_proc = RSVQADataProcessor(df_val_balanced, train_args, '/home/wouter/data/6d_data', '/home/wouter/data/prithvi_bert/validation_large', tokenizer, "6d")
    train_proc.process()