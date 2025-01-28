import os
import rasterio
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from rasterio.plot import reshape_as_image
from PIL import Image

class RSVQADataset(IterableDataset):
    def __init__(self, df, train_args, data_dir, tokenizer, model_type):
        self.df = df
        self.means = np.array(train_args["data_mean"]).reshape(-1, 1, 1)
        self.stds = np.array(train_args["data_std"]).reshape(-1, 1, 1)   
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.rgb_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __iter__(self):
        self.df.sample(frac=1)
        for idx, row in self.df.iterrows():

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
            
            yield normalized_img, input_ids, attention_mask, label

    def preprocess_6d(self, image):
        normalized = image.copy()
        normalized = np.expand_dims(((image - self.means) / self.stds), axis=1)
        normalized = torch.from_numpy(normalized).to(torch.float32)
        return normalized
    
    def preprocess_rgb(self, image):
        img = reshape_as_image(image.copy())
        pil_img = Image.fromarray(img).convert("RGB")
        normalized = self.rgb_transform(pil_img).unsqueeze(0)
        return normalized
