import os
import rasterio
import numpy as np
import torch
from torch.utils.data import IterableDataset

class RSVQADataset(IterableDataset):
    def __init__(self, df, train_args, data_dir, tokenizer, add_temporal_dimension):
        self.df = df
        self.means = np.array(train_args["data_mean"]).reshape(-1, 1, 1)
        self.stds = np.array(train_args["data_std"]).reshape(-1, 1, 1)   
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.add_temporal_dimension = add_temporal_dimension

    def __iter__(self):
        self.df.sample(frac=1)
        for idx, row in self.df.iterrows():

            path = os.path.join(self.data_dir, row.tile_name, row.patch_name)
            with rasterio.open(path) as src:
                img = src.read(out_shape=(src.count, 224, 224), resampling=rasterio.enums.Resampling.bilinear)
            
            normalized_img = self.preprocess_image(img)
            
            tokens = self.tokenizer(row.question, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)

            label = torch.tensor(np.expand_dims(np.array(row.binary_answer), axis=0), dtype=torch.float32)
            
            yield normalized_img, input_ids, attention_mask, label

    def preprocess_image(self, image):
        normalized = image.copy()
        if self.add_temporal_dimension:
            normalized = np.expand_dims(((image - self.means) / self.stds), axis=1)
        else:
            normalized = (image - self.means) / self.stds
        normalized = torch.from_numpy(normalized).to(torch.float32)
        return normalized