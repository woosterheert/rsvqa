from torch.utils.data import Dataset
import torch
import os
import random

class RSVQADataset(Dataset):
    def __init__(self, data_dir, frac=1.0):
        super().__init__()
        self.data_dir = data_dir
        self.file_names = [x for x in os.listdir(self.data_dir) if x.endswith('.pt')]
        self.frac = frac
        if self.frac < 1:
            nr = int(self.frac * len(self.file_names))
            self.file_names = random.sample(self.file_names, nr)

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        file_path = os.path.join(self.data_dir, self.file_names[index])
        data = torch.load(file_path)
        image = data['image']
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        label = data['label']

        return image, input_ids, attention_mask, label



