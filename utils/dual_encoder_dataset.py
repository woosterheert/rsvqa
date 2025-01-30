from torch.utils.data import Dataset
import torch
import os

class RSVQADataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.file_names = [x for x in os.listdir(self.data_dir) if x.endswith('.pt')]

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



