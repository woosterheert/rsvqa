import sys
sys.path.insert(1, '/home/wouter/rsvqa')

from utils.dual_encoder_dataset import RSVQADataset
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from models.dual_encoder import dual_encoder_with_classifier
import pytorch_lightning as pl
from torchvision.models import resnet50
from external.prithvi_mae import PrithviViT
import torch
import yaml

weights_path = "/home/wouter/data/Prithvi_EO_V1_100M.pt"
model_cfg_path = "/home/wouter/data/config.yaml"
with open(model_cfg_path) as f:
    model_config = yaml.safe_load(f)

model_args = model_config["model_args"]
model_args["num_frames"] = 1
model_args["encoder_only"] = True

checkpoint = torch.load(weights_path, map_location="cpu")
prithvi_encoder = PrithviViT(**model_args)
del checkpoint['encoder.pos_embed']
del checkpoint['decoder.decoder_pos_embed']
_ = prithvi_encoder.load_state_dict(checkpoint, strict=False)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_encoder = BertModel.from_pretrained('bert-base-uncased')

train_ds = RSVQADataset('/home/wouter/data/prithvi_bert_small')
val_ds = RSVQADataset('/home/wouter/data/prithvi_bert_small_validation')

train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

baseline_model_with_prithvi = dual_encoder_with_classifier(prithvi_encoder, bert_encoder, 768, 768, "6d")

trainer = pl.Trainer(max_epochs=1)
trainer.fit(model=baseline_model_with_prithvi, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)