import sys
sys.path.insert(1, '/home/wouter/rsvqa')

from utils.dual_encoder_dataset import RSVQADataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models.load_models import load_resnet_bert

resnet_bert_classifier = load_resnet_bert()

train_ds = RSVQADataset('/home/wouter/data/resnet_bert/training')
val_ds = RSVQADataset('/home/wouter/data/resnet_bert/validation', frac=0.1)
train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

trainer = pl.Trainer(max_epochs=5)
trainer.fit(model=resnet_bert_classifier, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)