from ..utils.dual_encoder_dataset import RSVQADataset
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from ..models.dual_encoder import dual_encoder_with_classifier
import pytorch_lightning as pl
from torchvision.models import resnet50


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
resnet_encoder = resnet50(pretrained=True)
bert_encoder = BertModel.from_pretrained('bert-base-uncased')

train_ds = RSVQADataset('/home/wouter/data/resnet_bert_small')
val_ds = RSVQADataset('/home/wouter/data/resnet_bert_small_validation')

train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

baseline_model = dual_encoder_with_classifier(resnet_encoder, bert_encoder, 1000, 768, "rgb")

trainer = pl.Trainer(max_epochs=1)
trainer.fit(model=baseline_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)