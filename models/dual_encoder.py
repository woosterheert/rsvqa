import torch.nn as nn
import torch
from torch import optim
import pytorch_lightning as pl

class dual_encoder_with_classifier(pl.LightningModule):
    def __init__(self, vision_encoder, text_encoder, vision_encoder_dim, text_encoder_dim, use_vision_cls_token=False):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_encoder_dim = vision_encoder_dim
        self.text_encoder_dim = text_encoder_dim
        self.use_vision_cls_token = use_vision_cls_token

        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.fusion_layer = nn.Sequential(nn.Linear(self.vision_encoder_dim+self.text_encoder_dim, 128), nn.ReLU())
        self.classification_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, normalised_image, text_tokens, attention_mask):
        img_embedding, _, _ = self.vision_encoder(normalised_image)
        if self.use_vision_cls_token:
            img_embedding = img_embedding[:,0,:]
        txt_embedding = self.text_encoder(input_ids=text_tokens, attention_mask=attention_mask)
        txt_embedding = txt_embedding.last_hidden_state[:,0,:]
        fused_embedding = torch.cat([img_embedding, txt_embedding], dim=1)
        fused_projection = self.fusion_layer(fused_embedding)
        prediction = self.classification_layer(fused_projection)
        return prediction

    def training_step(self, batch, batch_idx):
        images, input_ids, attention_mask, labels = batch
        prediction = self.forward(images, input_ids, attention_mask)
        loss = nn.functional.binary_cross_entropy(prediction, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, input_ids, attention_mask, labels = batch
        prediction = self.forward(images, input_ids, attention_mask)
        loss = nn.functional.binary_cross_entropy(prediction, labels)
        preds = prediction > 0.5
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer