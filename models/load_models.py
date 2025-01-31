import sys
sys.path.insert(1, '/home/wouter/rsvqa')

from transformers import BertModel
from models.dual_encoder import dual_encoder_with_classifier
from torchvision.models import resnet50
import pytorch_lightning as pl
from external.prithvi_mae import PrithviViT
import torch
import yaml

def load_resnet_bert():
    resnet_encoder = resnet50(pretrained=True)
    bert_encoder = BertModel.from_pretrained('bert-base-uncased')
    model = dual_encoder_with_classifier(resnet_encoder, bert_encoder, 1000, 768, "rgb")

    return model

def load_prithvi_bert():
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

    bert_encoder = BertModel.from_pretrained('bert-base-uncased')
    
    model = dual_encoder_with_classifier(prithvi_encoder, bert_encoder, 768, 768, "6d")

    return model