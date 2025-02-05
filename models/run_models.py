import sys
sys.path.insert(1, '/home/wouter/rsvqa')

from transformers import BertModel
from rsvqa.models.dual_encoder import dual_encoder_with_classifier
from torchvision.models import resnet50
import pytorch_lightning as pl
from rsvqa.external.prithvi_mae import PrithviViT
import torch
import yaml
import pandas as pd
from rsvqa.utils.dual_encoder_preproc import RSVQADataProcessor
from rsvqa.utils.paligemma_preproc import create_dataset
from transformers import BertTokenizer


resnet_bert_ckpt = "/home/wouter/rsvqa/lightning_logs/version_11/checkpoints/'epoch=9-step=6250.ckpt'"
prithvi_bert_ckpt = "/home/wouter/rsvqa/lightning_logs/version_12/checkpoints/'epoch=9-step=6250.ckpt'"
resnet_bert_model = load_resnet_bert(resnet_bert_ckpt)
prithvi_bert_model = load_prithvi_bert(prithvi_bert_ckpt)
