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

resnet_encoder = resnet50(pretrained=True)
bert_encoder = BertModel.from_pretrained('bert-base-uncased')

resnet_bert_model = dual_encoder_with_classifier.load_from_checkpoint(resnet_bert_ckpt,
                                                                      vision_encoder=resnet_encoder, 
                                                                      text_encoder=bert_encoder, 
                                                                      vision_encoder_dim=1000, 
                                                                      text_encoder_dim=768, 
                                                                      model_type='rgb')

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

prithvi_bert_model = dual_encoder_with_classifier.load_from_checkpoint(prithvi_bert_ckpt,
                                                                       vision_encoder=prithvi_encoder, 
                                                                       text_encoder=bert_encoder, 
                                                                       vision_encoder_dim=768, 
                                                                       text_encoder_dim=768, 
                                                                       model_type='6d')

df = pd.read_csv('/home/wouter/data/questions_and_answers_binary_new.csv', index_col=0)
df_val = df.query("split == 'validation'")
df_val_pos = df_val.query('binary_answer==1').sample(10)  
df_val_neg = df_val.query('binary_answer==0').sample(10)
df_val_balanced = pd.concat([df_val_pos, df_val_neg]).sample(frac=1)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
with open("/home/wouter/data/config.yaml") as f:
    model_config = yaml.safe_load(f)
train_args = model_config["train_params"]

resnet_proc = RSVQADataProcessor(df_val_balanced, train_args, '/home/wouter/data/rgb_data', '/home/wouter/data/app/resnet_bert', tokenizer, "rgb")
resnet_proc.process()
prithvi_proc = RSVQADataProcessor(df_val_balanced, train_args, '/home/wouter/data/6d_data', '/home/wouter/data/app/prithvi_bert', tokenizer, "6d")
prithvi_proc.process()

paligemma_ds = create_dataset(df_val_balanced, '/home/wouter/data/rgb_data')
paligemma_ds.save_to_disk('/home/wouter/data/app/paligemma')
