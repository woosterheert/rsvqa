import sys
sys.path.insert(1, '/home/wouter/rsvqa')

import yaml
import pandas as pd
from transformers import BertTokenizer
from models.load_models import load_prithvi_bert, load_resnet_bert
from utils.dual_encoder_preproc import RSVQADataProcessor
from utils.paligemma_preproc import create_dataset

resnet_bert_ckpt = "/home/wouter/rsvqa/lightning_logs/version_11/checkpoints/'epoch=9-step=6250.ckpt'"
prithvi_bert_ckpt = "/home/wouter/rsvqa/lightning_logs/version_12/checkpoints/'epoch=9-step=6250.ckpt'"
resnet_bert_model = load_resnet_bert(resnet_bert_ckpt)
prithvi_bert_model = load_prithvi_bert(prithvi_bert_ckpt)

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

df_val_balanced.to_csv('/home/wouter/data/app/df.csv')