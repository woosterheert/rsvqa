import sys
sys.path.insert(1, '/home/wouter/rsvqa')

import pandas as pd
from models.load_models import load_prithvi_bert, load_resnet_bert
from datasets import load_from_disk
import os
import torch
import tqdm

def load_data(file_path):
    data = torch.load(file_path)
    image = data['image']
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    label = data['label']
    return image, input_ids, attention_mask, label

if __name__ == "__main__" :

    resnet_bert_ckpt = "/home/wouter/rsvqa/lightning_logs/version_11/checkpoints/epoch=9-step=6250.ckpt"
    prithvi_bert_ckpt = "/home/wouter/rsvqa/lightning_logs/version_12/checkpoints/epoch=9-step=6250.ckpt"
    resnet_bert_model = load_resnet_bert(resnet_bert_ckpt)
    resnet_bert_model.eval()
    prithvi_bert_model = load_prithvi_bert(prithvi_bert_ckpt)
    prithvi_bert_model.eval()

    df = pd.read_csv('/home/wouter/data/app/df.csv').rename(columns={'Unnamed: 0': 'idx'})
    paligemma_data = load_from_disk('/home/wouter/data/app/paligemma')

    resnet_predictions = []
    prithvi_predictions = []

    for i in tqdm.tqdm(range(20)):
        resnet_data_path = os.path.join('/home/wouter/data/app/resnet_bert', str(df.iloc[i].idx)+'.pt') 
        image, input_ids, attention_mask, label = load_data(resnet_data_path)

        with torch.no_grad():
            resnet_predictions.append(resnet_bert_model(image, input_ids, attention_mask))

        prithvi_data_path = os.path.join('/home/wouter/data/app/prithvi_bert', str(df.iloc[i].idx)+'.pt') 
        image, input_ids, attention_mask, label = load_data(prithvi_data_path)
        
        with torch.no_grad():
            prithvi_predictions.append(prithvi_bert_model(image, input_ids, attention_mask))
        
    df['resnet_prediction'] = resnet_predictions
    df['prithvi_prediction'] = prithvi_predictions

    df.to_csv('/home/wouter/data/app/df_results.csv')








