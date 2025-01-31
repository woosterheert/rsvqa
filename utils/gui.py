import sys
sys.path.insert(1, '/Users/wouter/Documents/study/OU/capitaselecta/rsvqa')

import streamlit as st
import os
import numpy as np
import rasterio
from external.generate_labels import ask_questions
import pandas as pd
from models.dual_encoder import dual_encoder_with_classifier
from transformers import BertModel, BertTokenizer
from models.dual_encoder import dual_encoder_with_classifier
from torchvision.models import resnet50
from rasterio.plot import reshape_as_image
from PIL import Image
from torchvision import transforms

img_folder_rgb = '/Users/wouter/Documents/study/OU/capitaselecta/rsvqa_dev/rgb_data/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP'
img_folder_6d = '/Users/wouter/Documents/study/OU/capitaselecta/rsvqa_dev/6d_data/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP'
metadata = pd.read_parquet('/Users/wouter/Documents/study/OU/capitaselecta/rsvqa_dev/metadata.parquet')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
resnet_encoder = resnet50(pretrained=True)
bert_encoder = BertModel.from_pretrained('bert-base-uncased')
# resnet_bert_model = dual_encoder_with_classifier(resnet_encoder, bert_encoder, 1000, 768, "rgb")
resnet_bert_model = dual_encoder_with_classifier.load_from_checkpoint("/Users/wouter/Documents/study/OU/capitaselecta/rsvqa_dev/resnet_bert.ckpt",
                                                                      vision_encoder=resnet_encoder, 
                                                                      text_encoder=bert_encoder, 
                                                                      vision_encoder_dim=1000, 
                                                                      text_encoder_dim=768, 
                                                                      model_type="rgb")
resnet_bert_model.eval()

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

def enhance_raster_for_visualization(raster, ref_img=None):
    if ref_img is None:
        ref_img = raster
    channels = []
    for channel in range(raster.shape[0]):
        valid_mask = np.ones_like(ref_img[channel], dtype=bool)
        valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
        mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
        normalized_raster = (raster[channel] - mins) / (maxs - mins)
        normalized_raster[~valid_mask] = 0
        clipped = np.clip(normalized_raster, 0, 1)
        channels.append(clipped)
    clipped = np.stack(channels)
    channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
    rgb = channels_last[..., ::-1]
    return rgb

def prep_data(path, question):
    with rasterio.open(path) as src:
        img = src.read(out_shape=(src.count, 224, 224), resampling=rasterio.enums.Resampling.bilinear)
    
    normalized_img = preprocess_rgb(img).unsqueeze(0)
    
    tokens = tokenizer(question, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    return normalized_img, input_ids, attention_mask

def preprocess_rgb(image):
    rgb_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = reshape_as_image(image.copy())
    pil_img = Image.fromarray(img).convert("RGB")
    normalized = rgb_transform(pil_img)
    return normalized

# This is where the GUI starts
# st.set_page_config(layout="wide")
st.title('Remote Sensing Visual Question Answering')

col1, col2 = st.columns(2, gap='large')

with col1:
    st.header("Image")
    img_names = os.listdir(img_folder_rgb)[:10]
    img_name = st.selectbox("Pick an image from the test set",
    tuple(img_names))
    path = os.path.join(img_folder_rgb, img_name)
    patch_id = 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_' + img_name[:-4]
    df_label = metadata.query(f'patch_id == "{patch_id}"')
    labels = df_label.iloc[0].labels.tolist()

    with rasterio.open(path) as src:
        img = src.read()
    nice_img = enhance_raster_for_visualization(img)
    st.image(nice_img, use_container_width=True)
    
with col2:
    st.header('Question & Answer')
    st.text(f'land covers prsent in this image: {", ".join(labels)}')

    if st.button("Generate a new question"):
        question = ask_questions(labels, number=1, presence_only=True)

    question = ask_questions(labels, number=1, presence_only=True)
    st.markdown('**question**')
    st.text(question[0][0])
    st.markdown('**answer**')
    st.text(question[1][0])

    if st.button("Retrieve answer from model"):
        normalized_img, input_ids, attention_mask = prep_data(path, question[0][0])
        pred = resnet_bert_model(normalized_img, input_ids, attention_mask)
        st.text(pred)
