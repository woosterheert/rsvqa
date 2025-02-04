import os
from PIL import Image
import rasterio
from rasterio.plot import reshape_as_image
from datasets import Dataset
import pandas as pd
import tqdm

def data_gen(df, data_dir):
    for idx, row in tqdm.tqdm(df.iterrows()):

        path = os.path.join(data_dir, row.tile_name, row.patch_name)
        with rasterio.open(path) as src:
            image = src.read(out_shape=(src.count, 224, 224), resampling=rasterio.enums.Resampling.bilinear)
        
        yield {
            'image': Image.fromarray(image).convert("RGB"),
            'question': row.question,
            'answer': row.answer
        }

df = pd.read_csv('/home/wouter/data/questions_and_answers_binary_new.csv', index_col=0)

df_train = df.query("split == 'train'")
df_train_pos = df_train.query('binary_answer==1').sample(5000)  
df_train_neg = df_train.query('binary_answer==0').sample(5000)
df_train_balanced = pd.concat([df_train_pos, df_train_neg]).sample(frac=1)

df_val = df.query("split == 'validation'")
df_val_pos = df_val.query('binary_answer==1').sample(100)  
df_val_neg = df_val.query('binary_answer==0').sample(100)
df_val_balanced = pd.concat([df_val_pos, df_val_neg]).sample(frac=1)

train_ds = Dataset.from_generator(data_gen, gen_kwargs={"df": df_train_balanced, "data_dir": '/home/wouter/data/rgb_data'})
val_ds = Dataset.from_generator(data_gen, gen_kwargs={"df": df_val_balanced, "data_dir": '/home/wouter/data/rgb_data'})

train_ds.save_to_disk('/home/wouter/data/paligemma_train')
val_ds.save_to_disk('/home/wouter/data/paligemma_val')
