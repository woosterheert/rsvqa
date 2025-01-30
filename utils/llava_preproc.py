import os
from PIL import Image
import uuid
import json
import pandas as pd
import tqdm

def process_and_save(df, data_dir, output_folder, subset_name):
    # Define image subfolder within output folder
    subset_folder = os.path.join(output_folder, subset_name)
    image_subfolder = os.path.join(output_folder, 'images')

    if not os.path.exists(image_subfolder):
        os.makedirs(image_subfolder)

    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)

    # Initialize list to hold all JSON data
    json_data_list = []

    # Process and save images and labels
    for idx, row in tqdm.tqdm(df.iterrows()):

        # Process image
        img_file = os.path.join(data_dir, row.tile_name, row.patch_name)
        tif_image = Image.open(img_file)
        image = tif_image.convert("RGB")
        unique_id = str(uuid.uuid4())
        image_path = os.path.join(image_subfolder, f"{unique_id}.jpg")
        image.save(image_path)

        # Structure for LLaVA JSON
        json_data = {
            "id": unique_id,
            "image": f"{unique_id}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": row.question
                },
                {
                    "from": "gpt",
                    "value": row.answer
                }
            ]
        }

        # Append to list
        json_data_list.append(json_data)

    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, 'dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)


if __name__ == "__main__" :
    df = pd.read_csv('questions_and_answers_binary.csv', index_col=0)
    df_train = df.query("split == 'train'").sample(frac=0.1)
    df_val = df.query("split == 'validation'")[:1000]
    
    process_and_save(df_train, 'rgb_data', 'llava_data_medium', 'train')
    process_and_save(df_val, 'rgb_data', 'llava_data_medium', 'test')