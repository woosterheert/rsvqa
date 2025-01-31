import pandas as pd
import random
from external.generate_labels import ask_questions
import os
import tqdm

class QuestionGenerator:
    def __init__(self, dir_data_in, presence_only=False, nr_questions_per_img=10):
        self.dir_data_in = dir_data_in 
        self.metadata = pd.read_parquet('metadata.parquet')
        self.tile_names = [tile for tile in os.listdir(self.dir_data_in) 
                           if os.path.isdir(os.path.join(self.dir_data_in, tile))]
        self.data = []
        self.df_data = None
        self.presence_only = presence_only
        self.nr_questions = nr_questions_per_img
    
    def find_labels(self, patch_id):
        df_label = self.metadata.query(f'patch_id == "{patch_id}"')
        if len(df_label) == 0:
            # print(f'could not process patch {patch_id}, skipping')
            return None, None
        else:
            return df_label.iloc[0].labels.tolist(), df_label.iloc[0].split  
    
    def process_tile(self, tile_name):
        # print(f'processing {tile_name}')
        patch_names = [patch for patch in os.listdir(os.path.join(self.dir_data_in, tile_name)) 
                       if patch.endswith('.tif')]
        for patch_name in patch_names:
            # print(f'processing {patch_name}')
            patch_id = tile_name + '_' + patch_name[:-4]
            labels, split = self.find_labels(patch_id)
            if labels is not None:
                questions, answers, _ = ask_questions(labels, number=self.nr_questions, presence_only=self.presence_only)
                for i in range(self.nr_questions):
                    self.data.append([tile_name, patch_name, patch_id, questions[i], answers[i], split]) 

    def process_tiles(self):
        random.seed(42)
        for tile in tqdm.tqdm(self.tile_names):
            self.process_tile(tile)
        self.df_data = pd.DataFrame(self.data, columns=['tile_name', 'patch_name', 'patch_id', 
                                                        'question', 'answer', 'split'])

    def to_file(self, file):
        self.df_data.to_csv(file)

if __name__ == "__main__" :
    qgen_bin = QuestionGenerator('/home/wouter/data/rgb_data', presence_only=True, nr_questions_per_img=1)
    qgen_bin.process_tiles()
    qgen_bin.df_data['binary_answer'] = qgen_bin.df_data.answer.apply(lambda x: 0 if x=='no' else 1)
    qgen_bin.to_file('/home/wouter/data/questions_and_answers_binary_new.csv') 