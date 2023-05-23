import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import torchtext.vocab as vocab


cache_dir = os.path.join('..','vector_cache', 'glove')
glove = vocab.GloVe(name='6B', dim=100, cache=cache_dir)
word_list = set(glove.stoi.keys())


data_path_train = os.path.join("..", "Project_Dataset", "Selected_Train_Dataset")
data_path_test = os.path.join("..", "Project_Dataset", "Selected_Test_Dataset")
for data_path in [data_path_train, data_path_test]:
    dir_list = os.listdir(data_path)
    for _dir in dir_list:
        pair_path = os.path.join(data_path, _dir)
        prompts_path = os.path.join(pair_path, "prompt.txt")
        with open(prompts_path, 'w+') as f:
            prompt = _dir.strip("_").split()
            new_prompt = []
            for word in prompt:
                if word.lower() in word_list:
                    new_prompt.append(word.lower())
            # print(new_prompt)
            f.write(" ".join(new_prompt))
            
