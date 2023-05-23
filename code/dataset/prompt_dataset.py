import numpy as np 
import pandas as pd
import os
from PIL import Image
from sklearn.utils import shuffle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import torchtext.vocab as vocab
from transformers import AutoTokenizer
import torch
import cv2
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader 
import random


stop_words = set(stopwords.words('english'))
word_tokenizer = RegexpTokenizer(r'\w+')
cache_dir = os.path.join('..','vector_cache', 'glove')
glove = vocab.GloVe(name='6B', dim=100, cache=cache_dir)
word_list = set(glove.stoi.keys())

def remove_stopwords(sentence):
    return [word for word in word_tokenizer.tokenize(sentence.lower()) if word not in stop_words]

def glove_embedding(prompt):
    return np.array([np.array(glove.vectors[glove.stoi[word],:]) for word in prompt])


class PromptDataset(Dataset):
    def __init__(self, prompts_max_len=10, mode="train", valid_proportion=0.1):
        super(PromptDataset, self).__init__()
        self.prompts_max_len = prompts_max_len
        self.mode = mode
        self.img_size = 224
        self.train_path = os.path.join("..", "Project_Dataset", "Selected_Train_Dataset")
        self.test_path = os.path.join("..", "Project_Dataset", "Selected_Test_Dataset")
        if self.mode == "train":
            self.data = self.get_train_data(mode)
            # self.data = shuffle(self.data)
            length = len(self.data)
            self.data = self.data[:int(length * (1 - valid_proportion))]
        elif self.mode == "valid":
            self.data = self.get_train_data(mode)
            # self.data = shuffle(self.data)
            length = len(self.data)
            self.data = self.data[int(length * (1 - valid_proportion)):]
        else:
            self.data = self.get_test_data(mode)

    def get_train_data(self, mode):
        data_path = self.train_path
        df = pd.DataFrame(columns=["prompt", 'prompt_length',"image1", "image2", "idx"])
        dir_list = os.listdir(data_path)
        i = 0
        for _dir in dir_list:
            pair_path = os.path.join(data_path, _dir)
            prompts_path = os.path.join(pair_path, "prompt.txt")
            with open(prompts_path, 'r') as f:
                prompt = f.readline()
                prompt = remove_stopwords(prompt)[:self.prompts_max_len]
                prompt_length = len(prompt)
                prompt += ["anything"]*(self.prompts_max_len - prompt_length)
                prompt_length = 1 if prompt_length == 0 else prompt_length
                prompt = glove_embedding(prompt)
            good_dir_path = os.path.join(pair_path, "good")
            bad_dir_path = os.path.join(pair_path, "bad")
            for img_name in os.listdir(good_dir_path):
                good_path = os.path.join(good_dir_path, img_name)
                bad_path = os.path.join(bad_dir_path, img_name)
                good_img = np.array(Image.open(good_path))
                bad_img = np.array(Image.open(bad_path))
                good_img = cv2.resize(good_img, dsize=(self.img_size, self.img_size))
                bad_img = cv2.resize(bad_img, dsize=(self.img_size, self.img_size))
                if random.random() < 0.5:
                    data_piece = {"prompt": prompt, "prompt_length": prompt_length, "image1": good_img, "image2": bad_img, 'idx':0}
                else:
                    data_piece = {"prompt": prompt, "prompt_length": prompt_length, "image1": bad_img, "image2": good_img, 'idx':1}
                df.loc[len(df)] = data_piece
            # i+=1
            # if i > 20:
            #     break
        # TODO: data agumentation
        return df   

    def get_test_data(self, mode):
        data_path = self.test_path
        df = pd.DataFrame(columns=["prompt", "prompt_length","image1", "image2"])
        dir_list = os.listdir(data_path)
        i = 0
        for _dir in dir_list:
            pair_path = os.path.join(data_path, _dir)
            prompts_path = os.path.join(pair_path, "prompt.txt")
            with open(prompts_path, 'r') as f:
                prompt = f.readline()
                prompt = remove_stopwords(prompt)[:self.prompts_max_len]
                prompt_length = len(prompt)
                prompt_length = 1 if len(prompt) == 0 else prompt_length
                prompt += ["anything"]*(self.prompts_max_len - prompt_length)
                prompt = glove_embedding(prompt)
            img1_dir_path = os.path.join(pair_path, "image1")
            img2_dir_path = os.path.join(pair_path, "image2")
            for img_name in os.listdir(img1_dir_path):
                img1_path = os.path.join(img1_dir_path, img_name)
                img2_path = os.path.join(img2_dir_path, img_name)
                img1 = np.array(Image.open(img1_path))
                img2 = np.array(Image.open(img2_path))
                img1 = cv2.resize(img1, dsize=(self.img_size, self.img_size))
                img2 = cv2.resize(img2, dsize=(self.img_size, self.img_size))
                data_piece = {"prompt": prompt, "prompt_length": prompt_length, "image1": img1, "image2": img2}
                df.loc[len(df)] = data_piece
            # i+=1
            # if i > 20:
            #     break
        return df 

    def __getitem__(self, index:int):
        data_piece = self.data.iloc[index]
        prompt = data_piece["prompt"]
        prompt_length = data_piece["prompt_length"]
        image1 = data_piece["image1"]
        image2 = data_piece["image2"]
        if self.mode != "test":
            idx = data_piece["idx"]
            inputs = {"prompt": prompt, "prompt_length": prompt_length, "image1": image1, "image2": image2, "idx": idx}
        else:
            inputs = {"prompt": prompt, "prompt_length": prompt_length, "image1": image1, "image2": image2}
        return inputs
    
    def collate(self, batch):
        prompts = torch.tensor(np.array([item["prompt"] for item in batch]))
        prompts_length = torch.tensor(np.array([item["prompt_length"] for item in batch]))
        image1 = torch.tensor(np.array([item["image1"] for item in batch])).permute(0, 3, 1, 2).float()
        image2 = torch.tensor(np.array([item["image2"] for item in batch])).permute(0, 3, 1, 2).float()
        if self.mode != "test":
            idx = torch.tensor(np.array([item["idx"] for item in batch]))
            inputs = {"prompts": prompts, "prompts_length": prompts_length, "image1": image1, "image2": image2, "idx": idx}
        else:
            inputs = {"prompts": prompts, "prompts_length": prompts_length, "image1": image1, "image2": image2}
        return inputs

    def __len__(self,):
        return len(self.data)
    

class PromptDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False, sampler=None)->None:
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, sampler=sampler, collate_fn=dataset.collate)

