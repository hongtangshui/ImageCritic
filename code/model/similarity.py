import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

from .img_model import ResNet50
from .img_model import ViT
from .text_model import RNN

class Similarity(nn.Module):
    def __init__(self, device, output_dim=64, text_encoder_type="RNN", img_encoder_type="ResNet50"):
        super(Similarity, self).__init__()
        self.device = device
        self.text_encoder_type = text_encoder_type
        self.img_encoder_type = img_encoder_type
        print("  Loading text encoder ......")
        if text_encoder_type == "RNN":
            self.text_encoder = RNN(output_dim=output_dim, input_size=100)
        else:
            raise ValueError("Unavailable text encoder")
        print("  Loading image encoder ......")
        if img_encoder_type == "ResNet50":
            self.img_encoder = ResNet50(output_dim=output_dim)
        elif img_encoder_type == "ViT":
            self.img_encoder = ViT((3, 224, 224), n_patches=32, n_blocks=2, hidden_d=8, n_heads=2, output_dim=output_dim)
        else:
            raise ValueError("Unavailable image encoder")
        print("  Loading loss function ......")
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, prompts, prompts_length, img1, img2):
        '''
        Input:
            prompts (torch.tensor): [batch_size, max_length, length]: glove embedding of prompts
            prompts_length (list[int]): (batch_size,) : length of prompts
            img1 (troch.tensor): [batch_size, 3, img_size, img_size]
            img2 (torch.tensor): [batch_size, 3, img_size, img_size]
        Output:
            score (torch.tensor): [batch_size, 2], the first columns represent the matching score of the img1.
        '''
        prompts = prompts.permute(1, 0, 2)
        if self.text_encoder_type == "RNN":
            text_vec = self.text_encoder(prompts, prompts_length)       # [b, output_dim]
        else:
            pass
        if self.img_encoder_type == "ResNet50":
            img1_vec = self.img_encoder(img1)                           # [b, output_dim]
            img2_vec = self.img_encoder(img2)
        else:
            pass
        score1 = torch.sum(text_vec * img1_vec, dim=-1, keepdim=True)                 # [b]
        score2 = torch.sum(text_vec * img2_vec, dim=-1, keepdim=True)                 # [b]
        score = torch.concat((score1, score2), dim=-1)
        return score
    
    def calculate_loss(self, prompts, prompts_length, img1, img2, idx):
        '''
        Input:
            *
            idx (torch.tensor): [b,] idx of good img
        Output:
            result (np.array): [b, ]:  Cross Entropy Loss
        '''
        score = self.forward(prompts, prompts_length, img1, img2)
        loss = self.loss_fn(score, idx)
        return loss
    
    def predict(self, prompts, prompts_length, img1, img2):
        score = self.forward(prompts, prompts_length, img1, img2)
        predict = np.array(torch.argmax(score, dim=-1).cpu())
        return predict

if __name__ == "__main__":
    model = Similarity(output_dim=64, text_encoder_type='RNN', img_encoder_type='ResNet50')
    prompts = torch.rand((10, 4, 300))
    prompts_length = [9, 7, 6, 1]   
    img1 = torch.rand((4, 3, 224, 224))
    img2 = torch.rand((4, 3, 224, 224))
    good_idx = torch.tensor([1, 0, 0, 1])
    loss = model.calculat_loss(prompts, prompts_length, img1, img2, good_idx)
    print(loss)
