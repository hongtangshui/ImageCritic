import numpy as np
import os 
import pandas as pd
import argparse
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from time import gmtime, strftime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from model.similarity import Similarity
from dataset.prompt_dataset import PromptDataset, PromptDataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument('--text_encoder_type', type=str, default='RNN')
    parser.add_argument('--img_encoder_type', type=str, default='ResNet50')
    parser.add_argument('--prompts_max_len', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=224)

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs("./result", exist_ok=True)
    # set_environment(seed=4307)
    log_file = os.path.join("result", "{}-{}-{}".format(
        args.text_encoder_type, args.img_encoder_type, strftime('%Y%m%d%H%M%S', gmtime())))
    def printzzz(log):
        with open(log_file, 'a') as f:
            f.write(log+'\n')
        print(log)
    print("Loading device ......")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading model ......")
    model = Similarity(device=device, output_dim=args.output_dim, 
        text_encoder_type=args.text_encoder_type, img_encoder_type=args.img_encoder_type)
    model = model.to(device)
    model.zero_grad()

    print("Loading data ......")
    train_set = PromptDataset(prompts_max_len=args.prompts_max_len, mode="train")
    test_set = PromptDataset(prompts_max_len=args.prompts_max_len, mode="test")
    valid_set = PromptDataset(prompts_max_len=args.prompts_max_len, mode="valid")
    train_loader = PromptDataLoader(train_set, args.batch_size, shuffle=True)
    test_loader = PromptDataLoader(test_set, args.batch_size, shuffle=False)
    valid_loader = PromptDataLoader(valid_set, args.batch_size, shuffle=False)

    m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    m_optim.zero_grad()
    m_scheduler = get_linear_schedule_with_warmup(
        m_optim, num_warmup_steps=len(train_set)//args.batch_size*2,
        num_training_steps=len(train_set)*args.epoch//args.batch_size,
    )

    for epoch in range(args.epoch):
        avg_loss = 0.0
        batch_iterator = tqdm(train_loader, disable=False)
        for step, train_batch in enumerate(batch_iterator):
            loss = model.calculate_loss(
                train_batch["prompts"].to(device),
                train_batch["prompts_length"].to(device).cpu(),
                train_batch["image1"].to(device),
                train_batch["image2"].to(device),
                train_batch["idx"].to(device),
            )
            loss.backward()
            m_optim.step()
            m_scheduler.step()
            m_optim.zero_grad()
            avg_loss += loss.item()
        # evaluating on the training set
        print("Training on training set ... Epoch: {}, loss:".format(epoch), avg_loss)
        predicts_valid = []
        batch_iterator = tqdm(valid_loader, disable=False)
        for step, valid_batch in enumerate(batch_iterator):
            predict = model.predict(
                valid_batch["prompts"].to(device),
                valid_batch["prompts_length"].to(device).cpu(),
                valid_batch["image1"].to(device),
                valid_batch["image2"].to(device),
            )
            idx = np.array(valid_batch["idx"])
            result = np.array(idx == predict, dtype=np.int32)
            predicts_valid += list(result)
        print("Evaluating on validation set.... Epoch: {}, Precision:".format(epoch), np.sum(predicts_valid)/len(predicts_valid))
        predicts = []
        batch_iterator = tqdm(test_loader, disable=False)
        for step, train_batch in enumerate(batch_iterator):
            predict = model.predict(
                train_batch["prompts"].to(device),
                train_batch["prompts_length"].to(device).cpu(),
                train_batch["image1"].to(device),
                train_batch["image2"].to(device),
            )
            predicts += list(predict)
        print("Predicting on test set.... Epoch: {}, Precision:".format(epoch), 1 - np.sum(predicts)/len(predicts))
        print("           ")
        print("           ")
    pass

if __name__ == "__main__":
    main()
