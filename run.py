import numpy as np
import sentencepiece as spm
import os, yaml, random, argparse, nltk

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from module.model import load_model
from module.data import load_dataloader

from module.train import Trainer
from module.test import Tester
from module.search import Search




class Config(object):
    def __init__(self, args):    

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.task = args.task
        self.mode = args.mode
        self.ckpt = f"ckpt/{self.task}.pt"

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'
        self.device = torch.device(self.device_type)

        if self.task != 'train':
            if self.task == 'nmt':
                self.max_pred_len = self.nmt_max_pred_len
            elif self.task == 'dialog':
                self.max_pred_len = self.dialog_max_pred_len
            elif self.task == 'sum':
                self.max_pred_len = self.sum_max_pred_len
                self.batch_size = self.batch_size // 8

        if self.task == 'inference':
            self.search_method = args.search
            self.device = torch.device('cpu')

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")


def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



def load_tokenizer(task):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/{task}/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')
    return tokenizer



def inference(config, model, tokenizer):
    if config.search_method == 'beam':
        beam = Search(config, model)
    search_module = Search(config, model)

    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #Enc Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has Terminated! ---')
            break        

        if config.task == 'sum':
            input_seq = nltk.tokenize.sent_tokenize(input_seq)

        input_tensor = torch.LongTensor(tokenizer.encode(input_seq)).unsqueeze(0)


        if config.search_method == 'greedy':
            pred_tensor = search_module.greedy_search(input_tensor)
        elif config.search_method == 'beam':
            pred_tensor = search_module.beam_search(input_tensor)

        pred_seq = tokenizer.decode(pred_tensor)
        print(f"Model Out Sequence >> {tokenizer.decode(pred_seq)}")



def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)

    if config.mode == 'train':
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        tokenizer = load_tokenizer(args.task)
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, test_dataloader, tokenizer)
        tester.test()
    
    elif config.mode == 'inference':
        tokenizer = load_tokenizer(args.task)
        inference(model, tokenizer)
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.task in ['nmt', 'dialog', 'sum']
    assert args.mode in ['train', 'test', 'inference']

    if args.task == 'inference':
        assert args.search in ['greedy', 'beam']
        if args.task == 'sum':
            nltk.download('punkt')

    main(args)