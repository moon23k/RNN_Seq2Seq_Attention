import numpy as np
import os, yaml, random, argparse

import torch
import torch.backends.cudnn as cudnn

from module.test import Tester
from module.train import Trainer
from module.search import Search
from module.model import load_model
from module.data import load_dataloader

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



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

        if self.task == 'sum':
            self.batch_size = self.batch_size // 4

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'
        
        if self.task == 'inference':
            self.search_method = args.search
            self.device = torch.device('cpu')
        else:
            self.search = None
            self.device = torch.device(self.device_type)


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_tokenizer(config):
    tokenizer_path = f"data/{config.task}/tokenizer.json"
    assert os.path.exists(tokenizer_path)

    tokenizer = Tokenizer.from_file(tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
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
    tokenizer = load_tokenizer(config)

    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
    
    elif config.mode == 'inference':
        inference(model, tokenizer)
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-attention', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.task in ['nmt', 'dialog', 'sum']
    assert args.attention in ['additive', 'dot_product', 'scaled_dot_product']
    assert args.mode in ['train', 'test', 'inference']

    if args.task == 'inference':
        assert args.search in ['greedy', 'beam']

    main(args)