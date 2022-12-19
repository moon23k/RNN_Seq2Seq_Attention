import torch, math, time
import torch.nn as nn
import torch.nn.functional as F
from module.search import Search
from datasets import load_metric
from transformers import BertModel, BertTokenizerFast



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.batch_size = config.batch_size        
        self.vocab_size = config.vocab_size
        self.search = Search(config, self.model, tokenizer)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, label_smoothing=0.1).to(self.device)

        if self.task == 'nmt':
            self.metric_name = 'BLEU'
            self.metric_module = load_metric('bleu')

        elif self.task == 'dialog':
            self.metric_name = 'Similarity'
            self.metric_model = BertModel.from_pretrained('bert-base-uncased')
            self.metric_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        elif self.task == 'sum':
            self.metric_name = 'ROUGE'
            self.metric_module = load_metric('rouge')


    def loss_test(self):
        tot_loss = 0.0
        
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                src = batch['src'].to(self.device)
                trg = batch['trg'].to(self.device)

                logit = self.model(src, trg, teacher_forcing_ratio=0.0)
                loss = self.criterion(logit.contiguous().view(-1, self.output_dim), 
                                      trg[:, 1:].contiguous().view(-1)).item()
                tot_loss += loss
            tot_loss /= len(self.dataloader)
        
        print(f'Loss Test Results on {self.task} Task')
        print(f">> Loss: {tot_loss:.3f} | PPL: {math.exp(tot_loss):.2f}\n")


    
    def metric_score(self, pred, label, prev=None):
        
        if self.task == 'nmt':
            pred = [self.tokenizer.EncodeAsPieces(p)[1:-1] for p in pred]
            label = [[self.tokenizer.EncodeAsPieces(l)[1:-1]] for l in label]
            self.metric_moduel.add_batch(predictions=pred, references=label)
            score = self.metric_moduel.compute()['bleu']
        
        elif self.task == 'dialog':
            encoding = self.metric_tokenizer([prev, pred], padding=True, return_tensors='pt')
            bert_out = self.metric_model(**encoding).[0]

            normalized = F.normalize(bert_out[:, 0, :], p=2, dim=-1)  # Only use of [CLS] token embedding
            dist = normalized.matmul(normalized.T)
            sim_matrix = dist.new_ones(dist.shape) - dist
            score = sim_matrix[0, 1].item()
            
            
        elif self.task == 'sum':
            pred_batch = [self.tokenizer.EncodeAsPieces(p)[1:-1] for p in pred]
            label_batch = [[self.tokenizer.EncodeAsPieces(l)[1:-1]] for l in label]
            self.metric_moduel.add_batch(predictions=pred_batch, references=label_batch)
            score = self.metric_moduel.compute()['rouge2'].mid.fmeasure

        return score * 100


    def metric_test(self):
        metric_results = []
        batch = next(iter(self.dataloader))

        input_batch = batch['src'].tolist()
        label_batch = batch['trg'].tolist()

        for input_seq, label_seq in zip(input_batch, label_batch):
            temp_dict = dict()
            input_seq = self.tokenizer.decode(input_seq) 
            label_seq = self.tokenizer.decode(label_seq)

            temp_dict['input_seq'] = input_seq
            temp_dict['label_seq'] = label_seq

            temp_dict['greedy_pred'] = self.search.greedy_search(input_seq)
            temp_dict['beam_pred'] = self.search.beam_search(input_seq)
            
            if self.task != 'dialog':
                metric_ref = (label_seq)
            else:
                metric_ref = (label_seq, input_seq)

            temp_dict['greedy_metric'] = self.metric_score(temp_dict['greedy_pred'], *metric_ref)
            temp_dict['beam_metric'] = self.metric_score(temp_dict['beam_pred'], *metric_ref)
            
            metric_results.append(temp_dict)
        

        metric_results = sorted(metric_results, key=lambda d: d['beam_metric'])
        print_dicts = [metric_results[0]] + \
                      [metric_results[self.batch_size // 2]] + \
                      [metric_results[-1]]


        print(f'Metric Test on {self.task} model')
        for d in print_dicts:
            print(f">> Input Sequence: {d['input_seq']}")
            print(f">> Label Sequence: {d['label_seq']}")
            
            print(f">> Greedy Sequence: {d['greedy_out']}")
            print(f">> Beam   Sequence : {d['beam_out']}")
            
            print(f">> Greedy {self.metric_name.upper()} Score: {d['greedy_metric']:.2f}")
            print(f">> Beam   {self.metric_name.upper()} Score : {d['beam_metric']:.2f}\n")
