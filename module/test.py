import torch, math, time, evaluate
from module.search import Search
from transformers import BertModel, BertTokenizerFast



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.search = Search(config, self.model)
        
        if self.task == 'nmt':
            self.metric_name = 'BLEU'
            self.metric_module = evaluate.load('bleu')

        elif self.task == 'dialog':
            self.metric_name = 'Similarity'
            self.metric_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.metric_model = BertModel.from_pretrained('bert-base-uncased')
            self.metric_model.eval()

        elif self.task == 'sum':
            self.metric_name = 'ROUGE'
            self.metric_module = evaluate.load('rouge')



    def test(self):
        with torch.no_grad():
            print(f'Test Results on {config.task.upper()}')
            for idx, batch in enumerate(self.dataloader):
                src = batch['src'].to(self.device)
                trg = batch['trg'].to(self.device)

                greedy_pred = self.search.greedy_search(src)
                beam_pred = self.search.beam_search(src)
                
                greedy_metric_score = self.metric_score(greedy_pred, trg)
                beam_metric_score = self.metric_score(beam_pred, trg)

                if not (idx + 1 % 100):
                    print(f'Total Greedy Test Metric Score: {tot_greedy_metric_score}')
                    print(f'Total  Beam  Test Metric Score: {tot_beam_metric_score}')

        print(f'Total Greedy Test Metric Score: {tot_greedy_metric_score}')
        print(f'Total  Beam  Test Metric Score: {tot_beam_metric_score}')



    def metric_score(self, pred, label, prev=None):
        
        batch_size = label.size(0)
        
        if self.task == 'dialog':
            encoding = self.metric_tokenizer([prev, pred], padding=True, truncation=True, return_tensors='pt')
            bert_out = self.metric_model(**encoding)[0]

            normalized = torch.nn.functional.normalize(bert_out[:, 0, :], p=2, dim=-1)
            dist = normalized.matmul(normalized.T)
            sim_matrix = dist.new_ones(dist.shape) - dist
            score = sim_matrix[0, 1].item()

        else:
            pred_batch = [self.tokenizer.EncodeAsPieces(p)[1:-1] for p in pred]
            label_batch = [[self.tokenizer.EncodeAsPieces(l)[1:-1]] for l in label]
            self.metric_moduel.add_batch(predictions=pred_batch, references=label_batch)
            if self.task == 'nmt':
                score = self.metric_moduel.compute()['bleu']
            elif self.task == 'sum':        
                score = self.metric_moduel.compute()['rouge2'].mid.fmeasure


        return (score * 100) / batch_size
