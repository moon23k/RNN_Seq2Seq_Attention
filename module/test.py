import torch, math, time, evaluate




class Tester:
    def __init__(self, config, model, generator, test_dataloader, test_volumn=100):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.device = config.device
        self.test_volumn = test_volumn
        self.model_type = config.model_type

        self.generator = generator
        self.tokenizer = self.generator.tokenizer
        self.dataloader = test_dataloader
        
        self.metric_name = 'BLEU' if self.task == 'nmt' else 'ROUGE'
        self.metric_module = evaluate.load(self.metric_name.lower())
        


    def test(self):
        self.model.eval()
        volumn_cnt = 0
        greedy_score, beam_score = 0.0, 0.0

        with torch.no_grad():
            for batch in self.dataloader:
                
                #End Condifion
                volumn_cnt += 1
                if volumn_cnt == self.test_volumn:
                    break

                src = batch['src'].to(self.device)
                label = self.tokenizer.decode(batch['trg'].tolist()[0])            
        
                greedy_pred = self.generator.generate(src, search='greedy')
                beam_pred = self.generator.generate(src, search='beam')
                
                greedy_score += self.metric_score(greedy_pred, label)
                beam_score += self.metric_score(beam_pred, label)
        
        greedy_score = round(greedy_score/self.test_volumn, 2)
        beam_score = round(beam_score/self.test_volumn, 2)
        
        self.print_rst(greedy_score, beam_score)



    def print_rst(self, greedy_score, beam_score):
        txt = f"TEST Result on {self.task.upper()} with {self.model_type.upper()} model"
        txt += f"\n--Greedy Score: {greedy_score}"
        txt += f"\n--Beam   Score: {beam_score}" 
        print(txt)



    def metric_score(self, pred, label):
        if not pred:
            return 0.00

        #For Translation and Summarization Tasks
        if self.task == 'nmt':
            score = self.metric_module.compute(
                predictions=[pred], 
                references =[[label]]
            )['bleu']

        else:
            score = self.metric_module.compute(
                predictions=[pred], 
                references =[[label]]
            )['rouge2']

        return score * 100