import torch, operator
import torch.nn.functional as F
from itertools import groupby
from queue import PriorityQueue
from collections import namedtuple



class Search:
    def __init__(self, config, model):
        super(Search, self).__init__()
        
        self.beam_size = 4
        self.model = model
        self.task = config.task
        self.device = config.device
        
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id

        self.max_len = config.max_pred_len
        self.Node = namedtuple('Node', ['prev_node', 'pred', 'log_prob', 'hiddens', 'length'])


    def get_score(self, node, max_repeat=5, min_length=5, alpha=1.2): 
        if not node.log_prob:
            return node.log_prob

        #find max number of consecutively repeated tokens
        repeat = max([sum(1 for token in group if token != self.pad_id) for _, group in groupby(node.pred.tolist())])

        repeat_penalty = 0.5 if repeat > max_repeat else 1
        len_penalty = ((node.length + min_length) / (1 + min_length)) ** alpha
        
        score = node.log_prob / len_penalty
        score = score * repeat_penalty
        return score


    def get_nodes(self, hiddens):
        Node = self.Node
        nodes = PriorityQueue()
        bos_tokens = torch.LongTensor(1, 1).fill_(self.bos_id).to(self.device)

        start_node = Node(prev_node = None, 
                          pred = bos_tokens, 
                          log_prob = 0.0, 
                          hiddens = hiddens,                               
                          length = 0)

        for _ in range(self.beam_size):
            nodes.put((0, start_node))        

        return Node, nodes, [], []    



    def beam_search(self, input_tensor):
        batch_pred = []
        batch_size = input_tensor.size(0)
        batch_hiddens = self.model.encoder(input_tensor)

        for idx in range(batch_size):
            hiddens = (batch_hiddens[0][:, idx].unsqueeze(1).contiguous(), 
                       batch_hiddens[1][:, idx].unsqueeze(1).contiguous())
            Node, nodes, end_nodes, top_nodes = self.get_nodes(hiddens)
            
            for t in range(self.max_len):
                curr_nodes = []
                while True:
                    try:
                        curr_node = nodes.get()
                        curr_nodes.append(curr_node)
                    except:
                        continue
                    if len(curr_nodes) == self.beam_size:
                        break                    
                
                for curr_score, curr_node in curr_nodes:
                    last_token = curr_node.pred[:, -1]

                    if last_token.item() == self.eos_id:
                        end_nodes.append((curr_score, curr_node))
                        continue

                    out, hidden = self.model.decoder(last_token, curr_node.hiddens)                
                    logits, preds = torch.topk(out, self.beam_size)
                    log_probs = -F.log_softmax(logits, dim=-1)

                    for k in range(self.beam_size):
                        pred = preds[:, k].unsqueeze(0)
                        log_prob = log_probs[:, k].item()
                        pred = torch.cat([curr_node.pred, pred], dim=-1)
                        next_node = Node(prev_node = curr_node,
                                         pred = pred,
                                         log_prob = curr_node.log_prob + log_prob,
                                         hiddens = hidden,
                                         length = curr_node.length+1)

                        next_score = self.get_score(next_node)                        
                        try:
                            nodes.put((next_score, next_node))
                        except:
                            continue                            

                    if not t:
                        break

            if len(end_nodes) == 0:
                _, top_node = nodes.get()
            else:
                _, top_node = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)[0]
            
            pred = top_node.pred.squeeze()[1:]
            batch_pred.append(pred.tolist())

        return batch_pred
    

    def greedy_search(self, input_tensor):
        batch_pred = []
        batch_size = input_tensor.size(0)
        batch_hiddens = self.model.encoder(input_tensor)

        for idx in range(batch_size):
            
            hiddens = (batch_hiddens[0][:, idx].unsqueeze(1).contiguous(), 
                       batch_hiddens[1][:, idx].unsqueeze(1).contiguous())
            
            dec_input = torch.LongTensor(1).fill_(self.bos_id).to(self.device)
            pred = torch.LongTensor(self.max_len).fill_(self.pad_id).to(self.device)

            for t in range(1, self.max_len):
                out, hiddens = self.model.decoder(dec_input, hiddens)
                pred_token = out.argmax(-1)
                pred[t] = pred_token
                dec_input = pred_token

                if pred_token.item() == self.eos_id:
                    break
            batch_pred.append(pred[1:].tolist())

        return batch_pred