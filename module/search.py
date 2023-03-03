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
        self.Node = namedtuple('Node', ['prev_node', 'pred', 'log_prob', 'hidden', 'length'])


    def get_score(self, node, max_repeat = 5, min_length=5, alpha=1.2):
        #find max number of consecutively repeated tokens 
        repeat = max([sum(1 for token in group if token != self.pad_id) for _, group in groupby(node.pred.tolist())])
        #repeat = max([node.pred.tolist().count(token) for token in node.pred.tolist() if token != self.pad_id])

        if repeat > max_repeat:
            repeat_penalty = 0.5
        else:
            repeat_penalty = 1
        
        len_penalty = ((node.length + min_length) / (1 + min_length)) ** alpha
        score = node.log_prob / len_penalty
        score = score * repeat_penalty
        return score


    def get_nodes(self, hidden):
        Node = self.Node
        nodes = PriorityQueue()
        start_tensor = torch.LongTensor([[self.bos_id]]).to(self.device)

        start_node = Node(prev_node = None, 
                          pred = start_tensor, 
                          log_prob = 0.0, 
                          hidden = hidden,                               
                          length = 0)

        for _ in range(self.beam_size):
            nodes.put((0, start_node))        

        return Node, nodes, [], []    


    def beam_search(self, input_tensor):
        hidden = self.model.encoder(input_tensor)
        Node, nodes, end_nodes, top_nodes = self.get_nodes(hidden=hidden)

        for t in range(self.max_len):
            curr_nodes = [nodes.get() for _ in range(self.beam_size)]
            
            for curr_score, curr_node in curr_nodes:
                if curr_node.pred[:, -1].item() == self.eos_id and curr_node.prev_node != None:
                    end_nodes.append((curr_score, curr_node))
                    continue

                out, hidden = self.model.decoder(curr_node.pred[:, -1], curr_node.hidden)                
                logits, preds = torch.topk(out, self.beam_size)
                logits, preds = logits, preds
                log_probs = -F.log_softmax(logits, dim=-1)

                for k in range(self.beam_size):
                    pred = preds[:, k].unsqueeze(0)
                    log_prob = log_probs[:, k].item()
                    pred = torch.cat([curr_node.pred, pred], dim=-1)

                    next_node = Node(prev_node = curr_node,
                                     pred = pred,
                                     log_prob = curr_node.log_prob + log_prob,
                                     hidden = hidden,
                                     length = curr_node.length + 1)
                    
                    next_score = self.get_score(next_node, max_repeat)
                    nodes.put((next_score, next_node))    

                if not t:
                    break

        if len(end_nodes) == 0:
            _, top_node = nodes.get()
        else:
            _, top_node = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)[0]
              
        return top_node.pred[:, 1:]
    

    def greedy_search(self, input_tensor):
        hiddens = self.model.encoder(input_tensor)
        output_tensor = torch.zeros(self.max_len, input_tensor.size(0)).to(self.device)
        dec_input = output_tensor[:, 0]
        
        for i in range(1, self.max_len):
            out, hiddens = self.model.decoder(dec_input, hiddens)
            output_tensor[:, i] = out.argmax(-1)
            dec_input = output_tensor[:, i]

        return output_tensor.contiguous().permute(1, 0, 2)[:, 1:]