import random, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.H = config.hidden_dim
        self.bos_id = config.bos_id
        self.embedding = nn.Embedding(config.vocab_size, self.H)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.rnn = nn.GRU(self.H, self.H, batch_first=True)
        

    def forward(self, x):
        enc_mask = x == self.bos_id
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded)
        return output, hidden, enc_mask



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
    
        H = config.hidden_dim
        self.embedding = nn.Embedding(config.vocab_size, H)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.rnn = nn.GRU(H, H, batch_first=True)
        
        self.gru = nn.GRU(H, H, batch_first=True)
        self.fc_out = nn.Linear(H * 2, config.vocab_size)

        self.attn_type = config.attn_type
        if self.attn_type == 'additive':
            self.W_q = nn.Linear(H, H, bias=False)
            self.W_k = nn.Linear(H, H, bias=False)
            self.V = nn.Linear(H, 1, bias=False) 


    def attention(self, q, k, mask):
        if self.attn_type == 'additive':
            score = self.V(torch.tanh(self.W_q(q) + self.W_k(k)))
        else:
            score = torch.bmm(k, q.permute(0, 2, 1))
            if 'scaled' in self.attn_type:
                score /= score.size(-1)
        
        score = score.squeeze()
        score = score.masked_fill(mask==0, -1e4)
        
        weight = F.softmax(score, dim=-1).unsqueeze(1)
        value = torch.bmm(weight, k)
        
        return value, weight
    

    def forward(self, x, hiddens, enc_outputs, enc_mask):
        out = self.dropout(self.embedding(x))
        out, hiddens = self.rnn(out, hiddens)
        
        context, weights = self.attention(out, enc_outputs, enc_mask)
        out = torch.cat((out, context), dim=-1)
        out = self.fc_out(out)

        out = out.squeeze()
        return out, hiddens, weights


class Seq2SeqModel(nn.Module):
    def __init__(self, config):
        super(Seq2SeqModel, self).__init__()

        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.out = namedtuple('Out', 'logit loss weight')
        self.criterion = nn.CrossEntropyLoss().to(self.device)        


    def forward(self, x, y, teacher_forcing_ratio=0.5):

        batch_size, max_len = y.shape
        
        outputs = torch.Tensor(batch_size, max_len, self.vocab_size)
        outputs = outputs.fill_(self.pad_id).to(self.device)

        dec_input = y[:, :1]
        enc_out, hidden, enc_mask = self.encoder(x)

        for t in range(1, max_len):
            out, hidden, weight = self.decoder(dec_input, hidden, enc_out, enc_mask)

            outputs[:, t] = out
            pred = out.argmax(-1)

            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = y[:, t] if teacher_force else pred
            dec_input = dec_input.unsqueeze(1)
            
        logit = outputs[:, 1:] 
        
        self.out.weight = weight
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            y[:, 1:].contiguous().view(-1)
        )
        
        return self.out 
        