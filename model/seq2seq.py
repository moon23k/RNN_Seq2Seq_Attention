import random, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        H = config.hidden_dim
        self.embedding = nn.Embedding(config.vocab_size, H)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.rnn = nn.GRU(H, H, batch_first=True)
        

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        return self.rnn(embedded)



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        H = config.hidden_dim
        self.embedding = nn.Embedding(config.vocab_size, H)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.fc_out = nn.Linear(H, config.vocab_size)

        self.attn_type = config.attn_type
        if self.attn_type == 'additive':
            self.W_q = nn.Linear(H, H, bias=False)
            self.W_k = nn.Linear(H, H, bias=False)
            self.V = nn.Linear(H, 1, bias=False) 
            self.gru = nn.GRU(H * 2, H, batch_first=True)
        else:
            self.gru = nn.GRU(H, H, batch_first=True)
            self.concat = nn.Linear(H * 2, H)
            


    def forward(self, x, hidden, enc_outputs):
        out = self.dropout(self.embedding(x))
        
        #Additive Attention
        if self.attn_type == 'additive':
            attn_value = self.attention(hidden, enc_outputs)
            new_input = torch.cat((out, attn_value), dim=-1)
            dec_out, hidden = self.gru(new_input, hidden)

        #Dot-Product or Scaled Dot-Product Attention
        elif 'dot-product' in self.attn_type:
            dec_out, hidden = self.gru(out, hidden)
            attn_value = self.attention(dec_out, enc_outputs)
            dec_out = torch.cat((dec_out, attn_value), dim=-1)
            dec_out = torch.tanh(self.concat(dec_out))

        return self.fc_out(dec_out).squeeze(), hidden


    def attention(self, q, k):
        if self.attn_type == 'additive':
            q = q.permute(1, 0, 2)
            attn_score = self.V(torch.tanh(self.W_q(q) + self.W_k(k)))
            attn_score = attn_score.permute(0, 2, 1)
        else:
            attn_score = torch.bmm(q, k.permute(0,2,1))
            if 'scaled' in self.attn_type:
                attn_score /= attn_score.size(-1)
        
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_value = torch.bmm(attn_weight, k)
        return attn_value



        
class Seq2SeqModel(nn.Module):
    def __init__(self, config):
        super(Seq2SeqModel, self).__init__()

        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.out = namedtuple('Out', 'logit loss')
        self.criterion = nn.CrossEntropyLoss().to(self.device)        


    def forward(self, x, y, teacher_forcing_ratio=0.5):

        batch_size, max_len = y.shape
        
        outputs = torch.Tensor(batch_size, max_len, self.vocab_size)
        outputs = outputs.fill_(self.pad_id).to(self.device)

        dec_input = y[:, :1]
        enc_out, hidden = self.encoder(x)

        for t in range(1, max_len):
            out, hidden = self.decoder(dec_input, hidden, enc_out)

            outputs[:, t] = out
            pred = out.argmax(-1)

            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = y[:, t] if teacher_force else pred
            dec_input = dec_input.unsqueeze(1)
            
        logit = outputs[:, 1:] 
        
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            y[:, 1:].contiguous().view(-1)
        )
        
        return self.out 
        