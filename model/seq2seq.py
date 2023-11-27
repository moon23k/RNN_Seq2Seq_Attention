import random, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        H = config.hidden_dim

        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.rnn = nn.GRU(config.emb_dim, H, batch_first=True)
        self.fc = nn.Linear(H * 2, H)
        

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embedded)
        
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        hidden = torch.tanh(self.fc(hidden))

        return out, hidden


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        H = config.hidden_dim

        self.embedding = nn.Embedding(config.vocab_size, H)
        self.gru = nn.GRU(H, H, batch_first=True)

        self.dropout = nn.Dropout(config.dropout_ratio)
        self.fc_out=  nn.Linear(H, config.vocab_size)

        self.attn_type = config.attn_type
        if self.attn_type == 'additive':
            self.Wa = nn.Linear(H, H)
            self.Ua = nn.Linear(H, H)
            self.Va = nn.Linear(H, 1)


    def attention(self, query, key):

        if self.attn_type == 'additive':    
            scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
            scores = scores.squeeze(2).unsqueeze(1)
            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights, keys)
            
        else:
            scores = key.bmm(query)

            context = pass
            if 'scaled' in self.attn_type:
                context /= scale_factor            

        return context


    def forward(self, x, hidden, encoder_outputs):
        output = self.dropout(self.embedding(x))

        if self.attn_type == 'additive':            
            
            context = self.attention(
                query = hidden.permute(1, 0, 2), # [ batch_size, 1, hidden_dim ]
                key = hidden
            )

            input_gru = torch.cat((embedded, context), dim=2)
            output, hidden = self.gru(input_gru, hidden)

        else:
            output, hidden = self.rnn(x, hidden)
            score = encoder_outputs.bmm(hidden)
            weights = F.softmax(scores, dim=-1)

            context = torch.bmm(weights, encoder_outputs)
            output = torch.cat((embedded, context), dim=2)
        
        return self.fc_out(output)



        
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

        dec_input = y[:, 0]
        enc_out, hidden = self.encoder(x)

        for t in range(1, max_len):
            out, hidden = self.decoder(dec_input, enc_out, hidden)
            outputs[:, t] = out
            pred = out.argmax(-1)

            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = y[:, t] if teacher_force else pred
        
        logit = outputs[:, 1:] 
        
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            y[:, 1:].contiguous().view(-1)
        )
        
        return self.out 
        