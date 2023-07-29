import random, torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import (
    AdditiveAttention, 
    DotProductAttention, 
    ScaledDotProductAttention
)




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.rnn = nn.GRU(
            config.emb_dim, 
            config.hidden_dim, 
            bidirectional=True, 
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embedded)
        
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        hidden = torch.tanh(self.fc(hidden))

        return out, hidden



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.output_dim = config.vocab_size
        self.dropout = nn.Dropout(config.dropout_ratio)
        
        if config.attn_type == 'additive':
            self.attention = AdditiveAttention(config.hidden_dim)
        elif config.attn_type == 'dot_product':
            self.attention = DotProductAttention()
        elif config.attn_type == 'scaled_dot_product':
            self.attention = ScaledDotProductAttention()


        self.emb = nn.Embedding(self.output_dim, config.emb_dim)
        
        self.rnn = nn.GRU(
            (config.hidden_dim * 2) + config.emb_dim, 
            config.hidden_dim, 
            batch_first=True
        )
        
        self.fc_out = nn.Linear(
            (config.hidden_dim * 3) + config.emb_dim, 
            self.output_dim
        )
        

    def forward(self, x, enc_out, hidden):
        embedded = self.dropout(self.emb(x.unsqueeze(1)))
        attn_value = self.attention(hidden, enc_out).unsqueeze(1)
        weighted = torch.bmm(attn_value, enc_out)
        rnn_input = torch.cat((embedded, weighted), dim=2)    
        out, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        hidden = hidden.squeeze(0)
        out = out.permute(1, 0, 2).squeeze(0)
        assert (out == hidden).all()
        
        embedded = embedded.permute(1, 0, 2).squeeze(0)
        weighted = weighted.permute(1, 0, 2).squeeze(0)
        
        pred = self.fc_out(torch.cat((out, weighted, embedded), dim=1))
        return pred, hidden.squeeze(0)


        
class SeqGenModel(nn.Module):
    def __init__(self, config):
        super(SeqGenModel, self).__init__()

        self.device = config.device
        self.output_dim = config.vocab_size
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, max_len = trg.shape
        outputs = torch.zeros(max_len, batch_size, self.output_dim).to(self.device)

        dec_input = trg[:, 0]
        enc_out, hidden = self.encoder(src)

        for t in range(1, max_len):
            out, hidden = self.decoder(dec_input, enc_out, hidden)
            outputs[t] = out

            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t] if teacher_force else out.argmax(1)
        
        return outputs.contiguous().permute(1, 0, 2)[:, 1:]