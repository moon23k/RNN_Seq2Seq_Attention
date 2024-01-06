import random, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.H = config.hidden_dim
        self.embedding = nn.Embedding(config.vocab_size, self.H)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.rnn = nn.GRU(self.H, self.H, bidirectional=True, batch_first=True)


    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        outputs = outputs[:, :, :self.H] + outputs[:, : ,self.H:]
        return outputs, hidden




class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        H = config.hidden_dim
        self.attn_type = config.attn_type

        if self.attn_type == 'general':
            self.attn = nn.Linear(H, H, bias=False)
        elif self.attn_type == 'concat':
            self.attn = nn.Linear(H * 2, H, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(H))


    def forward(self, hidden, enc_outputs):
        if self.attn_type == 'dot':
            attn_energy = torch.sum(hidden * enc_outputs, dim=-1)
            
        elif self.attn_type == 'general':
            attn_energy = self.attn(enc_outputs)
            attn_energy = torch.sum(hidden * attn_energy, dim=-1)

        elif self.attn_type == 'concat':
            concat = torch.cat((hidden.expand(-1, enc_outputs.size(1), -1), enc_outputs), -1)
            attn_energy = self.attn(concat).tanh()
            attn_energy = torch.sum(self.v * attn_energy, dim=-1)
        out = F.softmax(attn_energy, dim=-1)

        return out.unsqueeze(1)




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.H = config.hidden_dim

        self.embedding = nn.Embedding(config.vocab_size, self.H)
        self.dropout = nn.Dropout(config.dropout_ratio)
        
        self.rnn = nn.GRU(self.H, self.H, bidirectional=True, batch_first=True)
        self.concat = nn.Linear(self.H * 2, self.H)
        self.out = nn.Linear(self.H, config.vocab_size)

        self.attention = Attention(config).to(config.device)


    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        rnn_output, hidden = self.rnn(embedded, hidden)
        rnn_output = rnn_output[:, :, :self.H] + rnn_output[:, :, self.H:]
        
        attn_weights = self.attention(rnn_output, encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)

        rnn_output = rnn_output.squeeze(1)
        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = self.concat(concat_input)

        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden        




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
