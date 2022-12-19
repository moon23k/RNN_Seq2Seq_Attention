import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, task, split):
        super().__init__()
        self.data = self.load_data(task, split)

    @staticmethod
    def load_data(task, split):
        with open(f"data/{task}/{split}.json", 'r') as f:
            data = json.load(f)
        if split=='train':
            return data[::3]
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx]['src']
        trg = self.data[idx]['trg']
        return src, trg



def load_dataloader(config, split):
    global pad_id
    pad_id = config.pad_id    

    def base_collate(batch):
        src_batch, trg_batch = [], []
        
        for src, trg in batch:
            src_batch.append(torch.LongTensor(src))
            trg_batch.append(torch.LongTensor(trg))
        
        src_batch = pad_sequence(src_batch,
                                 batch_first=True,
                                 padding_value=pad_id)
        
        trg_batch = pad_sequence(trg_batch, 
                                 batch_first=True, 
                                 padding_value=pad_id)
        
        return {'src': src_batch, 
                'trg': trg_batch}


    def sum_collate(batch):
        src_batch, _src_batch, trg_batch = [], [], []
        max_seq_num, max_seq_len = 0, 0

        for src, trg in batch:
            _src_batch.append(src)
            trg_batch.append(torch.tensor(trg, dtype=torch.long))

            if max_seq_num < len(src):
                max_seq_num = len(src)

            for seq in src:
                if max_seq_len < len(seq):
                    max_seq_len = len(seq)
        
        pad_seq = [pad_id for _ in range(max_seq_len)]
        for _doc in _src_batch:
            doc = []
            for seq in _doc:
                len_gap = max_seq_len - len(seq)
                if len_gap:
                    seq += [pad_id] * len_gap
                doc.append(seq)

            num_gap = max_seq_num - len(_doc)
            if num_gap:
                doc.extend([pad_seq for _ in range(num_gap)])

            src_batch.append(doc)

        src_batch = torch.tensor(src_batch, dtype=torch.long)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=pad_id)

        return {'src': src_batch, 
                'trg': trg_batch}


    if config.task == 'sum':
        return DataLoader(Dataset(config.task, split), 
                          batch_size=config.batch_size, 
                          shuffle=True, 
                          collate_fn=sum_collate,
                          num_workers=2)


    return DataLoader(Dataset(config.task, split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=base_collate,
                      num_workers=2)