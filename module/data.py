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
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx]['src']
        trg = self.data[idx]['trg']
        return src, trg




class Collator(object):
    def __init__(self, config):
        self.task = config.task
        self.pad_id = config.pad_id

    def __call__(self, batch):
        if self.task != 'sum':
            return self.base_collate(batch)
        elif self.task == 'sum':
            return self.sum_collate(batch)


    def base_collate(self, batch):
        src_batch, trg_batch = [], []
        
        for src, trg in batch:
            src_batch.append(torch.LongTensor(src))
            trg_batch.append(torch.LongTensor(trg))
        
        src_batch = self.pad_batch(src_batch)
        trg_batch = self.pad_batch(trg_batch)
        
        return {'src': src_batch, 
                'trg': trg_batch}


    def sum_collate(self, batch):
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
        
        pad_seq = [self.pad_id for _ in range(max_seq_len)]
        for _doc in _src_batch:
            doc = []
            for seq in _doc:
                len_gap = max_seq_len - len(seq)
                if len_gap:
                    seq += [self.pad_id] * len_gap
                doc.append(seq)

            num_gap = max_seq_num - len(_doc)
            if num_gap:
                doc.extend([pad_seq for _ in range(num_gap)])

            src_batch.append(doc)

        src_batch = torch.tensor(src_batch, dtype=torch.long)
        trg_batch = self.pad_batch(trg_batch)

        return {'src': src_batch, 
                'trg': trg_batch}


    def pad_batch(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


def load_dataloader(config, split):
    return DataLoader(Dataset(config.task, split), 
                      batch_size=config.batch_size, 
                      shuffle=True if config.mode=='train' else False,
                      collate_fn=Collator(config),
                      pin_memory=True,
                      num_workers=2)