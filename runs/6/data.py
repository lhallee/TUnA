import random
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModel
from torch.utils.data import Dataset as TorchDataset


class PPIDataset(TorchDataset):
    def __init__(self, dataset: Dataset, emb_dict: dict):
        self.seqs_a = dataset['SeqA']
        self.seqs_b = dataset['SeqB']
        self.labels = dataset['labels']
        self.emb_dict = emb_dict

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, index):
        seq_a, seq_b, label = self.seqs_a[index], self.seqs_b[index], self.labels[index]
        if random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a
        emb_a = self.emb_dict[seq_a]
        emb_b = self.emb_dict[seq_b]
        return emb_a, emb_b, label
    

class PPICollator:
    def __init__(self, max_length, base_size, test=False):
        self.max_length = max_length
        self.base_size = base_size
        self.test = test

    def make_masks(self, batch_size, lengths, max_length):
        mask = torch.zeros((batch_size, max_length, max_length))
        for i, length in enumerate(lengths):
            # Create a square mask for the non-padded sequences
            mask[i, :length, :length] = 1
        # Expand the mask to 4D: [batch, 1, max_len, max_len]
        mask = mask.unsqueeze(1)
        return mask

    def combine_masks(self, maskA, maskB):
        lenA, lenB = maskA.size(2), maskB.size(2)
        combined_mask = torch.zeros(maskA.size(0), 1, lenA + lenB, lenA + lenB)
        combined_mask[:, :, :lenA, :lenA] = maskA
        combined_mask[:, :, lenA:, lenA:] = maskB
        return combined_mask

    def __call__(self, batch):
        batch_size = len(batch)
        emb_a = [item[0] for item in batch]
        emb_b = [item[1] for item in batch]
        labels = [item[2] for item in batch]

        a_lengths = [len(a) for a in emb_a]
        b_lengths = [len(b) for b in emb_b]
        if self.test:
            max_length = max(max(a_lengths), max(b_lengths))
        else:
            max_length = self.max_length

        final_a_batch = torch.zeros((batch_size, max_length, self.base_size))
        final_b_batch = torch.zeros((batch_size, max_length, self.base_size))

        for i, (a, b) in enumerate(zip(emb_a, emb_b)):
            a_len, b_len = len(a), len(b)
            if a_len <= max_length:
                final_a_batch[i, :a_len, :] = a
            else:
                start_pos = random.randint(0, a_len - max_length)
                final_a_batch[i, :max_length, :] = a[start_pos:start_pos + max_length]
            
            if b_len <= max_length:
                final_b_batch[i, :b_len, :] = b
            else:
                start_pos = random.randint(0, b_len - max_length)
                final_b_batch[i, :max_length, :] = b[start_pos:start_pos + max_length]

        a_mask = self.make_masks(batch_size, a_lengths, max_length)
        b_mask = self.make_masks(batch_size, b_lengths, max_length)
        combined_mask_ab = self.combine_masks(a_mask, b_mask)
        combined_mask_ba = self.combine_masks(b_mask, a_mask)
        labels = torch.tensor(labels, dtype=torch.float)
        
        return {
            'x_a': final_a_batch,
            'x_b': final_b_batch,
            'a_mask': a_mask,
            'b_mask': b_mask,
            'combined_mask_ab': combined_mask_ab,
            'combined_mask_ba': combined_mask_ba,
            'labels': labels
        }


def get_data(config, device):
    # Get uniprot id:protein embedding dictionary via torch.load
    max_length = config['model']['max_sequence_length']
    batch_size = config['training']['batch_size']
    data = load_dataset('Synthyra/bernett_gold_ppi')
    train_data = data['train'].filter(lambda x: len(x['SeqA']) <= max_length and len(x['SeqB']) <= max_length)
    valid_data = data['valid'].filter(lambda x: len(x['SeqA']) <= max_length and len(x['SeqB']) <= max_length)
    test_data = data['test']

    all_seqs = list(set(
        train_data['SeqA'] + train_data['SeqB'] + valid_data['SeqA'] + valid_data['SeqB'] + test_data['SeqA'] + test_data['SeqB']
    ))

    esm150 = AutoModel.from_pretrained('Synthyra/ESMplusplus_large', trust_remote_code=True).to(device).eval()
    # this is a dict of seq:embedding
    embedding_dict = esm150.embed_dataset(
        sequences=all_seqs,
        tokenizer=esm150.tokenizer,
        batch_size=batch_size,
        max_len=100000, # prevent truncation
        full_embeddings=True,
        embed_dtype=torch.float32,
        num_workers=0,
        sql=False,
        sql_db_path='embeddings.db',
        save=True,
        save_path='embeddings.pth',
    )
    
    esm150.cpu()
    del esm150
    torch.cuda.empty_cache()

    return embedding_dict, train_data, valid_data, test_data
