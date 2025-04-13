import random
import torch
from tqdm.auto import tqdm
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
    

# Custom collate function for DataLoader
def list_collate(batch):
    proteinA_batch = [item[0] for item in batch]
    proteinB_batch = [item[1] for item in batch]
    y_batch = [item[2] for item in batch]
    return proteinA_batch, proteinB_batch, y_batch


def pack(protAs, protBs, labels, max_length, protein_dim, device):

    N = len(protAs)
    protA_lens = [protA.shape[0] for protA in protAs]
    protB_lens = [protB.shape[0] for protB in protBs]

    #batch_max_length = min(max(max(protA_lens), max(protB_lens)),max_length)
    batch_max_length = max_length


    protAs_new = torch.zeros((N, batch_max_length, protein_dim), device=device)
    for i, protA in enumerate(protAs):
        a_len = protA.shape[0]
        if a_len <= batch_max_length:
            protAs_new[i, :a_len, :] = protA
        else:
            start_pos = random.randint(0,a_len-batch_max_length)
            protAs_new[i, :batch_max_length, :]  = protA[start_pos:start_pos+batch_max_length]
    
    protBs_new = torch.zeros((N, batch_max_length, protein_dim), device=device)
    for i, protB in enumerate(protBs):
        b_len = protB.shape[0]
        if b_len <= batch_max_length:
            protBs_new[i, :b_len, :] = protB
        else:
            start_pos = random.randint(0,b_len-batch_max_length)
            protBs_new[i, :batch_max_length, :]  = protB[start_pos:start_pos+batch_max_length]
    # labels_new: torch.tensor [N,]
    
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    for i, label in enumerate(labels):
        # Convert the label (assuming it's a NumPy array) to a PyTorch tensor
        labels_new[i] = label

    return (protAs_new, protBs_new, labels_new, protA_lens, protB_lens, batch_max_length, batch_max_length)


def test_pack(protAs, protBs, labels, max_length, protein_dim, device):

    N = len(protAs)
    protA_lens = [protA.shape[0] for protA in protAs]
    protB_lens = [protB.shape[0] for protB in protBs]

    batch_protA_max_length = max(protA_lens)
    batch_protB_max_length = max(protB_lens)

    protAs_new = torch.zeros((N, batch_protA_max_length, protein_dim), device=device)
    for i, protA in enumerate(protAs):
        a_len = protA.shape[0]
        protAs_new[i, :a_len, :] = protA

    protBs_new = torch.zeros((N, batch_protB_max_length, protein_dim), device=device)
    for i, protB in enumerate(protBs):
        b_len = protB.shape[0]
        protBs_new[i, :b_len, :] = protB

    # labels_new: torch.tensor [N,]
    
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    for i, label in enumerate(labels):
        # Convert the label (assuming it's a NumPy array) to a PyTorch tensor
        labels_new[i] = label

    return (protAs_new, protBs_new, labels_new, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length)


def get_data(config, device):
    # Get uniprot id:protein embedding dictionary via torch.load
    max_length = config['model']['max_sequence_length']
    batch_size = config['training']['batch_size']
    data = load_dataset('Synthyra/bernett_gold_ppi')
    train_data = data['train'].filter(lambda x: len(x['SeqA']) <= max_length and len(x['SeqB']) <= max_length)
    valid_data = data['validation'].filter(lambda x: len(x['SeqA']) <= max_length and len(x['SeqB']) <= max_length)
    test_data = data['test']

    all_seqs = list(set(
        train_data['SeqA'] + train_data['SeqB'] + valid_data['SeqA'] + valid_data['SeqB'] + test_data['SeqA'] + test_data['SeqB']
    ))

    esm150 = AutoModel.from_pretrained('Synthyra/ESM2-150M', trust_remote_code=True).to(device).eval()
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
