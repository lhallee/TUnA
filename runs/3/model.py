import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
from torch.optim import Adam
from typing import Optional, Tuple
from functools import partial
from einops import rearrange
from utils import *
from data import pack, test_pack
from optimizer import Lookahead
from rotary import RotaryEmbedding


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout, rotary=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = hidden_size // n_heads
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"

        self.w_q = spectral_norm(nn.Linear(hidden_size, hidden_size))
        self.w_k = spectral_norm(nn.Linear(hidden_size, hidden_size))
        self.w_v = spectral_norm(nn.Linear(hidden_size, hidden_size))
        self.out_proj = spectral_norm(nn.Linear(hidden_size, hidden_size))

        self.dropout_rate = dropout
        self.rotary = RotaryEmbedding(hidden_size // n_heads) if rotary else None
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, L, _ = x.shape
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :].expand(b, 1, L, L).bool()
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1).bool()
            elif attention_mask.dim() == 4:
                attention_mask = attention_mask.bool()
            else:
                raise ValueError(f"Invalid attention mask dimension: {attention_mask.dim()}")

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        if self.rotary:
            q, k = self._apply_rotary(q, k)

        q, k, v = map(self.reshaper, (q, k, v))
        a = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_rate,
            is_causal=False
        )
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, seq_len, n_heads * d_head)
        return self.out_proj(a) # (bs, seq_len, hidden_size)


class Feedforward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout, activation_fn):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU implementation requires two projection matrices
        self.w1 = spectral_norm(nn.Linear(hidden_size, intermediate_size))
        self.w2 = spectral_norm(nn.Linear(hidden_size, intermediate_size))
        self.down = spectral_norm(nn.Linear(intermediate_size, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: output = (x * SiLU(W1x)) @ W3
        gate = self.act(self.w1(x))
        value = self.w2(x)
        x = gate * value
        x = self.dropout(x)
        return self.down(x)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, intermediate_size, dropout, activation_fn):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.attn = SelfAttention(hidden_size, n_heads, dropout)
        self.mlp = Feedforward(hidden_size, intermediate_size, dropout, activation_fn)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ln1(x + self.dropout(self.attn(x, attention_mask)))
        x = self.ln2(x + self.dropout(self.mlp(x)))
        return x


class IntraEncoder(nn.Module):
    def __init__(self, prot_dim, hidden_size, n_layers, n_heads, intermediate_size, dropout, activation_fn):
        super().__init__()   
        self.input_proj = spectral_norm(nn.Linear(prot_dim, hidden_size))
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        for _ in range(n_layers):
            self.layer.append(EncoderLayer(hidden_size, n_heads, intermediate_size, dropout, activation_fn))
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x = [batch_size, max_seq_len, protA_dim]
        x = self.input_proj(x)
        # x = [batch size, protA len, hid dim]
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


class InterEncoder(nn.Module):
    def __init__(self, prot_dim, hidden_size, n_layers, n_heads, intermediate_size, dropout, activation_fn):
        super().__init__()
        self.output_dim = prot_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.layer = nn.ModuleList()
        for _ in range(n_layers):
            self.layer.append(EncoderLayer(hidden_size, n_heads, intermediate_size, dropout, activation_fn))

    def forward(self, enc_protA: torch.Tensor, enc_protB: torch.Tensor, combined_mask: torch.Tensor) -> torch.Tensor:
        # Concatenate the encoded representations and masks
        combined = torch.cat([enc_protA, enc_protB], dim=1)

        for layer in self.layer:
            combined = layer(combined, combined_mask)

        combined_mask_2d = combined_mask[:,0,:,0]
        label = torch.sum(combined*combined_mask_2d[:,:,None], dim=1)/combined_mask_2d.sum(dim=1, keepdims=True)
        
        return label 


class ProteinInteractionNet(nn.Module):
    def __init__(self, intra_encoder, inter_encoder, gp_layer, device):
        super().__init__()
        self.intra_encoder = intra_encoder
        self.inter_encoder = inter_encoder
        self.device = device
        self.gp_layer = gp_layer
        self.bce_loss = nn.BCELoss()

    def make_masks(self, prot_lens, protein_max_len):
        N = len(prot_lens)  # batch size
        mask = torch.zeros((N, protein_max_len, protein_max_len), device=self.device)

        for i, lens in enumerate(prot_lens):
            # Create a square mask for the non-padded sequences
            mask[i, :lens, :lens] = 1

        # Expand the mask to 4D: [batch, 1, max_len, max_len]
        mask = mask.unsqueeze(1)
        return mask

    def combine_masks(self, maskA, maskB):
        lenA, lenB = maskA.size(2), maskB.size(2)
        combined_mask = torch.zeros(maskA.size(0), 1, lenA + lenB, lenA + lenB, device=self.device)
        combined_mask[:, :, :lenA, :lenA] = maskA
        combined_mask[:, :, lenA:, lenA:] = maskB
        return combined_mask

    def mean_field_average(self, logits, variance):
        print(f'logits.shape: {logits.shape}')
        print(f'variance.shape: {variance.shape}')
        adjusted_score = logits / torch.sqrt(1. + (np.pi /8.)*variance)
        print(f'adjusted_score.shape: {adjusted_score.shape}')
        adjusted_score = torch.sigmoid(adjusted_score).squeeze()
        print(f'adjusted_score.shape: {adjusted_score.shape}')
        return adjusted_score

    def forward(self, protAs, protBs, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length, last_epoch, train):

        protA_mask = self.make_masks(protA_lens, batch_protA_max_length)
        protB_mask = self.make_masks(protB_lens, batch_protB_max_length)

        enc_protA = self.intra_encoder(protAs, protA_mask)
        enc_protB = self.intra_encoder(protBs, protB_mask)
        
        combined_mask_AB = self.combine_masks(protA_mask, protB_mask)
        combined_mask_BA = self.combine_masks(protB_mask, protA_mask)

        AB_interaction = self.inter_encoder(enc_protA, enc_protB, combined_mask_AB)
        BA_interaction = self.inter_encoder(enc_protB, enc_protA, combined_mask_BA)
        
        #[batch, hidden_size*2]
        ppi_feature_vector, _ = torch.max(torch.stack([AB_interaction, BA_interaction], dim=-1), dim=-1)
        
        ### TRAINING ###
        # IF its not the last epoch, we don't need to update the precision
        if last_epoch==False and train==True:
            logit = self.gp_layer(ppi_feature_vector, update_precision=False)
            return logit
        # IF its the last epoch, we update precision
        elif last_epoch==True and train==True:
            logit =  self.gp_layer(ppi_feature_vector, update_precision=True)
            return logit

        ### TESTING ###
        # IF its not the last epoch, we don't need to get the variance
        elif last_epoch==False and train==False:
            logit = self.gp_layer(ppi_feature_vector, update_precision=False, get_var=False)
            return logit
        #This is the last test epoch. Generate variances.
        elif last_epoch==True and train==False:
            logit, var = self.gp_layer(ppi_feature_vector, update_precision=False, get_var=True)
            return logit, var

    def __call__(self, data, last_epoch, train):
        protAs, protBs, correct_interactions, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length = data
        
        if train:
        # We don't use variances during training
            logit = self.forward(protAs, protBs, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length, last_epoch, train=True)
            mean = torch.sigmoid(logit.squeeze())
            loss = self.bce_loss(mean, correct_interactions.float().squeeze())
            
            return loss

        #Test but not last epoch, we don't use variances still
        elif last_epoch==False and train==False:
            logit = self.forward(protAs, protBs, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length, last_epoch, train=False)
            print(f'logit.shape: {logit.shape}')
            mean = torch.sigmoid(logit.squeeze())
            print(f'mean.shape: {mean.shape}')
            print(f'correct_interactions.shape: {correct_interactions.shape}')
            loss = self.bce_loss(mean, correct_interactions.float().squeeze())
            correct_labels = correct_interactions.cpu().data.numpy()
            
            return loss, correct_labels, mean
        
        #Test and last epoch
        elif last_epoch==True and train==False:
            logit, var = self.forward(protAs, protBs, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length, last_epoch, train=False)
            adjusted_score = self.mean_field_average(logit, var)
            print(f'correct_interactions.shape: {correct_interactions.shape}')
            loss = self.bce_loss(adjusted_score, correct_interactions.float().squeeze())
            correct_labels = correct_interactions.cpu().data.numpy()
            return loss, correct_labels, adjusted_score


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        self.batch = batch
        self._initialize_weights()
        self._setup_optimizer(lr, weight_decay)

    def _initialize_weights(self):
        """ Initialize model weights using Xavier Uniform Initialization for layers with dimension > 1. """
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _setup_optimizer(self, lr, weight_decay):
        """ Setup RAdam + Lookahead optimizer with separate weight decay for biases and weights. """
        weight_p, bias_p = self._separate_weights_and_biases()
        self.optimizer_inner = Adam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, alpha=0.8, k=5)

    def _separate_weights_and_biases(self):
        """ Separate model parameters into weights and biases. """
        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        return weight_p, bias_p

    def train(self, dataset, max_length, protein_dim, device, last_epoch):
        """ Train the model on the provided dataset. """
        self.model.train()
        self.optimizer.zero_grad()
        
        protAs, protBs, labels = zip(*dataset)
        data_pack = pack(protAs, protBs, labels, max_length, protein_dim, device)
        loss = self.model(data_pack, last_epoch, train=True)
        loss.backward()
        self.optimizer.step()
        return loss.item() * len(protAs)
    

class Tester(object):
    def __init__(self, model):
        self.model = model
    
    def test(self, dataset, max_length, protein_dim, last_epoch):
        """ Test the model on the provided dataset. """
        self.model.eval()
        with torch.no_grad():
            protAs, protBs, labels = zip(*dataset)
            data_pack = test_pack(protAs, protBs, labels, max_length, protein_dim, device=self.model.device)

            loss, correct_labels, adjusted_score = self.model(data_pack, last_epoch, train=False)
            T = correct_labels
            Y = np.round(adjusted_score.flatten().cpu().numpy())
            S = adjusted_score.flatten().cpu().numpy()
            
            return loss.item() * len(dataset), T, Y, S
