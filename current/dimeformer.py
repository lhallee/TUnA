import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
from torch.optim import Adam
from typing import Optional, Tuple
from functools import partial
from utils import *
from optimizer import Lookahead
from attention import SelfAttention, CrossAttention


Linear = partial(nn.Linear, bias=False)


def pad_and_concatenate_dimer(
        A: torch.Tensor,
        B: torch.Tensor,
        a_mask: Optional[torch.Tensor] = None,
        b_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given two sequences A and B with masks, pad (if needed) and concatenate them.
    """
    batch_size, L, d = A.size()
    if a_mask is None:
        a_mask = torch.ones(batch_size, L, device=A.device)
    if b_mask is None:
        b_mask = torch.ones(batch_size, L, device=A.device)
    # Compute the maximum (valid) length in the batch.
    max_len = max(
        int(a_mask[i].sum().item() + b_mask[i].sum().item())
        for i in range(batch_size)
    )
    combined = torch.zeros(batch_size, max_len, d, device=A.device)
    combined_mask = torch.zeros(batch_size, max_len, device=A.device)
    for i in range(batch_size):
        a_len = int(a_mask[i].sum().item())
        b_len = int(b_mask[i].sum().item())
        combined[i, :a_len] = A[i, :a_len]
        combined[i, a_len:a_len+b_len] = B[i, :b_len]
        combined_mask[i, :a_len+b_len] = 1
    return combined, combined_mask


class Feedforward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout, use_spectral_norm=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU implementation requires two projection matrices
        self.w1 = spectral_norm(Linear(hidden_size, intermediate_size)) if use_spectral_norm else Linear(hidden_size, intermediate_size)
        self.w2 = spectral_norm(Linear(hidden_size, intermediate_size)) if use_spectral_norm else Linear(hidden_size, intermediate_size)
        self.down = spectral_norm(Linear(intermediate_size, hidden_size)) if use_spectral_norm else Linear(intermediate_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: output = (x * SiLU(W1x)) @ W3
        gate = self.act(self.w1(x))
        value = self.w2(x)
        x = gate * value
        x = self.dropout(x)
        return self.down(x)


class DimeformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, intermediate_size: int, dropout: float, use_spectral_norm: bool = True):
        super().__init__()
        self.cross_ab = CrossAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            rotary=True,
            use_spectral_norm=use_spectral_norm,
            causal=False
        )
        self.cross_ba = CrossAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            rotary=True,
            use_spectral_norm=use_spectral_norm,
            causal=False
        )
        self.self_a = SelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            rotary=True,
            use_spectral_norm=use_spectral_norm,
            causal=False,
        )
        self.self_b = SelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            rotary=True,
            use_spectral_norm=use_spectral_norm,
            causal=False,
        )
        self.ffn_a_1 = Feedforward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout
        )
        self.ffn_a_2 = Feedforward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout
        )
        self.ffn_b_1 = Feedforward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout
        )
        self.ffn_b_2 = Feedforward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout
        )

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.ln4 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        attention_mask_a: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_a = self.ln1(x_a)
        x_b = self.ln2(x_b)

        # Intra representation
        x_a = self.self_a(x_a, attention_mask_a) + x_a
        x_b = self.self_b(x_b, attention_mask_b) + x_b
        x_a = self.ffn_a_1(x_a) + x_a
        x_b = self.ffn_b_1(x_b) + x_b

        x_a = self.ln3(x_a)
        x_b = self.ln4(x_b)

        # Inter representation
        res_a, res_b = x_a, x_b
        x_a = self.cross_ab(res_a, res_b, attention_mask_a, attention_mask_b) + res_a
        x_b = self.cross_ba(res_b, res_a, attention_mask_b, attention_mask_a) + res_b
        x_a = self.ffn_a_2(x_a) + x_a
        x_b = self.ffn_b_2(x_b) + x_b
        return x_a, x_b


class DimeformerBase(nn.Module):
    def __init__(self, n_layers: int, hidden_size: int, n_heads: int, intermediate_size: int, dropout: float, use_spectral_norm: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([DimeformerBlock(
            hidden_size, n_heads, intermediate_size, dropout, use_spectral_norm
            ) for _ in range(n_layers)])
        self.hidden_size = hidden_size

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        attention_mask_a: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (b, L1, d), (b, L2, d) -> (b, L, d)
        for block in self.blocks:
            x_a, x_b = block(x_a, x_b, attention_mask_a, attention_mask_b)
        return pad_and_concatenate_dimer(x_a, x_b, attention_mask_a, attention_mask_b)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, intermediate_size, dropout, use_spectral_norm=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.attn = SelfAttention(hidden_size, n_heads, dropout, use_spectral_norm)
        self.mlp = Feedforward(hidden_size, intermediate_size, dropout, use_spectral_norm)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ln1(x + self.dropout(self.attn(x, attention_mask)))
        x = self.ln2(x + self.dropout(self.mlp(x)))
        return x


class IntraEncoder(nn.Module):
    def __init__(self, prot_dim, hidden_size, n_layers, n_heads, intermediate_size, dropout, use_spectral_norm=True):
        super().__init__()   
        self.input_proj = spectral_norm(Linear(prot_dim, hidden_size)) if use_spectral_norm else Linear(prot_dim, hidden_size)
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        for _ in range(n_layers):
            self.layer.append(EncoderLayer(hidden_size, n_heads, intermediate_size, dropout, use_spectral_norm))
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x = [batch_size, max_seq_len, protA_dim]
        x = self.input_proj(x)
        # x = [batch size, protA len, hid dim]
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


class InterEncoder(nn.Module):
    def __init__(self, prot_dim, hidden_size, n_layers, n_heads, intermediate_size, dropout, use_spectral_norm=True):
        super().__init__()
        self.output_dim = prot_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.layer = nn.ModuleList()
        for _ in range(n_layers):
            self.layer.append(EncoderLayer(hidden_size, n_heads, intermediate_size, dropout, use_spectral_norm))

    def forward(self, enc_protA: torch.Tensor, enc_protB: torch.Tensor, combined_mask: torch.Tensor) -> torch.Tensor:
        # Concatenate the encoded representations and masks
        combined = torch.cat([enc_protA, enc_protB], dim=1)

        for layer in self.layer:
            combined = layer(combined, combined_mask)

        combined_mask_2d = combined_mask[:,0,:,0]
        label = torch.sum(combined*combined_mask_2d[:,:,None], dim=1)/combined_mask_2d.sum(dim=1, keepdims=True)
        return label


class ProteinInteractionNet(nn.Module):
    def __init__(self, intra_encoder, inter_encoder, gp_layer, device, use_spectral_norm=True):
        super().__init__()
        self.intra_encoder = intra_encoder
        self.inter_encoder = inter_encoder
        hidden_size = self.inter_encoder.hidden_size

        self.final_proj = spectral_norm(Linear(hidden_size * 2, hidden_size)) if use_spectral_norm else Linear(hidden_size * 2, hidden_size)
        self.device = device
        self.gp_layer = gp_layer
        self.bce_loss = nn.BCELoss()

    def mean_field_average(self, logits, variance):
        adjusted_score = logits / torch.sqrt(1. + (np.pi /8.)*variance)
        adjusted_score = torch.sigmoid(adjusted_score).squeeze()
        return adjusted_score

    def forward(
            self,
            x_a,
            x_b,
            a_mask,
            b_mask,
            combined_mask_ab,
            combined_mask_ba,
            labels,
            last_epoch,
            train
        ):

        x_a = self.intra_encoder(x_a, a_mask)
        x_b = self.intra_encoder(x_b, b_mask)

        x_ab = self.inter_encoder(x_a, x_b, combined_mask_ab)
        x_ba = self.inter_encoder(x_b, x_a, combined_mask_ba)

        ppi_feature_vector, _ = torch.max(torch.stack([x_ab, x_ba], dim=-1), dim=-1)

        ### TRAINING ###
        # IF its not the last epoch, we don't need to update the precision
        if last_epoch==False and train==True:
            logit = self.gp_layer(ppi_feature_vector, update_precision=False)
            mean = torch.sigmoid(logit.squeeze())
            loss = self.bce_loss(mean, labels.squeeze())
            return loss
        
        # IF its the last epoch, we update precision
        elif last_epoch==True and train==True:
            logit =  self.gp_layer(ppi_feature_vector, update_precision=True)
            mean = torch.sigmoid(logit.squeeze())
            loss = self.bce_loss(mean, labels.squeeze())
            return loss

        ### TESTING ###
        # IF its not the last epoch, we don't need to get the variance
        elif last_epoch==False and train==False:
            logit = self.gp_layer(ppi_feature_vector, update_precision=False, get_var=False)
            mean = torch.sigmoid(logit.squeeze())
            loss = self.bce_loss(mean, labels.squeeze())
            correct_labels = labels.cpu().numpy()
            return loss, correct_labels, mean
    
        #This is the last test epoch. Generate variances.
        elif last_epoch==True and train==False:
            logit, var = self.gp_layer(ppi_feature_vector, update_precision=False, get_var=True)
            adjusted_score = self.mean_field_average(logit.squeeze(1), var)
            loss = self.bce_loss(adjusted_score, labels.squeeze())
            correct_labels = labels.cpu().numpy()
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
            [
                {'params': weight_p, 'weight_decay': weight_decay},
                {'params': bias_p, 'weight_decay': 0}
            ],
                lr=lr
        )
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

    def train(self, batch, last_epoch):
        """ Train the model on the provided dataset. """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model(**batch, last_epoch=last_epoch, train=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Tester(object):
    def __init__(self, model):
        self.model = model
    
    def test(self, batch, last_epoch):
        """ Test the model on the provided dataset. """
        self.model.eval()
        with torch.no_grad():
            loss, correct_labels, adjusted_score = self.model(**batch, last_epoch=last_epoch, train=False)
            T = correct_labels
            Y = np.round(adjusted_score.flatten().cpu().numpy())
            S = adjusted_score.flatten().cpu().numpy()
            
            return loss.item(), T, Y, S
