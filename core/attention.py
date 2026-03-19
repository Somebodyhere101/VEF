"""
Multi-head attention from frozen SVD.

A transformer learns Q/K/V projections through gradient descent.
SVD provides them analytically: different bands of singular values
capture different aspects of meaning. Each head attends to a
different SVD band.

No learned parameters. No gradient descent.
"""
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention:

    def __init__(self, embeddings, n_heads=4, n_layers=2):
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim = embeddings.dim
        if self.dim % n_heads != 0:
            raise ValueError(
                f"Embedding dimension {self.dim} must be divisible by n_heads {n_heads}")
        self.head_dim = self.dim // n_heads

        E = embeddings.raw[:min(embeddings.vocab_size, 20000)]
        _, S, Vt = np.linalg.svd(E, full_matrices=False)

        # Build Q/K/V projections from SVD bands
        self.W_Q, self.W_K, self.W_V = [], [], []
        for h in range(n_heads):
            s, e = h * self.head_dim, (h + 1) * self.head_dim

            # Query: direct SVD band
            self.W_Q.append(torch.tensor(
                Vt[s:e, :].T.astype(np.float32), device=DEVICE))

            # Key: shifted SVD band (asymmetry helps discrimination)
            shift = self.head_dim // 2
            ks = (s + shift) % self.dim
            ke = ks + self.head_dim
            if ke <= self.dim:
                wk = Vt[ks:ke, :].T
            else:
                wk = np.vstack([Vt[ks:, :], Vt[:ke - self.dim, :]]).T
            self.W_K.append(torch.tensor(wk.astype(np.float32), device=DEVICE))

            # Value: singular-value scaled (important dims contribute more)
            wv = (np.diag(S[s:e]) @ Vt[s:e, :]).T
            wv = wv / np.maximum(np.linalg.norm(wv, axis=0, keepdims=True), 1e-10)
            self.W_V.append(torch.tensor(wv.astype(np.float32), device=DEVICE))

        # Stack W_Q, W_K, W_V for batched multi-head attention
        self.W_Q_stacked = torch.stack(self.W_Q)  # (n_heads, dim, head_dim)
        self.W_K_stacked = torch.stack(self.W_K)  # (n_heads, dim, head_dim)
        self.W_V_stacked = torch.stack(self.W_V)  # (n_heads, dim, head_dim)

        self.W_O = torch.tensor(Vt.T.astype(np.float32), device=DEVICE)

        # Positional encoding
        max_len = 512
        pe = torch.zeros(max_len, self.dim, device=DEVICE)
        pos = torch.arange(max_len, device=DEVICE, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.dim, 2, device=DEVICE, dtype=torch.float32)
                        * (-np.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe

    @torch.no_grad()
    def contextualize(self, token_ids, E, causal=False):
        """Transform static embeddings into contextual ones.

        Each token's representation changes based on all other tokens.
        This is what makes "bank" mean different things near "river"
        versus near "money."
        """
        tids = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
        tids = tids.clamp(0, E.shape[0] - 1)
        n = len(tids)

        if n < 2:
            return E[tids]

        X = E[tids]
        pe_scale = 0.1 * X.norm(dim=1).mean().clamp(min=1e-10)
        if n > len(self.pe):
            self._extend_pe(n * 2)
        H = X + pe_scale * self.pe[:n]

        mask = None
        if causal:
            mask = torch.triu(torch.full((n, n), -1e9, device=DEVICE), diagonal=1)

        for _ in range(self.n_layers):
            # Batched multi-head projections: (n_heads, n, head_dim)
            Q = torch.einsum('nd,hdk->hnk', H, self.W_Q_stacked)
            K = torch.einsum('nd,hdk->hnk', H, self.W_K_stacked)
            V = torch.einsum('nd,hdk->hnk', H, self.W_V_stacked)
            # Attention scores: (n_heads, n, n)
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
            if mask is not None:
                scores = scores + mask.unsqueeze(0)
            attn = F.softmax(scores, dim=2)
            # Weighted values: (n_heads, n, head_dim)
            head_outs = torch.bmm(attn, V)
            # Concatenate heads: (n, n_heads * head_dim)
            concat = head_outs.permute(1, 0, 2).reshape(n, -1)
            H = F.layer_norm(H + concat @ self.W_O, [self.dim])

        return H

    def _extend_pe(self, new_len):
        pe = torch.zeros(new_len, self.dim, device=DEVICE)
        pos = torch.arange(new_len, device=DEVICE, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.dim, 2, device=DEVICE, dtype=torch.float32)
                        * (-np.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe
