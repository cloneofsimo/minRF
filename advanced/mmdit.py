## MM DiT model that was proposed by SD3 paper.
# I've tried to make this follow the work of MuP, so they scale in maximal feature-learning regime.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None):
        super().__init__()

        if not hidden_dim:
            hidden_dim = dim * 4
        if not out_dim:
            out_dim = dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class DoubleAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.w1q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.w1k = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.w1v = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.w1o = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.w2q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.w2k = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.w2v = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.w2o = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm1 = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm1 = nn.LayerNorm(self.n_heads * self.head_dim)

        self.q_norm2 = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm2 = nn.LayerNorm(self.n_heads * self.head_dim)

    def forward(self, c, x):

        bsz, seqlen1, _ = c.shape
        bsz, seqlen2, _ = x.shape
        seqlen = seqlen1 + seqlen2

        cq, ck, cv = self.w1q(c), self.w1k(c), self.w1v(c)
        cq, ck = self.q_norm1(cq), self.k_norm1(ck)
        xq, xk, xv = self.w2q(x), self.w2k(x), self.w2v(x)
        xq, xk = self.q_norm2(xq), self.k_norm2(xk)

        # concat all
        q = torch.cat((cq, xq), dim=1)
        k = torch.cat((ck, xk), dim=1)
        v = torch.cat((cv, xv), dim=1)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        output = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)
        c, x = output.split([seqlen1, seqlen2], dim=1)
        c = self.w1o(c)
        x = self.w2o(x)
        return c, x


class MMDiTBlock(nn.Module):
    def __init__(self, dim, heads=8, global_conddim=1024, is_last=False):
        super().__init__()

        self.normC1 = nn.LayerNorm(dim)
        self.normC2 = nn.LayerNorm(dim)
        if not is_last:
            self.mlpC = MLP(dim, hidden_dim=dim * 4)
            self.modC = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_conddim, 6 * dim, bias=True),
            )
        else:
            self.modC = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_conddim, 2 * dim, bias=True),
            )

        self.normX1 = nn.LayerNorm(dim)
        self.normX2 = nn.LayerNorm(dim)
        self.mlpX = MLP(dim, hidden_dim=dim * 4)
        self.modX = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, 6 * dim, bias=True),
        )

        self.attn = DoubleAttention(dim, heads)
        self.is_last = is_last

    def forward(self, c, x, global_cond, **kwargs):

        cres, xres = c, x
        # cpath
        if not self.is_last:
            cshift_msa, cscale_msa, cgate_msa, cshift_mlp, cscale_mlp, cgate_mlp = (
                self.modC(global_cond).chunk(6, dim=1)
            )
        else:
            cshift_msa, cscale_msa = self.modC(global_cond).chunk(2, dim=1)

        c = modulate(self.normC1(c), cshift_msa, cscale_msa)

        # xpath
        xshift_msa, xscale_msa, xgate_msa, xshift_mlp, xscale_mlp, xgate_mlp = (
            self.modX(global_cond).chunk(6, dim=1)
        )

        x = modulate(self.normX1(x), xshift_msa, xscale_msa)

        # attention

        c, x = self.attn(c, x)

        if not self.is_last:
            c = self.normC2(cres + cgate_msa.unsqueeze(1) * c)
            c = cgate_mlp.unsqueeze(1) * self.mlpC(modulate(c, cshift_mlp, cscale_mlp))
            c = cres + c

        x = self.normX2(xres + xgate_msa.unsqueeze(1) * x)
        x = xgate_mlp.unsqueeze(1) * self.mlpX(modulate(x, xshift_mlp, xscale_mlp))
        x = xres + x

        return c, x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class MMDiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        dim=2048,
        n_layers=8,
        n_heads=4,
        global_conddim=1024,
        cond_seq_dim=2048,
        cond_vector_dim=1024,
        max_seq=32 * 32,
    ):
        super().__init__()

        self.t_embedder = TimestepEmbedder(global_conddim)
        self.c_vec_embedder = MLP(cond_vector_dim, global_conddim)

        self.cond_linear = nn.Linear(
            cond_seq_dim, dim
        )  # linear for something like text sequence.
        self.init_x_linear = nn.Linear(
            patch_size * patch_size * in_channels, dim
        )  # init linear for patchified image.

        self.pe = nn.Parameter(torch.randn(1, max_seq, dim) * 0.1)

        self.layers = nn.ModuleList([])
        for idx in range(n_layers):
            self.layers.append(
                MMDiTBlock(dim, n_heads, global_conddim, is_last=(idx == n_layers - 1))
            )

        self.final_linear = nn.Linear(
            dim, patch_size * patch_size * out_channels, bias=True
        )

        self.modF = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, 2 * dim, bias=True),
        )
        # # init zero
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

        self.out_channels = out_channels
        self.patch_size = patch_size

        for pn, p in self.named_parameters():
            if pn.endswith("w1o.weight") or pn.endswith("w2o.weight"):
                # this is muP
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers * dim))
            # if its modulation
            if "mod" in pn:
                nn.init.constant_(p, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, conds, **kwargs):
        # patchify x, add PE
        x = self.init_x_linear(self.patchify(x))  # B, T_x, D
        x = x + self.pe[:, : x.size(1)]

        # process conditions for MMDiT Blocks
        c_seq = conds["c_seq"]  # B, T_c, D_c
        c_vec = conds["c_vec"]  # B, D_gc
        c = self.cond_linear(c_seq)  # B, T_c, D
        t_emb = self.t_embedder(t)  # B, D

        global_cond = self.c_vec_embedder(c_vec) + t_emb  # B, D

        for layer in self.layers:
            c, x = layer(c, x, global_cond, **kwargs)

        fshift, fscale = self.modF(global_cond).chunk(2, dim=1)

        x = modulate(x, fshift, fscale)
        x = self.final_linear(x)
        x = self.unpatchify(x)
        return x


class MMDiT_for_IN1K(MMDiT):
    # This will "simulate" having clip.
    # it will act as having clip that encodes both clip global vector and clip sequence vector.
    # in reality this is just one hot encoding.
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        dim=1024,
        n_layers=8,
        n_heads=4,
        global_conddim=1024,
        cond_seq_dim=2048,
        cond_vector_dim=1024,
        max_seq=32 * 32,
    ):
        super(MMDiT_for_IN1K, self).__init__(
            in_channels,
            out_channels,
            patch_size,
            dim,
            n_layers,
            n_heads,
            global_conddim,
            cond_seq_dim,
            cond_vector_dim,
            max_seq,
        )

    def forward(self, x, t, conds, **kwargs):
        # one hot
        conds_1 = F.one_hot(conds, num_classes=2048).unsqueeze(1).float()
        conds_2 = F.one_hot(conds + 1024, num_classes=2048).unsqueeze(1).float()
        conds_g = F.one_hot(conds, num_classes=1024).float()

        conds_dict = {"c_seq": torch.cat([conds_1, conds_2], dim=1), "c_vec": conds_g}
        return super(MMDiT_for_IN1K, self).forward(x, t, conds_dict, **kwargs)


if __name__ == "__main__":
    # model = MMDiT()
    # x = torch.randn(2, 4, 32, 32)
    # t = torch.randn(2)
    # conds = {"c_seq": torch.randn(2, 32, 2048), "c_vec": torch.randn(2, 1024)}
    # out = model(x, t, conds)
    # print(out.shape)
    # print(out)

    model = MMDiT_for_IN1K()
    x = torch.randn(2, 4, 32, 32)
    t = torch.randn(2)
    conds = torch.randint(0, 1000, (2,))
    out = model(x, t, conds)
    print(out.shape)
