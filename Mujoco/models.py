import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: list[int]):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
        )
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], action_dim * 2))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = self.network(x)
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim: list[int]):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
        )
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU()),
        self.network.append(nn.Linear(hidden_dim[-1], 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, num_head, embedding_dim, layer_idx):
        super(CausalSelfAttention, self).__init__()
        assert embedding_dim % num_head == 0
        self.num_head = num_head
        self.layer_idx = layer_idx
        self.q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None, kv_cache=None):
        batch_size, seq_length, embedding_dim = x.size()
        q = self.q(x).view(batch_size, seq_length, self.num_head, -1)
        k = self.k(x).view(batch_size, seq_length, self.num_head, -1)
        v = self.v(x).view(batch_size, seq_length, self.num_head, -1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
            len_q, len_k = q.size(2), k.size(2)
            mask = torch.zeros((len_q, len_k), dtype=torch.bool, device=q.device)
            mask[:, :len_k - len_q] = True
            mask[:, len_k - len_q:] = torch.tril(torch.ones((len_q, len_q), dtype=torch.bool, device=q.device))
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.proj(output)


class MLP(nn.Module):
    def __init__(self, embedding_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, embedding_dim, num_head, layer_idx):
        super(Block, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(num_head, embedding_dim, layer_idx)
        self.mlp = MLP(embedding_dim)

    def forward(self, x, mask=None, kv_cache=None):
        x = x + self.attn(self.layer_norm1(x), mask=mask, kv_cache=kv_cache)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class NewActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, embedding_dim: int, num_head: int, max_len=2000):
        super(NewActor, self).__init__()
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.proj_state = nn.Linear(state_dim, embedding_dim)
        self.proj_action = nn.Linear(action_dim, embedding_dim)
        self.embed_timestep = nn.Embedding(max_len, embedding_dim)
        self.block_list = nn.ModuleList([Block(embedding_dim, num_head, layer_idx) for layer_idx in range(3)])
        self.output_layer = nn.Linear(embedding_dim, 2 * action_dim)

    def forward(self, x: list[tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]):
        if isinstance(x, tuple):
            x = [x]

        device = x[0][0].device
        input_seqs = []
        lengths = []
        for state, action in x:
            action = action[:-1, :]
            assert state.ndim == action.ndim == 2
            assert state.size(0) == action.size(0) + 1
            length = state.size(0) + action.size(0)
            lengths.append(length)
            input_seq = torch.empty([length, self.embedding_dim], device=device)
            input_seq[0::2] = self.proj_state(state)
            input_seq[1::2] = self.proj_action(action)
            input_seqs.append(input_seq)

        input_seqs = pad_sequence(input_seqs, batch_first=True)

        batch_size, max_length, _ = input_seqs.size()
        lengths = torch.tensor(lengths, device=device)
        padding_mask = torch.arange(max_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        padding_mask = padding_mask.view(-1, 1, 1, max_length)

        causal_mask = torch.tril(torch.ones([max_length, max_length], device=device, dtype=torch.bool))
        causal_mask = causal_mask.view(1, 1, max_length, max_length)

        mask = padding_mask & causal_mask

        time_steps = torch.arange(max_length, device=device)
        time_steps = time_steps.unsqueeze(0).repeat(batch_size, 1)
        pos_embedding = self.embed_timestep(time_steps)
        input_seqs = input_seqs + pos_embedding

        for block in self.block_list:
            input_seqs = block(input_seqs, mask=mask)

        output = self.output_layer(input_seqs[:, 0::2, :])
        mu, log_std = torch.split(output, self.action_dim, dim=-1)
        std = torch.sigmoid(log_std)
        mu = torch.cat([mu[i, :(length + 1) // 2] for i, length in enumerate(lengths)])
        std = torch.cat([std[i, :(length + 1) // 2] for i, length in enumerate(lengths)])

        return mu, std

    def inference(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor], kv_cache):
        if isinstance(x, torch.Tensor):
            x = self.proj_state(x).unsqueeze(0)
            x = x + self.embed_timestep(torch.tensor([0], device=x.device))
        else:
            action, state = x
            state = self.proj_state(state)
            action = self.proj_action(action)
            x = torch.stack([action, state])
            prefix_len = kv_cache.get_prefix_len()
            x = x + self.embed_timestep(torch.tensor([prefix_len, prefix_len + 1], device=x.device))

        x = x.unsqueeze(0)

        for block in self.block_list:
            x = block(x, kv_cache=kv_cache)

        x = x.squeeze(0)

        output = self.output_layer(x[-1])
        mu, log_std = torch.split(output, self.action_dim)
        std = torch.sigmoid(log_std)
        return mu, std


class KvCache:
    def __init__(self):
        self.kv_cache = []

    def insert_kv(self, layer_idx, k, v):
        if len(self.kv_cache) <= layer_idx:
            self.kv_cache.append((k, v))
            return k, v
        else:
            k_cache, v_cache = self.kv_cache[layer_idx]
            k_cache = torch.cat((k_cache, k), dim=2)
            v_cache = torch.cat((v_cache, v), dim=2)
            self.kv_cache[layer_idx] = (k_cache, v_cache)
            return k_cache, v_cache

    def clear(self):
        self.kv_cache = []

    def get_prefix_len(self):
        if not self.kv_cache:
            return 0
        return self.kv_cache[0][0].size(2)
