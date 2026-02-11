import os
import sys
import math
import time
import json
import argparse
import urllib.request
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import tqdm.auto as tqdm


# ============================================================================
# Configuration and Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Signature Decision Transformer on D4RL datasets (MuJoCo, Maze2D, AntMaze)"
    )
    
    # Dataset
    parser.add_argument("--env", type=str, default="halfcheetah-medium",
                        help="D4RL environment name (e.g., halfcheetah-medium, maze2d-umaze, antmaze-umaze-v0)")
    parser.add_argument("--data_dir", type=str, default="./d4rl_datasets",
                        help="Directory containing HDF5 dataset files")
    parser.add_argument("--data_size", type=int, default=200000,
                        help="Number of windows to sample for training (used in DT-style sampling)")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset if not found locally")
    
    # Window and Segmentation
    parser.add_argument("--window_size", "-T", type=int, default=50,
                        help="Context window size (timesteps)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for sliding window segmentation (auto-set for navigation tasks if not specified)")
    
    # Normalization
    parser.add_argument("--disable_normalization", action="store_true",
                        help="Disable observation/action normalization (not recommended)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of warmup epochs")
    parser.add_argument("--min_lr_ratio", type=float, default=0.3,
                        help="Minimum LR ratio for cosine annealing")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    
    # Model
    parser.add_argument("--n_layer", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128,
                        help="Embedding dimension")
    
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=50,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--cache_dir", type=str, default="./cache_sigdt",
                        help="Directory for caching preprocessed data")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Rebuild cache even if it exists")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, or auto")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


# ============================================================================
# GPT Model and Utilities
# ============================================================================

class CfgNode:
    """A lightweight configuration class inspired by yacs"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)


class NewGELU(nn.Module):
    """GELU activation function"""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with Flash Attention support."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = config.attn_pdrop

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Use Flash Attention on CUDA, manual attention on CPU
        if x.device.type == 'cuda':
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention for CPU compatibility
            hs = k.size(-1)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            if self.training and self.dropout_p > 0:
                att = F.dropout(att, p=self.dropout_p)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model"""

    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given

        if type_given:
            config.merge_from_dict({
                'openai-gpt': dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
                'gopher-44m': dict(n_layer=8, n_head=16, n_embd=512),
                'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro': dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano': dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("Number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


def build_model(vocab_size: int = 100, block_size: int = 512,
                n_layer: int = 4, n_head: int = 4, n_embd: int = 128) -> GPT:
    """Build a GPT model with the specified configuration."""
    config = GPT.get_default_config()
    config.model_type = None
    config.vocab_size = vocab_size
    config.block_size = block_size
    config.n_layer = n_layer
    config.n_head = n_head
    config.n_embd = n_embd
    return GPT(config)


# ============================================================================
# Token Types and Group Specifications
# ============================================================================

class Tok:
    GOAL = 0
    ACTION = 1
    OBS = 2
    INC = 3
    CROSS = 4
    RTG = 5  # Return-to-go token (for RTG mode)


@dataclass
class GroupSpec:
    name: str
    idx: List[int]

    @property
    def dim(self) -> int:
        return len(self.idx)


def get_state_groups(env_name: str, obs_dim: int = None) -> List[GroupSpec]:
    """
    Get state dimension groups for the given environment.
    
    Supported environments:
      - MuJoCo locomotion (halfcheetah, hopper, walker2d) with 17-dim state
      - Maze2D with 4-dim state (pos_x, pos_y, vel_x, vel_y)
      - AntMaze with 29-dim state
    """
    env_lower = env_name.lower()
    
    # MuJoCo locomotion tasks
    if any(x in env_lower for x in ['halfcheetah', 'hopper', 'walker2d', 'walker']):
        JA = list(range(2, 8))    # 6 dims - Joint Angles
        JV = list(range(11, 17))  # 6 dims - Joint Velocities  
        Body = [0, 1, 8, 9, 10]   # 5 dims - Body position/velocity
        return [
            GroupSpec("JA", JA),
            GroupSpec("JV", JV),
            GroupSpec("Body", Body),
        ]
    
    # Maze2D environments (4-dim state)
    elif 'maze2d' in env_lower:
        return [
            GroupSpec("pos", [0, 1]),    # Position (x, y)
            GroupSpec("vel", [2, 3]),    # Velocity (vx, vy)
        ]
    
    # AntMaze environments (29-dim state)
    elif 'antmaze' in env_lower:
        # For simplicity, use obs_dim-based grouping
        if obs_dim is None:
            obs_dim = 29
        # Split into body pose, joint angles, velocities
        return [
            GroupSpec("body", list(range(0, 7))),      # Body pose
            GroupSpec("joints", list(range(7, 15))),   # Joint angles
            GroupSpec("vel", list(range(15, min(29, obs_dim)))),  # Velocities
        ]
    
    else:
        raise ValueError(f"Unknown environment: {env_name}. Supported: halfcheetah, hopper, walker2d, maze2d, antmaze")


# ============================================================================
# Sequence Builder
# ============================================================================

@dataclass
class BuiltSeq:
    token_type: torch.LongTensor
    token_time: torch.LongTensor
    token_group: torch.LongTensor
    token_value: torch.FloatTensor
    pred_mask: torch.BoolTensor
    target_action: torch.FloatTensor


class SignatureSeqBuilder:
    """
    Signature-based sequence builder.
    
    Structure: GOAL + a_{t0-1} + (OBS, INC, CROSS)*T
    Predict a_t from the LAST token of each step block (CROSS).
    """
    
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        groups: List[GroupSpec],
        goal_dim: int = 1,
        include_self_term: bool = True,
    ):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.groups = groups
        self.goal_dim = goal_dim
        self.include_self_term = include_self_term

        self.group_inc_dims = [g.dim for g in groups]
        self.group_cross_dims = [g.dim * g.dim for g in groups]
        self.total_inc_dim = sum(self.group_inc_dims)
        self.total_cross_dim = sum(self.group_cross_dims)
        self.max_token_value_dim = max(
            goal_dim, act_dim, state_dim, self.total_inc_dim, self.total_cross_dim
        )

    def _cross_complete(self, s1_prev: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """
        Strict 2nd-level signature increment for piecewise linear paths:
            ΔS^(2) = S^(1)_{t-1} ⊗ Δx_t + 1/2 Δx_t ⊗ Δx_t
        """
        term_hist = np.outer(s1_prev, ds)
        if self.include_self_term:
            term_self = 0.5 * np.outer(ds, ds)
            M = term_hist + term_self
        else:
            M = term_hist
        return M.reshape(-1).astype(np.float32)

    def build_window(self, states, actions, goal, t0_action_prev=None):
        T = states.shape[0]
        assert states.shape[1] == self.state_dim
        assert actions.shape[1] == self.act_dim

        goal_vec = np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal_vec.shape[0] == 1 and self.goal_dim != 1:
            goal_vec = np.repeat(goal_vec, self.goal_dim).astype(np.float32)
        assert goal_vec.shape[0] == self.goal_dim

        if t0_action_prev is None:
            a_prev = np.zeros((self.act_dim,), dtype=np.float32)
        else:
            a_prev = np.asarray(t0_action_prev, dtype=np.float32).reshape(-1)
            assert a_prev.shape[0] == self.act_dim

        L = 2 + T * 3  # GOAL + ACTION + T * (OBS + INC + CROSS)

        token_type = np.zeros((L,), dtype=np.int64)
        token_time = np.zeros((L,), dtype=np.int64)
        token_group = np.full((L,), -1, dtype=np.int64)
        token_value = np.zeros((L, self.max_token_value_dim), dtype=np.float32)
        pred_mask = np.zeros((L,), dtype=np.bool_)
        target_action = np.zeros((T, self.act_dim), dtype=np.float32)

        s1 = [np.zeros((g.dim,), dtype=np.float32) for g in self.groups]
        ptr = 0

        # GOAL
        token_type[ptr] = Tok.GOAL
        token_value[ptr, :self.goal_dim] = goal_vec
        ptr += 1

        # Initial ACTION = a_{t0-1}
        token_type[ptr] = Tok.ACTION
        token_value[ptr, :self.act_dim] = a_prev
        ptr += 1

        prev_state = None
        for t in range(T):
            s_t = states[t].astype(np.float32)

            # OBS(s_t)
            token_type[ptr] = Tok.OBS
            token_time[ptr] = t
            token_value[ptr, :self.state_dim] = s_t
            ptr += 1

            # Compute Δs_t
            if t == 0:
                ds_full = np.zeros((self.state_dim,), dtype=np.float32)
            else:
                ds_full = (s_t - prev_state).astype(np.float32)

            # Compute and concatenate INC/CROSS from all groups
            inc_parts = []
            cross_parts = []
            for gi, g in enumerate(self.groups):
                idx = np.array(g.idx, dtype=np.int64)
                ds_g = ds_full[idx]
                inc_parts.append(ds_g)
                cross = self._cross_complete(s1[gi], ds_g)
                cross_parts.append(cross)
                s1[gi] = s1[gi] + ds_g

            # INC token
            inc_concat = np.concatenate(inc_parts)
            token_type[ptr] = Tok.INC
            token_time[ptr] = t
            token_value[ptr, :self.total_inc_dim] = inc_concat
            ptr += 1

            # CROSS token
            cross_concat = np.concatenate(cross_parts)
            token_type[ptr] = Tok.CROSS
            token_time[ptr] = t
            token_value[ptr, :self.total_cross_dim] = cross_concat
            ptr += 1

            # Predict action from CROSS token
            pred_mask[ptr - 1] = True
            target_action[t] = actions[t].astype(np.float32)
            prev_state = s_t

        return BuiltSeq(
            token_type=torch.from_numpy(token_type),
            token_time=torch.from_numpy(token_time),
            token_group=torch.from_numpy(token_group),
            token_value=torch.from_numpy(token_value),
            pred_mask=torch.from_numpy(pred_mask),
            target_action=torch.from_numpy(target_action),
        )

    def build_window_rtg(self, states, actions, rtg_sequence, t0_action_prev=None):
        """
        Build sequence with per-step RTG tokens instead of window-level GOAL.
        
        Structure: ACTION + T * (RTG + OBS + INC + CROSS)
        RTG mode is useful for navigation tasks like Maze2D where goal changes per step.
        """
        T = states.shape[0]
        assert states.shape[1] == self.state_dim
        assert actions.shape[1] == self.act_dim
        
        rtg_seq = np.asarray(rtg_sequence, dtype=np.float32).reshape(-1)
        assert len(rtg_seq) == T, f"RTG sequence length {len(rtg_seq)} != T {T}"
        
        if t0_action_prev is None:
            a_prev = np.zeros((self.act_dim,), dtype=np.float32)
        else:
            a_prev = np.asarray(t0_action_prev, dtype=np.float32).reshape(-1)
        
        G = len(self.groups)
        TOKENS_PER_STEP = 2 + 2 * G  # RTG + OBS + INC*G + CROSS*G (token per group)
        # Simplified: we concatenate groups into single INC/CROSS tokens
        TOKENS_PER_STEP_SIMPLE = 4  # RTG + OBS + INC + CROSS (concatenated)
        
        L = 1 + T * TOKENS_PER_STEP_SIMPLE  # ACTION + T * (RTG + OBS + INC + CROSS)
        
        token_type = np.zeros((L,), dtype=np.int64)
        token_time = np.zeros((L,), dtype=np.int64)
        token_group = np.full((L,), -1, dtype=np.int64)
        token_value = np.zeros((L, self.max_token_value_dim), dtype=np.float32)
        pred_mask = np.zeros((L,), dtype=np.bool_)
        target_action = np.zeros((T, self.act_dim), dtype=np.float32)
        
        s1 = [np.zeros((g.dim,), dtype=np.float32) for g in self.groups]
        ptr = 0
        
        # Initial ACTION = a_{t0-1}
        token_type[ptr] = Tok.ACTION
        token_value[ptr, :self.act_dim] = a_prev
        ptr += 1
        
        prev_state = None
        for t in range(T):
            s_t = states[t].astype(np.float32)
            
            # RTG token for this step
            token_type[ptr] = Tok.RTG
            token_time[ptr] = t
            token_value[ptr, 0] = rtg_seq[t]
            ptr += 1
            
            # OBS token
            token_type[ptr] = Tok.OBS
            token_time[ptr] = t
            token_value[ptr, :self.state_dim] = s_t
            ptr += 1
            
            # Compute Δs_t
            if t == 0:
                ds_full = np.zeros((self.state_dim,), dtype=np.float32)
            else:
                ds_full = (s_t - prev_state).astype(np.float32)
            
            # Compute INC/CROSS for all groups
            inc_parts = []
            cross_parts = []
            for gi, g in enumerate(self.groups):
                idx = np.array(g.idx, dtype=np.int64)
                ds_g = ds_full[idx]
                inc_parts.append(ds_g)
                cross = self._cross_complete(s1[gi], ds_g)
                cross_parts.append(cross)
                s1[gi] = s1[gi] + ds_g
            
            # INC token (concatenated)
            inc_concat = np.concatenate(inc_parts)
            token_type[ptr] = Tok.INC
            token_time[ptr] = t
            token_value[ptr, :self.total_inc_dim] = inc_concat
            ptr += 1
            
            # CROSS token (concatenated)
            cross_concat = np.concatenate(cross_parts)
            token_type[ptr] = Tok.CROSS
            token_time[ptr] = t
            token_value[ptr, :self.total_cross_dim] = cross_concat
            ptr += 1
            
            # Predict action from CROSS token
            pred_mask[ptr - 1] = True
            target_action[t] = actions[t].astype(np.float32)
            prev_state = s_t
        
        return BuiltSeq(
            token_type=torch.from_numpy(token_type),
            token_time=torch.from_numpy(token_time),
            token_group=torch.from_numpy(token_group),
            token_value=torch.from_numpy(token_value),
            pred_mask=torch.from_numpy(pred_mask),
            target_action=torch.from_numpy(target_action),
        )


# ============================================================================
# Token Embedding
# ============================================================================

class SignatureTokenEmbedding(nn.Module):
    def __init__(
        self,
        n_embd: int,
        state_dim: int,
        act_dim: int,
        goal_dim: int,
        total_inc_dim: int,
        total_cross_dim: int,
        max_value_dim: int,
        embd_pdrop: float = 0.1,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.goal_dim = goal_dim
        self.total_inc_dim = total_inc_dim
        self.total_cross_dim = total_cross_dim
        self.max_value_dim = max_value_dim

        self.goal_proj = nn.Linear(goal_dim, n_embd)
        self.action_proj = nn.Linear(act_dim, n_embd)
        self.obs_proj = nn.Linear(state_dim, n_embd)
        self.inc_proj = nn.Linear(total_inc_dim, n_embd)
        self.cross_proj = nn.Linear(total_cross_dim, n_embd)
        self.rtg_proj = nn.Linear(1, n_embd)  # RTG is 1-dimensional

        self.type_emb = nn.Embedding(6, n_embd)  # 6 token types: GOAL, ACTION, OBS, INC, CROSS, RTG
        self.drop = nn.Dropout(embd_pdrop)

    def forward(self, token_type, token_time, token_group, token_value):
        B, L = token_type.shape
        device = token_type.device
        x = torch.zeros((B, L, self.n_embd), device=device, dtype=token_value.dtype)
        x = x + self.type_emb(token_type.clamp(max=5))  # Clamp to valid range

        m = token_type == Tok.GOAL
        if m.any():
            x[m] = x[m] + self.goal_proj(token_value[m][:, :self.goal_dim])

        m = token_type == Tok.RTG
        if m.any():
            x[m] = x[m] + self.rtg_proj(token_value[m][:, :1])

        m = token_type == Tok.ACTION
        if m.any():
            x[m] = x[m] + self.action_proj(token_value[m][:, :self.act_dim])

        m = token_type == Tok.OBS
        if m.any():
            x[m] = x[m] + self.obs_proj(token_value[m][:, :self.state_dim])

        m = token_type == Tok.INC
        if m.any():
            x[m] = x[m] + self.inc_proj(token_value[m][:, :self.total_inc_dim])

        m = token_type == Tok.CROSS
        if m.any():
            x[m] = x[m] + self.cross_proj(token_value[m][:, :self.total_cross_dim])

        return self.drop(x)


# ============================================================================
# Data Loading and Normalization
# ============================================================================

D4RL_DATASET_URLS = {
    # MuJoCo locomotion
    'halfcheetah-medium-v2': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5',
    'halfcheetah-medium-expert-v2': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_expert-v2.hdf5',
    'halfcheetah-medium-replay-v2': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5',
    'hopper-medium-v2': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5',
    'hopper-medium-expert-v2': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_expert-v2.hdf5',
    'hopper-medium-replay-v2': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5',
    'walker2d-medium-v2': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5',
    'walker2d-medium-expert-v2': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_expert-v2.hdf5',
    'walker2d-medium-replay-v2': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5',
    # Maze2D
    'maze2d-umaze-v1': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5',
    'maze2d-medium-v1': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5',
    'maze2d-large-v1': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5',
    # AntMaze
    'antmaze-umaze-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
    'antmaze-medium-play-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'antmaze-large-play-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
}


def download_d4rl_dataset(env_name: str, data_dir: str) -> str:
    """Download D4RL dataset if not present locally."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Normalize environment name
    env_key = env_name.lower().replace('_', '-')
    if not env_key.endswith('-v0') and not env_key.endswith('-v1') and not env_key.endswith('-v2'):
        # Add version suffix based on environment type
        if 'maze2d' in env_key:
            env_key = env_key + '-v1'
        elif 'antmaze' in env_key:
            env_key = env_key + '-v0'
        else:
            env_key = env_key + '-v2'
    
    if env_key not in D4RL_DATASET_URLS:
        raise ValueError(f"Unknown environment for download: {env_name}. Available: {list(D4RL_DATASET_URLS.keys())}")
    
    url = D4RL_DATASET_URLS[env_key]
    filename = env_key.replace('-', '_') + '.hdf5'
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Dataset already exists: {filepath}")
        return filepath
    
    print(f"Downloading {env_key} from {url}...")
    print(f"Saving to: {filepath}")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Download complete!")
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")
    
    return filepath


def compute_normalization_stats(observations: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute mean and std for observations and actions."""
    obs_mean = observations.mean(axis=0).astype(np.float32)
    obs_std = observations.std(axis=0).astype(np.float32)
    obs_std = np.clip(obs_std, 1e-6, None)  # Avoid division by zero
    
    act_mean = actions.mean(axis=0).astype(np.float32)
    act_std = actions.std(axis=0).astype(np.float32)
    act_std = np.clip(act_std, 1e-6, None)
    
    return {
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'act_mean': act_mean,
        'act_std': act_std,
    }


def normalize_data(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize data using z-score."""
    return ((data - mean) / std).astype(np.float32)


def denormalize_data(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Denormalize data."""
    return (data * std + mean).astype(np.float32)


def load_d4rl_dataset_from_h5(filepath: str):
    """Load D4RL dataset from local HDF5 file."""
    assert os.path.exists(filepath), f"Dataset file not found: {filepath}"
    
    print(f"Loading dataset from {filepath}...")
    with h5py.File(filepath, "r") as f:
        observations = np.array(f["observations"])
        actions = np.array(f["actions"])
        rewards = np.array(f["rewards"]).reshape(-1)

        if "terminals" in f:
            terminals = np.array(f["terminals"]).reshape(-1).astype(np.bool_)
        elif "dones" in f:
            terminals = np.array(f["dones"]).reshape(-1).astype(np.bool_)
        else:
            raise KeyError("HDF5 must contain 'terminals' or 'dones'")

        if "timeouts" in f:
            timeouts = np.array(f["timeouts"]).reshape(-1).astype(np.bool_)
        else:
            timeouts = np.zeros_like(terminals, dtype=np.bool_)

    dones = np.logical_or(terminals, timeouts)

    dataset = {
        "observations": np.ascontiguousarray(observations.astype(np.float32)),
        "actions": np.ascontiguousarray(actions.astype(np.float32)),
        "rewards": np.ascontiguousarray(rewards.astype(np.float32)),
        "terminals": np.ascontiguousarray(dones.astype(np.bool_)),
    }

    print(f"  Observations: {dataset['observations'].shape}")
    print(f"  Actions: {dataset['actions'].shape}")
    print(f"  Rewards: {dataset['rewards'].shape}")
    print(f"  Terminal states: {dataset['terminals'].sum()}")

    return dataset


def extract_trajectories(dataset: dict, min_len: int = 1):
    """Extract full trajectories from flat dataset."""
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']

    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]

    terminal_indices = np.where(terminals)[0]
    trajectories = []
    start_idx = 0

    for end_idx in terminal_indices:
        ep_len = end_idx - start_idx + 1

        if ep_len >= min_len:
            traj = {
                'observations': observations[start_idx:end_idx + 1].astype(np.float32),
                'actions': actions[start_idx:end_idx + 1].astype(np.float32),
                'rewards': rewards[start_idx:end_idx + 1].astype(np.float32),
                'length': ep_len,
            }
            trajectories.append(traj)

        start_idx = end_idx + 1

    lengths = np.array([t['length'] for t in trajectories])
    print(f"Extracted {len(trajectories)} trajectories with length >= {min_len}")
    print(f"  Length range: [{lengths.min()}, {lengths.max()}], mean={lengths.mean():.1f}")
    print(f"  Total transitions: {lengths.sum()}")

    return trajectories, obs_dim, act_dim


def segment_into_trajectories_stride(
    dataset: dict,
    window_size: int,
    stride: int,
    discount: float = 1.0,
    rtg_scale: str = "episode",
) -> tuple:
    """
    Segment dataset into fixed-length windows using stride-based sliding windows.
    
    For RTG mode: each window includes return-to-go sequence computed from episode returns.
    
    Args:
        dataset: Raw dataset with observations, actions, rewards, terminals
        window_size: Length of each window
        stride: Number of steps to advance for each window
        discount: Discount factor for computing returns
        rtg_scale: How to scale RTG ("episode" normalizes by max episode return)
    
    Returns:
        trajectories_np: Array of shape (N, T, D) where D = 1 + obs_dim + act_dim
                         (RTG, obs, action) per timestep
        terminal_rtgs: Initial RTG for each trajectory
        obs_dim, act_dim: Dimensions
    """
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    
    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]
    
    # Find episode boundaries
    terminal_indices = np.where(terminals)[0]
    
    # Compute per-episode returns and RTGs
    episode_returns = []
    episode_rtgs = []  # RTG at each step within episode
    start_idx = 0
    
    for end_idx in terminal_indices:
        ep_rewards = rewards[start_idx:end_idx + 1]
        ep_len = len(ep_rewards)
        
        # Compute return-to-go for each step
        rtg = np.zeros(ep_len, dtype=np.float32)
        running_return = 0.0
        for i in reversed(range(ep_len)):
            running_return = ep_rewards[i] + discount * running_return
            rtg[i] = running_return
        
        episode_returns.append(rtg[0])  # Total episode return
        episode_rtgs.append(rtg)
        start_idx = end_idx + 1
    
    # Compute max return for normalization
    max_return = max(episode_returns) if episode_returns else 1.0
    if rtg_scale == "episode" and max_return > 0:
        scale = max_return
    else:
        scale = 1.0
    
    print(f"Episode returns: min={min(episode_returns):.2f}, max={max_return:.2f}, mean={np.mean(episode_returns):.2f}")
    print(f"RTG scale: {scale:.2f}")
    
    # Build windows with sliding stride
    windows = []
    terminal_rtgs = []
    
    start_idx = 0
    for ep_idx, end_idx in enumerate(terminal_indices):
        ep_obs = observations[start_idx:end_idx + 1]
        ep_act = actions[start_idx:end_idx + 1]
        ep_rtg = episode_rtgs[ep_idx] / scale  # Normalized RTG
        ep_len = len(ep_obs)
        
        # Generate windows with stride
        for win_start in range(0, max(1, ep_len - window_size + 1), stride):
            win_end = win_start + window_size
            
            if win_end <= ep_len:
                # Full window within episode
                win_obs = ep_obs[win_start:win_end]
                win_act = ep_act[win_start:win_end]
                win_rtg = ep_rtg[win_start:win_end]
            else:
                # Partial window - pad with last state (for short episodes)
                actual_len = ep_len - win_start
                win_obs = np.zeros((window_size, obs_dim), dtype=np.float32)
                win_act = np.zeros((window_size, act_dim), dtype=np.float32)
                win_rtg = np.zeros(window_size, dtype=np.float32)
                
                win_obs[:actual_len] = ep_obs[win_start:]
                win_act[:actual_len] = ep_act[win_start:]
                win_rtg[:actual_len] = ep_rtg[win_start:]
                
                # Pad with last values
                if actual_len > 0:
                    win_obs[actual_len:] = ep_obs[-1]
                    win_act[actual_len:] = ep_act[-1]
                    win_rtg[actual_len:] = ep_rtg[-1]
            
            # Stack: RTG (1) + obs (obs_dim) + act (act_dim)
            window = np.concatenate([
                win_rtg.reshape(-1, 1),
                win_obs,
                win_act
            ], axis=1)
            
            windows.append(window)
            terminal_rtgs.append(win_rtg[0])
        
        start_idx = end_idx + 1
    
    trajectories_np = np.array(windows, dtype=np.float32)
    terminal_rtgs = np.array(terminal_rtgs, dtype=np.float32)
    
    print(f"Created {len(windows)} windows with stride={stride}, window_size={window_size}")
    
    return trajectories_np, terminal_rtgs, obs_dim, act_dim


# ============================================================================
# Cache Building
# ============================================================================

def build_cache_dt_style(
    trajectories: list,
    sampling_weights: np.ndarray,
    builder: SignatureSeqBuilder,
    data_size: int,
    T: int,
    obs_dim: int,
    act_dim: int,
    seq_len: int,
    seed: int = 42,
):
    """Build cache using DT-style sampling (proportional to trajectory length)."""
    L = seq_len
    Dmax = builder.max_token_value_dim

    token_type = torch.empty((data_size, L), dtype=torch.long)
    token_time = torch.empty((data_size, L), dtype=torch.long)
    token_group = torch.empty((data_size, L), dtype=torch.long)
    token_value = torch.empty((data_size, L, Dmax), dtype=torch.float32)
    pred_mask = torch.empty((data_size, L), dtype=torch.bool)
    target_act = torch.empty((data_size, T, act_dim), dtype=torch.float32)

    num_trajs = len(trajectories)
    traj_indices = np.arange(num_trajs)
    rng = np.random.default_rng(seed=seed)

    i = 0
    pbar = tqdm.tqdm(total=data_size, desc="Building cache (DT-style)", leave=True)

    while i < data_size:
        traj_idx = rng.choice(traj_indices, p=sampling_weights)
        traj = trajectories[traj_idx]
        traj_len = traj['length']

        max_start = traj_len - T
        if max_start < 0:
            continue

        start = rng.integers(0, max_start + 1)
        end = start + T

        states = traj['observations'][start:end]
        actions = traj['actions'][start:end]
        window_rewards = traj['rewards'][start:end]

        goal = np.array([float(window_rewards.sum())], dtype=np.float32)
        a_prev = traj['actions'][start - 1] if start > 0 else None

        built = builder.build_window(
            states=states,
            actions=actions,
            goal=goal,
            t0_action_prev=a_prev,
        )

        token_type[i].copy_(built.token_type)
        token_time[i].copy_(built.token_time)
        token_group[i].copy_(built.token_group)
        token_value[i].copy_(built.token_value)
        pred_mask[i].copy_(built.pred_mask)
        target_act[i].copy_(built.target_action)

        i += 1
        pbar.update(1)

    pbar.close()

    return {
        "token_type": token_type,
        "token_time": token_time,
        "token_group": token_group,
        "token_value": token_value,
        "pred_mask": pred_mask,
        "target_action": target_act,
        "meta": {
            "N": data_size, "L": L, "Dmax": Dmax, "T": T,
            "act_dim": act_dim, "obs_dim": obs_dim,
        }
    }


class CachedSeqDataset(Dataset):
    def __init__(self, cache_dict):
        self.ct = cache_dict["token_type"]
        self.ctm = cache_dict["token_time"]
        self.cg = cache_dict["token_group"]
        self.cv = cache_dict["token_value"]
        self.pm = cache_dict["pred_mask"]
        self.ta = cache_dict["target_action"]

    def __len__(self):
        return self.ct.shape[0]

    def __getitem__(self, i):
        return {
            "token_type": self.ct[i],
            "token_time": self.ctm[i],
            "token_group": self.cg[i],
            "token_value": self.cv[i],
            "pred_mask": self.pm[i],
            "target_action": self.ta[i],
        }


def collate_fixed(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


def build_cache_rtg_mode(
    trajectories_np: np.ndarray,
    builder: SignatureSeqBuilder,
    T: int,
    act_dim: int,
    obs_dim: int,
):
    """Build cache for RTG mode (stride-based windows with per-step RTG)."""
    N = trajectories_np.shape[0]
    
    # RTG mode: each step has RTG + OBS + INC + CROSS = 4 tokens
    TOKENS_PER_STEP = 4
    INITIAL_TOKENS_COUNT = 1  # Only initial ACTION
    SEQ_LEN = INITIAL_TOKENS_COUNT + T * TOKENS_PER_STEP
    
    L = SEQ_LEN
    Dmax = builder.max_token_value_dim

    token_type = torch.empty((N, L), dtype=torch.long)
    token_time = torch.empty((N, L), dtype=torch.long)
    token_group = torch.empty((N, L), dtype=torch.long)
    token_value = torch.empty((N, L, Dmax), dtype=torch.float32)
    pred_mask = torch.empty((N, L), dtype=torch.bool)
    target_act = torch.empty((N, T, act_dim), dtype=torch.float32)

    # Trajectories_np has shape (N, T, 1 + obs_dim + act_dim) = (RTG, obs, action)
    RTG_COL = 0
    STATE_SL = slice(1, 1 + obs_dim)
    ACT_SL = slice(1 + obs_dim, 1 + obs_dim + act_dim)

    for i in tqdm.tqdm(range(N), desc="Building cache (RTG mode)", leave=True):
        w = trajectories_np[i]
        rtg_sequence = w[:, RTG_COL].astype(np.float32)
        states = w[:, STATE_SL].astype(np.float32)
        actions = w[:, ACT_SL].astype(np.float32)

        built = builder.build_window_rtg(
            states=states,
            actions=actions,
            rtg_sequence=rtg_sequence,
            t0_action_prev=None,
        )

        token_type[i].copy_(built.token_type)
        token_time[i].copy_(built.token_time)
        token_group[i].copy_(built.token_group)
        token_value[i].copy_(built.token_value)
        pred_mask[i].copy_(built.pred_mask)
        target_act[i].copy_(built.target_action)

    return {
        "token_type": token_type,
        "token_time": token_time,
        "token_group": token_group,
        "token_value": token_value,
        "pred_mask": pred_mask,
        "target_action": target_act,
        "meta": {
            "N": N, "L": L, "Dmax": Dmax, "T": T,
            "act_dim": act_dim, "obs_dim": obs_dim,
            "mode": "rtg",
        }
    }


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    args = parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Memory optimization for CUDA
    if device == "cuda":
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()

    # Determine mode based on environment type
    # Navigation tasks (maze2d, antmaze) → RTG mode with per-step RTG tokens
    # Locomotion tasks (halfcheetah, hopper, walker2d) → GOAL mode with window-level goal
    env_lower = args.env.lower()
    is_navigation_task = 'maze2d' in env_lower or 'antmaze' in env_lower
    is_locomotion_task = any(x in env_lower for x in ['halfcheetah', 'hopper', 'walker2d', 'walker'])
    
    if is_navigation_task:
        use_rtg_mode = True
        # Auto-set stride for navigation if not specified
        if args.stride is None:
            args.stride = 10
            print(f"Auto-setting stride={args.stride} for navigation task")
    else:
        use_rtg_mode = False
    
    use_stride = args.stride is not None
    
    # ========================================================================
    # Load Dataset
    # ========================================================================
    print("=" * 60)
    print(f"Loading D4RL dataset: {args.env}")
    print("=" * 60)

    # Try to find the file with various naming conventions
    env_normalized = args.env.lower().replace('-', '_')
    possible_filenames = [
        f"{args.env.replace('-', '_')}-v2.hdf5",
        f"{args.env.replace('-', '_')}_v2.hdf5",
        f"{args.env.replace('_', '-')}-v2.hdf5",
        f"{env_normalized}-v1.hdf5",
        f"{env_normalized}_v1.hdf5",
        f"{args.env}.hdf5",
    ]
    
    filepath = None
    for filename in possible_filenames:
        candidate = os.path.join(args.data_dir, filename)
        if os.path.exists(candidate):
            filepath = candidate
            break
    
    # Download if not found and --download is specified
    if filepath is None:
        if args.download:
            print("Dataset not found locally. Downloading...")
            filepath = download_d4rl_dataset(args.env, args.data_dir)
        else:
            raise FileNotFoundError(
                f"Dataset not found in {args.data_dir}. "
                f"Tried: {possible_filenames}. "
                f"Use --download to download automatically."
            )
    
    raw = load_d4rl_dataset_from_h5(filepath)
    obs_dim = raw['observations'].shape[1]
    act_dim = raw['actions'].shape[1]
    
    # ========================================================================
    # Normalization
    # ========================================================================
    norm_stats = None
    if not args.disable_normalization:
        print("\nComputing normalization statistics...")
        norm_stats = compute_normalization_stats(raw['observations'], raw['actions'])
        
        print("Applying normalization...")
        raw['observations'] = normalize_data(
            raw['observations'], norm_stats['obs_mean'], norm_stats['obs_std']
        )
        raw['actions'] = normalize_data(
            raw['actions'], norm_stats['act_mean'], norm_stats['act_std']
        )
        print(f"  obs_mean: {norm_stats['obs_mean'][:3]}... (first 3 dims)")
        print(f"  obs_std:  {norm_stats['obs_std'][:3]}... (first 3 dims)")
    else:
        print("\nNormalization disabled.")

    # ========================================================================
    # Setup Groups and Builder
    # ========================================================================
    groups = get_state_groups(args.env, obs_dim)
    print(f"\nState groups for {args.env}:")
    for g in groups:
        print(f"  {g.name}: indices {g.idx[:5]}{'...' if len(g.idx) > 5 else ''} (dim={g.dim})")
    
    builder = SignatureSeqBuilder(
        state_dim=obs_dim,
        act_dim=act_dim,
        groups=groups,
        goal_dim=1,
    )

    T = args.window_size
    
    # ========================================================================
    # Extract Data (mode-dependent)
    # ========================================================================
    if use_stride:
        stride = args.stride
        print(f"\nUsing stride-based segmentation (stride={stride}, RTG mode)")
        
        trajectories_np, terminal_rtgs, obs_dim, act_dim = segment_into_trajectories_stride(
            dataset=raw,
            window_size=T,
            stride=stride,
            discount=1.0,
            rtg_scale="episode",
        )
        
        # RTG mode sequence length: ACTION + T * (RTG + OBS + INC + CROSS)
        TOKENS_PER_STEP = 4
        INITIAL_TOKENS_COUNT = 1
        SEQ_LEN = INITIAL_TOKENS_COUNT + T * TOKENS_PER_STEP
        data_size = len(trajectories_np)
        
    else:
        # DT-style sampling with window-level GOAL
        print(f"\nUsing DT-style sampling (data_size={args.data_size})")
        
        trajectories, obs_dim, act_dim = extract_trajectories(
            dataset=raw,
            min_len=T,
        )
        
        # Compute sampling weights
        traj_lengths = np.array([t['length'] for t in trajectories])
        valid_starts = np.maximum(traj_lengths - T + 1, 1)
        sampling_weights = valid_starts / valid_starts.sum()
        
        print(f"Total valid (traj, start) pairs: {valid_starts.sum()}")
        
        # GOAL mode sequence length: GOAL + ACTION + T * (OBS + INC + CROSS)
        TOKENS_PER_STEP = 3
        INITIAL_TOKENS_COUNT = 2
        SEQ_LEN = INITIAL_TOKENS_COUNT + T * TOKENS_PER_STEP
        data_size = args.data_size
        stride = None

    MODEL_BLOCK_SIZE = 512
    BLOCK_SIZE = max(SEQ_LEN, MODEL_BLOCK_SIZE)

    print(f"\nSequence length: {SEQ_LEN} (init={INITIAL_TOKENS_COUNT}, per_step={TOKENS_PER_STEP}, T={T})")
    print(f"Transformer block_size: {BLOCK_SIZE}")
    print(f"Total INC dim: {builder.total_inc_dim}")
    print(f"Total CROSS dim: {builder.total_cross_dim}")

    # ========================================================================
    # Build or Load Cache
    # ========================================================================
    os.makedirs(args.cache_dir, exist_ok=True)
    mode_str = "rtg" if use_stride else "goal"
    stride_str = f"_stride{stride}" if stride else ""
    cache_tag = f"{args.env}_T{T}_N{data_size}{stride_str}_{mode_str}_seq{SEQ_LEN}_Dmax{builder.max_token_value_dim}"
    cache_path = os.path.join(args.cache_dir, f"cache_{cache_tag}.pt")

    cache = None
    if os.path.exists(cache_path) and not args.rebuild_cache:
        try:
            cache = torch.load(cache_path, map_location="cpu")
            assert cache["token_type"].shape[1] == SEQ_LEN
            print(f"Loaded cache: {cache_path}")
        except Exception as e:
            print(f"Cache load failed, rebuilding. Reason: {e}")
            cache = None

    if cache is None:
        if use_stride:
            cache = build_cache_rtg_mode(
                trajectories_np=trajectories_np,
                builder=builder,
                T=T,
                act_dim=act_dim,
                obs_dim=obs_dim,
            )
        else:
            cache = build_cache_dt_style(
                trajectories=trajectories,
                sampling_weights=sampling_weights,
                builder=builder,
                data_size=data_size,
                T=T,
                obs_dim=obs_dim,
                act_dim=act_dim,
                seq_len=SEQ_LEN,
                seed=args.seed,
            )
        torch.save(cache, cache_path)
        print(f"Saved cache: {cache_path}")

    dataset = CachedSeqDataset(cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=(device == "cuda"),
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
        collate_fn=collate_fixed,
    )

    print(f"Dataset ready: N={len(dataset)}, SEQ_LEN={SEQ_LEN}")

    # ========================================================================
    # Build Model
    # ========================================================================
    model = build_model(
        vocab_size=1, 
        block_size=BLOCK_SIZE, 
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd
    ).to(device)

    embed = SignatureTokenEmbedding(
        n_embd=args.n_embd,
        state_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=1,
        total_inc_dim=builder.total_inc_dim,
        total_cross_dim=builder.total_cross_dim,
        max_value_dim=builder.max_token_value_dim,
    ).to(device)

    action_head = nn.Linear(args.n_embd, act_dim).to(device)

    params = list(model.parameters()) + list(embed.parameters()) + list(action_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch: int):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, (args.epochs - args.warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    os.makedirs(args.save_dir, exist_ok=True)
    losses = []
    start_time = time.time()

    for epoch in tqdm.tqdm(range(args.epochs), desc="Epochs", position=0):
        model.train()
        embed.train()
        action_head.train()

        epoch_loss = 0.0
        n_batches = 0

        batch_pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}", position=1, leave=False)
        for batch in batch_pbar:
            token_type = batch["token_type"].to(device)
            token_time = batch["token_time"].to(device)
            token_group = batch["token_group"].to(device)
            token_value = batch["token_value"].to(device)
            pred_mask = batch["pred_mask"].to(device)
            target_action = batch["target_action"].to(device)

            B, L = token_type.shape

            x = embed(token_type, token_time, token_group, token_value)
            pos = torch.arange(0, L, device=device, dtype=torch.long).unsqueeze(0)
            x = model.transformer.drop(x + model.transformer.wpe(pos))

            for block in model.transformer.h:
                x = block(x)
            x = model.transformer.ln_f(x)

            x_sel = x[pred_mask]
            pred = action_head(x_sel).view(B, T, act_dim)

            loss = F.mse_loss(pred, target_action)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            batch_pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:03d}/{args.epochs} | loss={avg_loss:.6f} | lr={lr_now:.2e} | elapsed={elapsed/60:.1f}m")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            run_name = f"SIGDT_{args.env}_T{T}_EPOCH{epoch}_{mode_str}_{int(time.time())}"
            ckpt_path = os.path.join(args.save_dir, f"{run_name}.pt")
            meta_path = os.path.join(args.save_dir, f"{run_name}.json")

            groups_meta = [{"name": g.name, "idx": list(map(int, g.idx)), "dim": int(g.dim)} for g in groups]

            builder_meta = {
                "class": type(builder).__name__,
                "state_dim": int(builder.state_dim),
                "act_dim": int(builder.act_dim),
                "goal_dim": int(builder.goal_dim),
                "include_self_term": bool(builder.include_self_term),
                "group_inc_dims": list(map(int, builder.group_inc_dims)),
                "group_cross_dims": list(map(int, builder.group_cross_dims)),
                "max_token_value_dim": int(builder.max_token_value_dim),
            }

            train_meta = {
                "env_name": args.env,
                "data_source": args.env,
                "obs_dim": int(obs_dim),
                "act_dim": int(act_dim),
                "T_window": int(T),
                "stride": stride,
                "mode": mode_str,
                "data_size": int(len(dataset)),
                "TOKENS_PER_STEP": int(TOKENS_PER_STEP),
                "INITIAL_TOKENS_COUNT": int(INITIAL_TOKENS_COUNT),
                "SEQ_LEN": int(SEQ_LEN),
                "BLOCK_SIZE": int(BLOCK_SIZE),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "grad_clip": float(args.grad_clip),
                "warmup_epochs": int(args.warmup_epochs),
                "min_lr_ratio": float(args.min_lr_ratio),
                "normalized": not args.disable_normalization,
                "groups": groups_meta,
                "builder": builder_meta,
                "transformer": {
                    "n_layer": int(args.n_layer),
                    "n_head": int(args.n_head),
                    "n_embd": int(args.n_embd),
                    "block_size": int(BLOCK_SIZE),
                },
            }

            ckpt = {
                "train_meta": train_meta,
                "normalization": norm_stats,
                "gpt_trunk_state_dict": model.state_dict(),
                "embed_state_dict": embed.state_dict(),
                "action_head_state_dict": action_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "losses": losses,
            }

            torch.save(ckpt, ckpt_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(train_meta, f, indent=2, ensure_ascii=False)

            print(f"\nSaved checkpoint: {ckpt_path}")
            if norm_stats is not None:
                print("  (includes normalization statistics)")

    print("\nTraining complete.")
    print(f"Final loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
