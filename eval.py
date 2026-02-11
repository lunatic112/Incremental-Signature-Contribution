#!/usr/bin/env python3
"""
Signature Decision Transformer Evaluation Script

Release-ready evaluation script for Signature-based Decision Transformer on D4RL environments.
Loads a checkpoint and runs online rollouts to compute normalized D4RL scores.

Supports:
  - MuJoCo locomotion (HalfCheetah, Hopper, Walker2d)
  - Maze2D environments with trajectory visualization
  - AntMaze environments
  - Both GOAL and RTG modes

Usage:
    python eval.py --checkpoint ./checkpoints/model.pt --goal 300 --episodes 10
    python eval.py --checkpoint ./checkpoints/maze_model.pt --episodes 5 --plot

Requirements:
    pip install torch numpy gym d4rl mujoco-py iisignature matplotlib
"""

import os
import sys
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional imports
try:
    import iisignature
    HAS_IISIGNATURE = True
except ImportError:
    HAS_IISIGNATURE = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# Configuration and Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Signature Decision Transformer on D4RL environments (MuJoCo, Maze2D, AntMaze)"
    )
    
    # Checkpoint
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to checkpoint .pt file")
    
    # Environment override
    parser.add_argument("--env", type=str, default=None,
                        help="Override environment (e.g., halfcheetah-medium-v2). If None, uses checkpoint metadata.")
    
    # Evaluation settings
    parser.add_argument("--goal", type=float, default=None,
                        help="Target return (RTG-like scalar). If None, uses default based on environment.")
    parser.add_argument("--episodes", "-n", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--render", action="store_true",
                        help="Render environment (slower)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate trajectory visualization (Maze2D only)")
    
    # Model settings
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic action (no sampling)")
    parser.add_argument("--action_clip", type=float, default=1.0,
                        help="Clip actions to [-clip, clip]")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, or auto")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU even if CUDA available")
    
    # MuJoCo path (Windows)
    parser.add_argument("--mujoco_path", type=str, default=None,
                        help="Path to MuJoCo bin directory (Windows)")
    
    return parser.parse_args()


# ============================================================================
# GPT Model and Utilities
# ============================================================================

class CfgNode:
    """A lightweight configuration class"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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
        assert t <= self.block_size
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


@dataclass
class BuiltSeq:
    token_type: torch.LongTensor
    token_time: torch.LongTensor
    token_group: torch.LongTensor
    token_value: torch.FloatTensor
    pred_mask: torch.BoolTensor
    target_action: torch.FloatTensor


def get_default_groups() -> List[GroupSpec]:
    """Default groups for HalfCheetah/Walker2d/Hopper (17 dim state)."""
    JA = list(range(2, 8))
    JV = list(range(11, 17))
    Body = [0, 1, 8, 9, 10]
    return [
        GroupSpec("JA", JA),
        GroupSpec("JV", JV),
        GroupSpec("Body", Body),
    ]


# ============================================================================
# Sequence Builders (Baseline + Ablations)
# ============================================================================

class SignatureSeqBuilder:
    """Baseline: Original signature-based sequence builder."""
    
    def __init__(self, state_dim: int, act_dim: int, groups: List[GroupSpec],
                 goal_dim: int = 1, include_self_term: bool = True):
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
        term_hist = np.outer(s1_prev, ds)
        if self.include_self_term:
            term_self = 0.5 * np.outer(ds, ds)
            M = term_hist + term_self
        else:
            M = term_hist
        return M.reshape(-1).astype(np.float32)

    def build_window(self, states, actions, goal, t0_action_prev=None):
        T = states.shape[0]
        goal_vec = np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal_vec.shape[0] == 1 and self.goal_dim != 1:
            goal_vec = np.repeat(goal_vec, self.goal_dim).astype(np.float32)
        a_prev = np.zeros((self.act_dim,), dtype=np.float32) if t0_action_prev is None else np.asarray(t0_action_prev, dtype=np.float32).reshape(-1)

        L = 2 + T * 3
        token_type = np.zeros((L,), dtype=np.int64)
        token_time = np.zeros((L,), dtype=np.int64)
        token_group = np.full((L,), -1, dtype=np.int64)
        token_value = np.zeros((L, self.max_token_value_dim), dtype=np.float32)
        pred_mask = np.zeros((L,), dtype=np.bool_)
        target_action = np.zeros((T, self.act_dim), dtype=np.float32)
        s1 = [np.zeros((g.dim,), dtype=np.float32) for g in self.groups]

        ptr = 0
        token_type[ptr] = Tok.GOAL
        token_value[ptr, :self.goal_dim] = goal_vec
        ptr += 1
        token_type[ptr] = Tok.ACTION
        token_value[ptr, :self.act_dim] = a_prev
        ptr += 1

        prev_state = None
        for t in range(T):
            s_t = states[t].astype(np.float32)
            token_type[ptr] = Tok.OBS
            token_time[ptr] = t
            token_value[ptr, :self.state_dim] = s_t
            ptr += 1

            ds_full = np.zeros((self.state_dim,), dtype=np.float32) if t == 0 else (s_t - prev_state).astype(np.float32)
            inc_parts, cross_parts = [], []
            for gi, g in enumerate(self.groups):
                idx = np.array(g.idx, dtype=np.int64)
                ds_g = ds_full[idx]
                inc_parts.append(ds_g)
                cross_parts.append(self._cross_complete(s1[gi], ds_g))
                s1[gi] = s1[gi] + ds_g

            token_type[ptr] = Tok.INC
            token_time[ptr] = t
            token_value[ptr, :self.total_inc_dim] = np.concatenate(inc_parts)
            ptr += 1
            token_type[ptr] = Tok.CROSS
            token_time[ptr] = t
            token_value[ptr, :self.total_cross_dim] = np.concatenate(cross_parts)
            ptr += 1
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

    def build_window_seq(self, states, actions, goal, t0_action_prev=None):
        built = self.build_window(states, actions, goal, t0_action_prev)
        return (built.token_type.numpy(), built.token_time.numpy(), built.token_group.numpy(),
                built.token_value.numpy(), built.pred_mask.numpy())


class FixedAnchorObsSeqBuilder:
    """ABLATION 1: All OBS tokens use the first observation in window."""
    
    def __init__(self, state_dim: int, act_dim: int, groups: List[GroupSpec],
                 goal_dim: int = 1, include_self_term: bool = True):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.groups = groups
        self.goal_dim = goal_dim
        self.include_self_term = include_self_term
        self.group_inc_dims = [g.dim for g in groups]
        self.group_cross_dims = [g.dim * g.dim for g in groups]
        self.total_inc_dim = sum(self.group_inc_dims)
        self.total_cross_dim = sum(self.group_cross_dims)
        self.max_token_value_dim = max(goal_dim, act_dim, state_dim,
                                        self.total_inc_dim, self.total_cross_dim)

    def _cross_complete(self, s1_prev: np.ndarray, ds: np.ndarray) -> np.ndarray:
        term_hist = np.outer(s1_prev, ds)
        if self.include_self_term:
            term_self = 0.5 * np.outer(ds, ds)
            M = term_hist + term_self
        else:
            M = term_hist
        return M.reshape(-1).astype(np.float32)

    def build_window(self, states, actions, goal, t0_action_prev=None):
        T = states.shape[0]
        goal_vec = np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal_vec.shape[0] == 1 and self.goal_dim != 1:
            goal_vec = np.repeat(goal_vec, self.goal_dim).astype(np.float32)
        a_prev = np.zeros((self.act_dim,), dtype=np.float32) if t0_action_prev is None else np.asarray(t0_action_prev, dtype=np.float32).reshape(-1)

        L = 2 + T * 3
        token_type = np.zeros((L,), dtype=np.int64)
        token_time = np.zeros((L,), dtype=np.int64)
        token_group = np.full((L,), -1, dtype=np.int64)
        token_value = np.zeros((L, self.max_token_value_dim), dtype=np.float32)
        pred_mask = np.zeros((L,), dtype=np.bool_)
        target_action = np.zeros((T, self.act_dim), dtype=np.float32)
        s1 = [np.zeros((g.dim,), dtype=np.float32) for g in self.groups]

        ptr = 0
        token_type[ptr] = Tok.GOAL
        token_value[ptr, :self.goal_dim] = goal_vec
        ptr += 1
        token_type[ptr] = Tok.ACTION
        token_value[ptr, :self.act_dim] = a_prev
        ptr += 1

        anchor_obs = states[0].astype(np.float32)  # Fixed anchor
        prev_state = None

        for t in range(T):
            s_t = states[t].astype(np.float32)
            token_type[ptr] = Tok.OBS
            token_time[ptr] = t
            token_value[ptr, :self.state_dim] = anchor_obs  # Use anchor!
            ptr += 1

            ds_full = np.zeros((self.state_dim,), dtype=np.float32) if t == 0 else (s_t - prev_state).astype(np.float32)
            inc_parts, cross_parts = [], []
            for gi, g in enumerate(self.groups):
                idx = np.array(g.idx, dtype=np.int64)
                ds_g = ds_full[idx]
                inc_parts.append(ds_g)
                cross_parts.append(self._cross_complete(s1[gi], ds_g))
                s1[gi] = s1[gi] + ds_g

            token_type[ptr] = Tok.INC
            token_time[ptr] = t
            token_value[ptr, :self.total_inc_dim] = np.concatenate(inc_parts)
            ptr += 1
            token_type[ptr] = Tok.CROSS
            token_time[ptr] = t
            token_value[ptr, :self.total_cross_dim] = np.concatenate(cross_parts)
            ptr += 1
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

    def build_window_seq(self, states, actions, goal, t0_action_prev=None):
        built = self.build_window(states, actions, goal, t0_action_prev)
        return (built.token_type.numpy(), built.token_time.numpy(), built.token_group.numpy(),
                built.token_value.numpy(), built.pred_mask.numpy())


class CorrelationTokenSeqBuilder:
    """ABLATION 2: Signature tokens replaced with correlation-based features."""
    
    def __init__(self, state_dim: int, act_dim: int, groups: List[GroupSpec],
                 goal_dim: int = 1, include_self_term: bool = True):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.groups = groups
        self.goal_dim = goal_dim
        self.include_self_term = include_self_term
        self.group_inc_dims = [g.dim for g in groups]
        self.group_cross_dims = [g.dim * g.dim for g in groups]
        self.total_inc_dim = sum(self.group_inc_dims)
        self.total_cross_dim = sum(self.group_cross_dims)
        self.max_token_value_dim = max(goal_dim, act_dim, state_dim,
                                        self.total_inc_dim, self.total_cross_dim)

    def _correlation_inc(self, history: np.ndarray, current: np.ndarray) -> np.ndarray:
        if len(history) > 1:
            mean_state = np.mean(history, axis=0)
            std_state = np.std(history, axis=0) + 1e-8
            return ((current - mean_state) / std_state).astype(np.float32)
        return np.zeros_like(current, dtype=np.float32)

    def _correlation_cross(self, history: np.ndarray, current: np.ndarray) -> np.ndarray:
        if len(history) > 1:
            mean_state = np.mean(history, axis=0)
            std_state = np.std(history, axis=0) + 1e-8
            centered = (current - mean_state) / std_state
        else:
            centered = current
        return np.outer(centered, centered).reshape(-1).astype(np.float32)

    def build_window(self, states, actions, goal, t0_action_prev=None):
        T = states.shape[0]
        goal_vec = np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal_vec.shape[0] == 1 and self.goal_dim != 1:
            goal_vec = np.repeat(goal_vec, self.goal_dim).astype(np.float32)
        a_prev = np.zeros((self.act_dim,), dtype=np.float32) if t0_action_prev is None else np.asarray(t0_action_prev, dtype=np.float32).reshape(-1)

        L = 2 + T * 3
        token_type = np.zeros((L,), dtype=np.int64)
        token_time = np.zeros((L,), dtype=np.int64)
        token_group = np.full((L,), -1, dtype=np.int64)
        token_value = np.zeros((L, self.max_token_value_dim), dtype=np.float32)
        pred_mask = np.zeros((L,), dtype=np.bool_)
        target_action = np.zeros((T, self.act_dim), dtype=np.float32)

        ptr = 0
        token_type[ptr] = Tok.GOAL
        token_value[ptr, :self.goal_dim] = goal_vec
        ptr += 1
        token_type[ptr] = Tok.ACTION
        token_value[ptr, :self.act_dim] = a_prev
        ptr += 1

        state_history = []
        for t in range(T):
            s_t = states[t].astype(np.float32)
            state_history.append(s_t)

            token_type[ptr] = Tok.OBS
            token_time[ptr] = t
            token_value[ptr, :self.state_dim] = s_t
            ptr += 1

            inc_parts, cross_parts = [], []
            for g in self.groups:
                idx = np.array(g.idx, dtype=np.int64)
                group_hist = [s[idx] for s in state_history]
                current_g = s_t[idx]
                inc_parts.append(self._correlation_inc(np.array(group_hist), current_g))
                cross_parts.append(self._correlation_cross(np.array(group_hist), current_g))

            token_type[ptr] = Tok.INC
            token_time[ptr] = t
            token_value[ptr, :self.total_inc_dim] = np.concatenate(inc_parts)
            ptr += 1
            token_type[ptr] = Tok.CROSS
            token_time[ptr] = t
            token_value[ptr, :self.total_cross_dim] = np.concatenate(cross_parts)
            ptr += 1
            pred_mask[ptr - 1] = True
            target_action[t] = actions[t].astype(np.float32)

        return BuiltSeq(
            token_type=torch.from_numpy(token_type),
            token_time=torch.from_numpy(token_time),
            token_group=torch.from_numpy(token_group),
            token_value=torch.from_numpy(token_value),
            pred_mask=torch.from_numpy(pred_mask),
            target_action=torch.from_numpy(target_action),
        )

    def build_window_seq(self, states, actions, goal, t0_action_prev=None):
        built = self.build_window(states, actions, goal, t0_action_prev)
        return (built.token_type.numpy(), built.token_time.numpy(), built.token_group.numpy(),
                built.token_value.numpy(), built.pred_mask.numpy())


class FullSignatureSeqBuilder:
    """ABLATION 3: Full signature from window start at each timestep."""
    
    def __init__(self, state_dim: int, act_dim: int, groups: List[GroupSpec],
                 goal_dim: int = 1, include_self_term: bool = True):
        if not HAS_IISIGNATURE:
            raise ImportError("iisignature required for FullSignatureSeqBuilder. Install with: pip install iisignature")
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.groups = groups
        self.goal_dim = goal_dim
        self.group_inc_dims = [g.dim for g in groups]
        self.group_cross_dims = [g.dim * g.dim for g in groups]
        self.total_inc_dim = sum(self.group_inc_dims)
        self.total_cross_dim = sum(self.group_cross_dims)
        self.max_token_value_dim = max(goal_dim, act_dim, state_dim,
                                        self.total_inc_dim, self.total_cross_dim)

    def _compute_full_signature_level2(self, path: np.ndarray) -> np.ndarray:
        d = path.shape[1]
        if path.shape[0] < 2:
            return np.zeros((d * d,), dtype=np.float32)
        
        sig = iisignature.sig(path, 2)
        level2_start = 1 + d
        level2_end = level2_start + d * d
        
        if len(sig) < level2_end:
            level2_sig = np.zeros((d * d,), dtype=np.float32)
            available = max(0, len(sig) - level2_start)
            if available > 0:
                level2_sig[:available] = sig[level2_start:level2_start + available]
        else:
            level2_sig = sig[level2_start:level2_end]
        
        return level2_sig.astype(np.float32)

    def build_window(self, states, actions, goal, t0_action_prev=None):
        T = states.shape[0]
        goal_vec = np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal_vec.shape[0] == 1 and self.goal_dim != 1:
            goal_vec = np.repeat(goal_vec, self.goal_dim).astype(np.float32)
        a_prev = np.zeros((self.act_dim,), dtype=np.float32) if t0_action_prev is None else np.asarray(t0_action_prev, dtype=np.float32).reshape(-1)

        L = 2 + T * 3
        token_type = np.zeros((L,), dtype=np.int64)
        token_time = np.zeros((L,), dtype=np.int64)
        token_group = np.full((L,), -1, dtype=np.int64)
        token_value = np.zeros((L, self.max_token_value_dim), dtype=np.float32)
        pred_mask = np.zeros((L,), dtype=np.bool_)
        target_action = np.zeros((T, self.act_dim), dtype=np.float32)

        ptr = 0
        token_type[ptr] = Tok.GOAL
        token_value[ptr, :self.goal_dim] = goal_vec
        ptr += 1
        token_type[ptr] = Tok.ACTION
        token_value[ptr, :self.act_dim] = a_prev
        ptr += 1

        for t in range(T):
            s_t = states[t].astype(np.float32)
            token_type[ptr] = Tok.OBS
            token_time[ptr] = t
            token_value[ptr, :self.state_dim] = s_t
            ptr += 1

            inc_parts, cross_parts = [], []
            for gi, g in enumerate(self.groups):
                idx = np.array(g.idx, dtype=np.int64)
                path_group = states[:t+1, idx].astype(np.float32)
                s1_full = path_group[-1] - path_group[0]
                inc_parts.append(s1_full)
                s2_full = self._compute_full_signature_level2(path_group)
                cross_parts.append(s2_full)

            inc_concat = np.concatenate(inc_parts)
            token_type[ptr] = Tok.INC
            token_time[ptr] = t
            token_value[ptr, :self.total_inc_dim] = inc_concat
            ptr += 1

            cross_concat = np.concatenate(cross_parts)
            if cross_concat.shape[0] != self.total_cross_dim:
                if cross_concat.shape[0] < self.total_cross_dim:
                    padded = np.zeros((self.total_cross_dim,), dtype=np.float32)
                    padded[:cross_concat.shape[0]] = cross_concat
                    cross_concat = padded
                else:
                    cross_concat = cross_concat[:self.total_cross_dim]
            
            token_type[ptr] = Tok.CROSS
            token_time[ptr] = t
            token_value[ptr, :self.total_cross_dim] = cross_concat
            ptr += 1
            pred_mask[ptr - 1] = True
            target_action[t] = actions[t].astype(np.float32)

        return BuiltSeq(
            token_type=torch.from_numpy(token_type),
            token_time=torch.from_numpy(token_time),
            token_group=torch.from_numpy(token_group),
            token_value=torch.from_numpy(token_value),
            pred_mask=torch.from_numpy(pred_mask),
            target_action=torch.from_numpy(target_action),
        )

    def build_window_seq(self, states, actions, goal, t0_action_prev=None):
        built = self.build_window(states, actions, goal, t0_action_prev)
        return (built.token_type.numpy(), built.token_time.numpy(), built.token_group.numpy(),
                built.token_value.numpy(), built.pred_mask.numpy())


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
# Utility Functions
# ============================================================================

def env_id_from_d4rl_name(env_name: str) -> str:
    """Return the D4RL v2 env name from checkpoint metadata."""
    if env_name is None:
        return "halfcheetah-medium-v2"
    if "-v2" in env_name or "-v1" in env_name or "-v0" in env_name:
        return env_name
    
    s = env_name.lower()
    
    # Maze2D environments
    if "maze2d" in s:
        if "large" in s:
            return "maze2d-large-v1"
        elif "medium" in s:
            return "maze2d-medium-v1"
        else:
            return "maze2d-umaze-v1"
    
    # AntMaze environments
    if "antmaze" in s:
        if "large" in s:
            return "antmaze-large-play-v0"
        elif "medium" in s:
            return "antmaze-medium-play-v0"
        else:
            return "antmaze-umaze-v0"
    
    # MuJoCo locomotion
    if "halfcheetah" in s:
        base = "halfcheetah"
    elif "hopper" in s:
        base = "hopper"
    elif "walker2d" in s or "walker" in s:
        base = "walker2d"
    elif "swimmer" in s:
        base = "swimmer"
    elif "ant" in s:
        base = "ant"
    elif "humanoid" in s:
        base = "humanoid"
    else:
        return "halfcheetah-medium-v2"
    
    if "medium-expert" in s:
        return f"{base}-medium-expert-v2"
    elif "medium-replay" in s:
        return f"{base}-medium-replay-v2"
    elif "expert" in s:
        return f"{base}-expert-v2"
    else:
        return f"{base}-medium-v2"


def get_env_id(data_source: str) -> str:
    """Convert data source to proper environment ID for evaluation."""
    s = data_source.lower()
    
    # Maze2D: use dense version for evaluation
    if 'maze2d' in s:
        if 'large' in s:
            return 'maze2d-large-dense-v1'
        elif 'medium' in s:
            return 'maze2d-medium-dense-v1'
        else:
            return 'maze2d-umaze-dense-v1'
    
    # AntMaze
    if 'antmaze' in s:
        if 'large' in s:
            return 'antmaze-large-play-v0'
        elif 'medium' in s:
            return 'antmaze-medium-play-v0'
        else:
            return 'antmaze-umaze-v0'
    
    # MuJoCo locomotion
    return env_id_from_d4rl_name(data_source)


def normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize observation."""
    return ((obs - mean) / std).astype(np.float32)


def denormalize_action(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Denormalize action."""
    return (action * std + mean).astype(np.float32)


def normalize_score(returns: np.ndarray, env_id: str, env) -> np.ndarray:
    """Compute D4RL normalized score."""
    try:
        ref_min = env.ref_min_score
        ref_max = env.ref_max_score
        return 100.0 * (returns - ref_min) / (ref_max - ref_min)
    except AttributeError:
        return returns


def plot_maze2d_trajectories(
    trajectories: List[np.ndarray],
    env_id: str,
    save_path: str = None,
    target_return: float = None,
):
    """Visualize Maze2D trajectories with maze layout."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Draw maze boundaries based on environment type
    if 'large' in env_id:
        maze_size = 12
        walls = [
            ((0, 0), (12, 0)), ((0, 0), (0, 12)), ((12, 0), (12, 12)), ((0, 12), (12, 12)),
            ((2, 2), (2, 10)), ((4, 0), (4, 6)), ((4, 8), (4, 12)),
            ((6, 2), (6, 10)), ((8, 0), (8, 6)), ((8, 8), (8, 12)),
            ((10, 2), (10, 10)),
        ]
    elif 'medium' in env_id:
        maze_size = 8
        walls = [
            ((0, 0), (8, 0)), ((0, 0), (0, 8)), ((8, 0), (8, 8)), ((0, 8), (8, 8)),
            ((2, 2), (2, 6)), ((4, 0), (4, 4)), ((4, 6), (4, 8)),
            ((6, 2), (6, 6)),
        ]
    else:  # umaze
        maze_size = 4
        walls = [
            ((0, 0), (4, 0)), ((0, 0), (0, 4)), ((4, 0), (4, 4)), ((0, 4), (4, 4)),
            ((1, 1), (3, 1)), ((1, 1), (1, 3)), ((3, 1), (3, 3)),
        ]
    
    # Draw walls
    for (x1, y1), (x2, y2) in walls:
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # Draw trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    for i, traj in enumerate(trajectories):
        if len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], alpha=0.7, linewidth=1.5, label=f'Episode {i+1}')
            ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='x', zorder=5)
    
    ax.set_xlim(-0.5, maze_size + 0.5)
    ax.set_ylim(-0.5, maze_size + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    title = f'Maze2D Trajectories ({env_id})'
    if target_return is not None:
        title += f' | Target RTG: {target_return}'
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main():
    args = parse_args()
    
    # Set device
    if args.force_cpu or args.device == "cpu":
        device = "cpu"
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Setup MuJoCo path (Windows)
    if args.mujoco_path:
        if args.mujoco_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = args.mujoco_path + ";" + os.environ.get("PATH", "")
            print(f"Added MuJoCo to PATH: {args.mujoco_path}")
    
    # Import gym and d4rl after PATH setup
    import gym
    import d4rl
    
    # ========================================================================
    # Load Checkpoint
    # ========================================================================
    print("=" * 60)
    print(f"Loading checkpoint: {args.checkpoint}")
    print("=" * 60)
    
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_meta = ckpt["train_meta"]
    
    # Detect mode and ablation variant
    ablation_variant = train_meta.get("ablation_variant", "baseline")
    mode = train_meta.get("mode", "goal")  # "goal" or "rtg"
    is_rtg_mode = mode == "rtg" or train_meta.get("stride") is not None
    
    print(f"Mode: {mode} (RTG mode: {is_rtg_mode})")
    print(f"Trained on: {train_meta.get('env_name', train_meta.get('data_source'))}")
    print(f"T_window: {train_meta['T_window']}")
    
    T = int(train_meta["T_window"])
    
    # Extract dimensions
    if "builder" in train_meta:
        act_dim = int(train_meta["builder"]["act_dim"])
        obs_dim = int(train_meta["builder"]["state_dim"])
        goal_dim = int(train_meta["builder"].get("goal_dim", 1))
        groups_meta = train_meta.get("groups", None)
    elif "obs_dim" in train_meta and "act_dim" in train_meta:
        obs_dim = int(train_meta["obs_dim"])
        act_dim = int(train_meta["act_dim"])
        goal_dim = 1
        groups_meta = train_meta.get("groups", None)
    else:
        env_name = train_meta.get("env_name", "halfcheetah-medium")
        if "halfcheetah" in env_name.lower() or "walker2d" in env_name.lower() or "hopper" in env_name.lower():
            obs_dim = 17
            act_dim = 6
        elif "maze2d" in env_name.lower():
            obs_dim = 4
            act_dim = 2
        else:
            obs_dim = 17
            act_dim = 6
        goal_dim = 1
        groups_meta = None
    
    print(f"obs_dim={obs_dim}, act_dim={act_dim}, goal_dim={goal_dim}")
    
    # Load normalization stats (new format)
    normalized = False
    state_mean = state_std = action_mean = action_std = None
    
    if "normalization" in ckpt and ckpt["normalization"] is not None:
        norm_stats = ckpt["normalization"]
        state_mean = norm_stats['obs_mean']
        state_std = norm_stats['obs_std']
        action_mean = norm_stats['act_mean']
        action_std = norm_stats['act_std']
        # Convert to numpy if tensors
        if isinstance(state_mean, torch.Tensor):
            state_mean = state_mean.numpy()
            state_std = state_std.numpy()
            action_mean = action_mean.numpy()
            action_std = action_std.numpy()
        normalized = True
        print("[NORMALIZED MODEL]")
    elif "state_mean" in ckpt and ckpt["state_mean"] is not None:
        # Legacy format
        state_mean = ckpt["state_mean"].numpy() if isinstance(ckpt["state_mean"], torch.Tensor) else ckpt["state_mean"]
        state_std = ckpt["state_std"].numpy() if isinstance(ckpt["state_std"], torch.Tensor) else ckpt["state_std"]
        action_mean = ckpt["action_mean"].numpy() if isinstance(ckpt["action_mean"], torch.Tensor) else ckpt["action_mean"]
        action_std = ckpt["action_std"].numpy() if isinstance(ckpt["action_std"], torch.Tensor) else ckpt["action_std"]
        normalized = True
        print("[NORMALIZED MODEL (legacy)]")
    else:
        print("[UNNORMALIZED MODEL]")
    
    def normalize_state(s: np.ndarray) -> np.ndarray:
        return (s - state_mean) / state_std if normalized else s
    
    def normalize_action_fn(a: np.ndarray) -> np.ndarray:
        return (a - action_mean) / action_std if normalized else a
    
    def denormalize_action_fn(a: np.ndarray) -> np.ndarray:
        return a * action_std + action_mean if normalized else a
    
    # Environment ID
    env_name = train_meta.get("env_name", train_meta.get("data_source", "halfcheetah-medium"))
    if args.env:
        env_id = get_env_id(args.env)
        print(f"Using user-specified environment: {args.env} -> {env_id}")
    else:
        env_id = get_env_id(env_name)
        print(f"Auto-detected environment: {env_name} -> {env_id}")
    
    # Set default goal based on environment
    if args.goal is not None:
        goal_value = args.goal
    elif 'maze2d' in env_id.lower():
        goal_value = 1.0  # Normalized RTG for maze
    elif 'antmaze' in env_id.lower():
        goal_value = 1.0
    elif 'halfcheetah' in env_id.lower():
        goal_value = 300.0
    elif 'hopper' in env_id.lower():
        goal_value = 300.0
    elif 'walker2d' in env_id.lower():
        goal_value = 300.0
    else:
        goal_value = 300.0
    
    print(f"Goal value: {goal_value}")
    
    # ========================================================================
    # Setup Groups and Builder
    # ========================================================================
    if groups_meta is not None:
        groups = [GroupSpec(name=g["name"], idx=list(map(int, g["idx"]))) for g in groups_meta]
    else:
        groups = get_default_groups()
    
    include_self_term = True
    if "builder" in train_meta:
        include_self_term = bool(train_meta["builder"].get("include_self_term", True))
    
    # Create the correct builder based on ablation variant
    print(f"\nCreating builder for variant: {ablation_variant}")
    
    if ablation_variant == "fixed_anchor_obs":
        builder = FixedAnchorObsSeqBuilder(
            state_dim=obs_dim, act_dim=act_dim, groups=groups,
            goal_dim=goal_dim, include_self_term=include_self_term,
        )
    elif ablation_variant == "correlation_tokens":
        builder = CorrelationTokenSeqBuilder(
            state_dim=obs_dim, act_dim=act_dim, groups=groups,
            goal_dim=goal_dim, include_self_term=include_self_term,
        )
    elif ablation_variant == "full_signature":
        builder = FullSignatureSeqBuilder(
            state_dim=obs_dim, act_dim=act_dim, groups=groups,
            goal_dim=goal_dim, include_self_term=include_self_term,
        )
    else:
        builder = SignatureSeqBuilder(
            state_dim=obs_dim, act_dim=act_dim, groups=groups,
            goal_dim=goal_dim, include_self_term=include_self_term,
        )
    
    print(f"  total_inc_dim: {builder.total_inc_dim}, total_cross_dim: {builder.total_cross_dim}")
    
    # ========================================================================
    # Build Model
    # ========================================================================
    tfm = train_meta["transformer"]
    n_layer = int(tfm["n_layer"])
    n_head = int(tfm["n_head"])
    n_embd = int(tfm["n_embd"])
    block_size = int(tfm.get("block_size", 512))
    
    model = build_model(vocab_size=1, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd).to(device)
    embed = SignatureTokenEmbedding(
        n_embd=n_embd, state_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim,
        total_inc_dim=builder.total_inc_dim, total_cross_dim=builder.total_cross_dim,
        max_value_dim=builder.max_token_value_dim,
    ).to(device)
    action_head = nn.Linear(n_embd, act_dim).to(device)
    
    # Load weights
    model.load_state_dict(ckpt["gpt_trunk_state_dict"], strict=True)
    embed.load_state_dict(ckpt["embed_state_dict"], strict=False)  # Allow missing RTG proj for old models
    action_head.load_state_dict(ckpt["action_head_state_dict"], strict=True)
    
    model.eval()
    embed.eval()
    action_head.eval()
    print("[OK] Loaded weights successfully.")
    
    # ========================================================================
    # Policy Function
    # ========================================================================
    @torch.no_grad()
    def policy_action_from_history(states_hist: deque, actions_hist: deque, current_rtg: float):
        """Given history deques, predict action for the LAST step."""
        if len(states_hist) == 0:
            raise ValueError("states_hist is empty")
        
        states = np.stack(list(states_hist), axis=0).astype(np.float32)
        actions = np.stack(list(actions_hist), axis=0).astype(np.float32) if len(actions_hist) > 0 else np.zeros((0, act_dim), dtype=np.float32)
        
        # Pad to length T
        n_s = states.shape[0]
        n_a = actions.shape[0]
        
        if n_s < T:
            pad_n = T - n_s
            pad_state = np.repeat(states[:1], pad_n, axis=0)
            states = np.concatenate([pad_state, states], axis=0)
        else:
            states = states[-T:]
        
        if n_a < (T - 1):
            pad_na = (T - 1) - n_a
            pad_a_left = np.zeros((pad_na, act_dim), dtype=np.float32)
            actions_core = np.concatenate([pad_a_left, actions], axis=0)
        else:
            actions_core = actions[-(T - 1):]
        
        dummy_last = np.zeros((1, act_dim), dtype=np.float32)
        actions = np.concatenate([actions_core, dummy_last], axis=0)
        
        # Normalize
        states_for_tokens = normalize_state(states)
        actions_for_tokens = normalize_action_fn(actions)
        
        goal = np.array([current_rtg], dtype=np.float32) if goal_dim == 1 else np.full((goal_dim,), current_rtg, dtype=np.float32)
        
        # Previous action
        if len(actions_hist) >= T:
            t0_action_prev = list(actions_hist)[-T].astype(np.float32)
            t0_action_prev = normalize_action_fn(t0_action_prev)
        else:
            t0_action_prev = None
        
        token_type, token_time, token_group, token_value, pred_mask = builder.build_window_seq(
            states=states_for_tokens,
            actions=actions_for_tokens,
            goal=goal,
            t0_action_prev=t0_action_prev,
        )
        
        token_type_t = torch.from_numpy(token_type[None, :]).to(device)
        token_time_t = torch.from_numpy(token_time[None, :]).to(device)
        token_group_t = torch.from_numpy(token_group[None, :]).to(device)
        token_value_t = torch.from_numpy(token_value[None, :, :]).to(device)
        
        tok = embed(token_type_t, token_time_t, token_group_t, token_value_t)
        L = tok.shape[1]
        pos = torch.arange(0, L, device=tok.device, dtype=torch.long)
        x = model.transformer.drop(tok + model.transformer.wpe(pos)[None, :, :])
        for blk in model.transformer.h:
            x = blk(x)
        x = model.transformer.ln_f(x)
        
        pred_mask_t = torch.from_numpy(pred_mask[None, :]).to(device)
        x_sel = x[pred_mask_t]
        pred = action_head(x_sel).view(T, act_dim)
        a = pred[-1].detach().cpu().numpy()
        
        a = denormalize_action_fn(a)
        
        if args.action_clip is not None:
            a = np.clip(a, -args.action_clip, args.action_clip)
        return a
    
    # ========================================================================
    # Run Evaluation
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"EVALUATION: {ablation_variant}")
    print(f"Environment: {env_id}")
    print(f"Goal/RTG Value: {goal_value}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)
    
    env = gym.make(env_id)
    returns = []
    lengths = []
    trajectories = []  # For visualization
    
    np.random.seed(args.seed)
    
    for ep in range(args.episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        env.seed(args.seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        
        states_hist = deque(maxlen=T)
        actions_hist = deque(maxlen=T)
        states_hist.append(obs.astype(np.float32))
        
        current_rtg = goal_value
        trajectory = [obs.copy()]  # For visualization
        
        for step in range(args.max_steps):
            a = policy_action_from_history(states_hist, actions_hist, current_rtg=current_rtg)
            obs, reward, done, info = env.step(a)
            
            ep_ret += float(reward)
            ep_len += 1
            
            # Update RTG (decrease by reward received)
            current_rtg = max(0, current_rtg - reward)
            
            actions_hist.append(a.astype(np.float32))
            states_hist.append(obs.astype(np.float32))
            trajectory.append(obs.copy())
            
            if args.render:
                env.render()
            
            if done:
                break
        
        returns.append(ep_ret)
        lengths.append(ep_len)
        trajectories.append(np.array(trajectory))
        
        try:
            ep_score = normalize_score(np.array([ep_ret]), env_id, env)[0]
            print(f"Episode {ep:02d}: return={ep_ret:.1f}, score={ep_score:.1f}, len={ep_len}")
        except:
            print(f"Episode {ep:02d}: return={ep_ret:.1f}, len={ep_len}")
    
    env.close()
    
    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)
    
    try:
        norm_scores = normalize_score(returns, env_id, env)
        has_norm_scores = True
    except:
        norm_scores = returns
        has_norm_scores = False
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Environment: {env_id}")
    print(f"Return: mean={returns.mean():.1f} std={returns.std():.1f} min={returns.min():.1f} max={returns.max():.1f}")
    if has_norm_scores:
        print(f"Normalized Score: mean={norm_scores.mean():.1f} std={norm_scores.std():.1f} min={norm_scores.min():.1f} max={norm_scores.max():.1f}")
    print(f"Episode Length: mean={lengths.mean():.1f}")
    
    # Visualization for Maze2D
    if args.plot and 'maze2d' in env_id.lower():
        viz_path = args.checkpoint.replace('.pt', '_trajectories.png')
        plot_maze2d_trajectories(
            trajectories=trajectories,
            env_id=env_id,
            save_path=viz_path,
            target_return=goal_value,
        )
    elif args.plot and 'maze2d' not in env_id.lower():
        print("Note: Trajectory visualization is only supported for Maze2D environments.")
    
    # Save results
    results_path = args.checkpoint.replace('.pt', '_eval_results.json')
    save_dict = {
        'checkpoint': args.checkpoint,
        'env_id': env_id,
        'goal_value': goal_value,
        'num_episodes': args.episodes,
        'avg_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'avg_length': float(lengths.mean()),
        'returns': [float(r) for r in returns],
        'lengths': [int(l) for l in lengths],
    }
    if has_norm_scores:
        save_dict['normalized_score'] = float(norm_scores.mean())
    
    with open(results_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
