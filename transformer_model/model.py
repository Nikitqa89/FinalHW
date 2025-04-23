"""
GPTLanguageModel — реализация упрощённой GPT-модели с нуля.

Компоненты:
- Multi-Head Self Attention
- Position-wise Feed Forward
- Layer Normalization
- Residual Connections
- Embedding слои
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *

# ============================================
# 🎯 Один self-attention "голова" (внутри блока)
# ============================================

class Head(nn.Module):
    """
    Одна "голова" самовнимания (Self-Attention).

    Преобразует вход в ключи, запросы и значения,
    затем вычисляет матрицу внимания с маской для причинности.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Нижнетреугольная маска для обеспечения причинности (future tokens запрещены)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Attention weights
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Взвешенное агрегирование значений
        v = self.value(x)
        return wei @ v  # (B, T, C)

# ===================================================
# 🔁 Многоголовое внимание (несколько Head в параллель)
# ===================================================

class MultiHeadAttention(nn.Module):
    """
    Несколько голов внимания (параллельно).

    Их выходы конкатенируются и проецируются обратно.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

# ========================================
# 🧠 FeedForward — MLP между attention блоками
# ========================================

class FeedForward(nn.Module):
    """
    Полносвязный слой с GELU-активацией.

    Расширяет размерность в 4 раза, затем проецирует обратно.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ======================
# 🔁 Блок трансформера
# ======================

class Block(nn.Module):
    """
    Один блок трансформера:
    Attention + FeedForward + Residual + LayerNorm
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Сначала attention с нормализацией и резидуальным соединением
        x = x + self.sa(self.ln1(x))
        # Потом FeedForward с нормализацией и резидуальным соединением
        x = x + self.ffwd(self.ln2(x))
        return x

# =======================================
# 🧠 GPT-модель как последовательность блоков
# =======================================

class GPTLanguageModel(nn.Module):
    """
    Полная GPT-модель:
    - Эмбеддинг токенов и позиций
    - Последовательность блоков (Attention + FeedForward)
    - Финальная нормализация и линейный слой
    """
    def __init__(self, vocab_size):
        super().__init__()

        # Эмбеддинг токенов и позиций
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Блоки трансформера
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Финальная нормализация

        # Линейный слой для предсказания токенов
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Прямой проход по модели.

        Args:
            idx: Входной тензор индексов (B, T)
            targets: Целевые значения (B, T), для расчёта потерь

        Returns:
            logits: Предсказания модели (B, T, vocab_size)
            loss: Функция потерь (если заданы targets)
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # Сумма токенов и позиций

        x = self.blocks(x)  # Последовательно проходим через блоки
        x = self.ln_f(x)    # Финальная нормализация
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Вычисляем функцию потерь (если нужно)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss