import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GPTLanguageModel
from config import *
from tokenizer import build_vocab, encode

# ================================
# 📥 Загрузка и кодировка текста
# ================================

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Создание словаря
stoi, itos = build_vocab(text)
vocab_size = len(stoi)

# ================================
# 🔢 Кодировка и случайный split
# ================================

encoded_data = encode(text, stoi)
total_len = len(encoded_data)
indices = np.arange(total_len - block_size)

np.random.seed(42)
np.random.shuffle(indices)

n = len(indices)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
n_test = n - (n_train + n_val)

train_ix = indices[:n_train]
val_ix = indices[n_train:n_train+n_val]
test_ix = indices[n_train+n_val:]

def make_blocks(ix_list):
    return torch.stack([torch.tensor(encoded_data[i:i+block_size], dtype=torch.long) for i in ix_list])

train_data = make_blocks(train_ix)
val_data = make_blocks(val_ix)
test_data = make_blocks(test_ix)

print(f"🧐 Словарь: {vocab_size} токенов, Блоков train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

# ======================
# 📦 Генерация батча
# ======================

def get_batch(split):
    data = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }[split]

    if len(data) < batch_size:
        raise ValueError(f"⛔ Недостаточно блоков для batch_size={batch_size}")

    ix = torch.randint(0, len(data), (batch_size,))
    x = data[ix]
    y = torch.roll(x, shifts=-1, dims=1)
    return x.to(device), y.to(device)

# ================================
# 📉 Расчёт потерь
# ================================

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        total = 0.0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            total += loss.item()
        losses[split] = total / eval_iters
    model.train()
    return losses

# =======================
# 🚀 Обучение модели
# =======================

if torch.cuda.is_available():
    print(f"🚀 Используем GPU: {torch.cuda.get_device_name(0)}")
else:
    print("🧠 Используем CPU")

model = GPTLanguageModel(vocab_size).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"✅ Модель инициализирована: {num_params:,} параметров")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

warmup_steps = 500
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0)
)

os.makedirs("models", exist_ok=True)
log_steps, train_losses, val_losses = [], [], []
max_logs = 1000
best_val_loss = float('inf')
print("\n🚀 Начало обучения...")
start_time = time.time()

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print(f"Step {step}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f} — ⏱ {mins}m {secs}s")

        log_steps.append(step)
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])

        if len(log_steps) > max_logs:
            log_steps.pop(0); train_losses.pop(0); val_losses.pop(0)

        torch.save(model.state_dict(), f"models/model_step_{step}.pth")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"💾 ✅ Лучшая модель сохранена (val loss = {best_val_loss:.4f})")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# Сохраняем финальную модель
torch.save(model.state_dict(), "models/model.pth")
print("\n✅ Финальная модель сохранена в models/model.pth")

# Время
total_time = time.time() - start_time
mins, secs = divmod(int(total_time), 60)
print(f"\n🏁 Обучение завершено за {mins} мин {secs} сек")

# =======================
# 📈 График потерь
# =======================

plt.plot(log_steps, train_losses, label='Train loss')
plt.plot(log_steps, val_losses, label='Val loss')
plt.xlabel("Итерации")
plt.ylabel("Loss")
plt.legend()
plt.title("Потери на обучении")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()

# ================================
# 🥺 Финальная оценка на test
# ================================
@torch.no_grad()
def test_model(model):
    model.eval()
    total_loss = 0.0
    for _ in range(100):
        xb, yb = get_batch('test')
        _, loss = model(xb, yb)
        total_loss += loss.item()
    avg_loss = total_loss / 100
    print(f"\n📊 Финальный тест-\u043bосс: {avg_loss:.4f}")

test_model(model)