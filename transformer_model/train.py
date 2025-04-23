import os
import time
import torch
import matplotlib.pyplot as plt
from model import GPTLanguageModel
from config import *

# ================================
# 📁 Безопасные абсолютные пути
# ================================

base_dir = os.path.dirname(__file__)
data_prepared_path = os.path.join(base_dir, "data_prepared.pt")
final_model_path = os.path.join(base_dir, "model.pt")
loss_plot_path = os.path.join(base_dir, "loss_plot.png")
best_model_path = os.path.join(base_dir, "best_model.pt")

# ================================
# 📥 Загрузка подготовленных данных
# ================================

train_data, val_data, stoi, itos = torch.load(data_prepared_path)
vocab_size = len(stoi)
print(f"🧠 Размер словаря: {vocab_size}, Токенов в обучении: {len(train_data)}")

# ======================
# 📦 Генерация батча
# ======================

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ================================
# 📉 Расчёт потерь на train/val
# ================================

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {'train': [], 'val': []}
    for split in ['train', 'val']:
        total = 0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            total += loss.item()
        avg = total / eval_iters
        losses[split].append(avg)
    model.train()
    return {k: sum(v) / len(v) for k, v in losses.items()}

# =======================
# 🚀 Обучение модели
# =======================

model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 🔁 Добавим scheduler с warmup
warmup_steps = 500
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda step: min((step + 1) / warmup_steps, 1.0)
)

log_steps, train_losses, val_losses = [], [], []
best_val_loss = float('inf')

print("🚀 Начало обучения...\n")
start_time = time.time()

for step in range(max_iters):
    # Валидация и лог
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print(f"Step {step}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f} — ⏱ {mins}m {secs}s")
        log_steps.append(step)
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])

        # Сохраняем промежуточную модель
        ckpt_path = os.path.join(base_dir, f"model_step_{step}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # Сохраняем лучшую модель
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 ✅ Лучшая модель обновлена (val loss = {best_val_loss:.4f})")

    # Обучение
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# Сохраняем финальную модель
torch.save(model.state_dict(), final_model_path)
print(f"\n✅ Финальная модель сохранена в {final_model_path}")

# Время
total_time = time.time() - start_time
mins, secs = divmod(int(total_time), 60)
print(f"\n🏁 Обучение завершено за {mins} мин {secs} сек")

# =======================
# 📈 Отображение графика
# =======================

plt.plot(log_steps, train_losses, label='Train loss')
plt.plot(log_steps, val_losses, label='Val loss')
plt.xlabel("Итерации")
plt.ylabel("Loss")
plt.legend()
plt.title("Потери на обучении")
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_plot_path)
plt.show()