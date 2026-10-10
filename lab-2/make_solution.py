import json

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text
    })


md(r'''# Неделя 02. Механизм внимания (Attention) и позиционное кодирование (Базовый уровень)

**ФИО:** Черноморов Кирилл Александрович  
**ИСУ:** 367617

**Цель работы:** Реализовать с нуля scaled dot-product attention, причинную маску и KV-кеш, проверить каждый блок тестами и понять, как KV-кеш ускоряет пошаговую генерацию.
''')

code(r'''# ruff: noqa: E402, F811
import sys
import subprocess
import math
import random
import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def print_ok(msg="Все проверки пройдены"):
    """
    Печатает зеленое сообщение об успешной проверке.
    """
    print(f"\033[92m[OK]\033[0m {msg}")


def print_error(msg=""):
    """
    Печатает красное сообщение об ошибке проверки.
    """
    print(f"\033[91m[ERROR]\033[0m {msg}")


print(f"PyTorch: {torch.__version__}")
''')

md(r'''# Задание 1. Scaled Dot-Product Attention — реализация с нуля

Механизм scaled dot-product attention вычисляет сходство между query и key, масштабирует его, применяет softmax и взвешивает векторы value.

Формула:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
''')

code(r'''def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет Scaled Dot-Product Attention.

    q : torch.Tensor
        Query тензор формы (batch, heads, seq_len_q, d_k).
    k : torch.Tensor
        Key тензор формы (batch, heads, seq_len_k, d_k).
    v : torch.Tensor
        Value тензор формы (batch, heads, seq_len_k, d_v).
    mask : torch.Tensor, optional
        Булева маска формы (seq_len_q, seq_len_k); False означает запрещенную позицию.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        output: результат внимания формы (batch, heads, seq_len_q, d_v),
        attn_weights: веса внимания формы (batch, heads, seq_len_q, seq_len_k).
    """
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    return output, attn_weights
''')

code(r'''try:
    batch, heads, seq_len, d_k, d_v = 2, 3, 5, 16, 32
    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_v)

    output, attn = scaled_dot_product_attention(q, k, v)
    assert output.shape == (batch, heads, seq_len, d_v), f"Неверная форма выхода: {output.shape}"
    assert attn.shape == (batch, heads, seq_len, seq_len), f"Неверная форма весов: {attn.shape}"
    assert torch.allclose(attn.sum(dim=-1), torch.ones(batch, heads, seq_len), atol=1e-5), \
        "Веса внимания по строкам должны суммироваться в 1"
    assert torch.allclose(output, attn @ v, atol=1e-5), "output должен равняться attn @ v"

    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(output, ref, atol=1e-5), "Результат расходится с F.scaled_dot_product_attention"

    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    _, attn_masked = scaled_dot_product_attention(q, k, v, mask=mask)
    assert attn_masked.triu(diagonal=1).abs().max() < 1e-6, \
        "Запрещенные позиции маски должны получать нулевой вес"
    print_ok("scaled_dot_product_attention: все проверки пройдены")
except Exception as e:
    print_error(f"Тест scaled_dot_product_attention провален: {e}")
''')

code(r'''# Визуализация: случайные K и K, близкий к Q
torch.manual_seed(0)

seq_len = 8
d_k = 32
q = torch.randn(1, 1, seq_len, d_k)
k_random = torch.randn(1, 1, seq_len, d_k)
k_close = q + 0.1 * torch.randn_like(q)
v = torch.randn(1, 1, seq_len, d_k)

_, attn_random = scaled_dot_product_attention(q, k_random, v)
_, attn_close = scaled_dot_product_attention(q, k_close, v)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(attn_random[0, 0].detach().numpy(), cmap="viridis")
axes[0].set_title("Случайные K")
axes[0].set_xlabel("key")
axes[0].set_ylabel("query")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(attn_close[0, 0].detach().numpy(), cmap="viridis")
axes[1].set_title("K близко к Q")
axes[1].set_xlabel("key")
axes[1].set_ylabel("query")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
''')

md(r'''> ***Вопрос на подумать***
>
> Почему веса внимания нормируются делителем $\sqrt{d_k}$? Что произойдет с softmax, если для больших $d_k$ этот делитель не применять?
>
> ***Ответ:***  
> Деление на $\sqrt{d_k}$ стабилизирует логиты. Если компоненты $q$ и $k$ независимы, имеют нулевое среднее и единичную дисперсию, то скалярное произведение $q \cdot k$ имеет дисперсию порядка $d_k$. При больших $d_k$ логиты становятся очень большими по модулю, и softmax насыщается: один вес становится близок к 1, остальные — к 0. Это приводит к почти one-hot распределению, маленьким градиентам и нестабильному обучению. Масштабирование возвращает дисперсию логитов примерно к 1, распределение остаётся достаточно мягким, а градиенты — информативными.
''')

md(r'''# Задание 2. Causal Mask — причинная маска

В авторегрессионных генеративных моделях токен не должен видеть будущее. Причинная маска запрещает внимание к будущим позициям.
''')

code(r'''def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Создаёт причинную (causal) маску внимания для генерации.

    seq_len : int
        Длина последовательности токенов.

    Returns
    -------
    torch.Tensor
        Булева нижнетреугольная матрица формы (seq_len, seq_len);
        True разрешает внимание к позиции, False запрещает.
    """
    row = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    col = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
    return col <= row
''')

code(r'''try:
    seq_len = 6
    causal_mask = create_causal_mask(seq_len)
    expected = torch.tensor([[j <= i for j in range(seq_len)] for i in range(seq_len)])
    assert causal_mask.shape == (seq_len, seq_len), f"Неверная форма маски: {causal_mask.shape}"
    assert causal_mask.dtype == torch.bool, "Тип данных маски должен быть torch.bool"
    assert torch.equal(causal_mask, expected), \
        "Маска должна разрешать внимание только к текущей и прошлым позициям"

    q = torch.randn(1, 1, seq_len, 8)
    k = torch.randn(1, 1, seq_len, 8)
    v = torch.randn(1, 1, seq_len, 8)
    out_1, attn = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 1, seq_len), atol=1e-5), \
        "Строки весов должны суммироваться в 1"
    assert attn.triu(diagonal=1).abs().max() < 1e-6, \
        "Веса будущих позиций должны быть нулевыми"

    v_changed = v.clone()
    v_changed[:, :, 3:] += 10.0
    out_2, _ = scaled_dot_product_attention(q, k, v_changed, mask=causal_mask)
    assert torch.allclose(out_1[:, :, :3], out_2[:, :, :3], atol=1e-5), \
        "Выходы для ранних позиций не должны зависеть от будущих значений V"
    print_ok("create_causal_mask: все проверки пройдены")
except Exception as e:
    print_error(f"Тест create_causal_mask провален: {e}")
''')

code(r'''# Визуализация: bidirectional vs causal attention
torch.manual_seed(1)

seq_len = 8
d_k = 16
q = torch.randn(1, 1, seq_len, d_k)
k = torch.randn(1, 1, seq_len, d_k)
v = torch.randn(1, 1, seq_len, d_k)

mask = create_causal_mask(seq_len)

_, attn_bidir = scaled_dot_product_attention(q, k, v)
_, attn_causal = scaled_dot_product_attention(q, k, v, mask=mask)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(mask.numpy(), cmap="gray_r")
axes[0].set_title("Causal mask")
axes[0].set_xlabel("key")
axes[0].set_ylabel("query")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(attn_bidir[0, 0].detach().numpy(), cmap="viridis")
axes[1].set_title("Bidirectional attention")
axes[1].set_xlabel("key")
axes[1].set_ylabel("query")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(attn_causal[0, 0].detach().numpy(), cmap="viridis")
axes[2].set_title("Causal attention")
axes[2].set_xlabel("key")
axes[2].set_ylabel("query")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()
''')

md(r'''> ***Вопрос на подумать***
>
> Почему маска нужна при обучении на всей последовательности, а при пошаговой генерации с KV-кешем явная маска по сути не требуется?
>
> ***Ответ:***  
> При обучении на всей последовательности все токены обрабатываются параллельно. Без причинной маски каждый query мог бы видеть future key/value. Это утечка будущей информации, и модель не соответствовала бы авторегрессионной генерации. При пошаговой генерации с KV-кешем на шаге $t$ в кеше находятся только K и V для позиций $\le t$. Будущих токенов там просто нет, поэтому внимание по кешу автоматически causal. Явная маска может понадобиться только для padding-токенов или если в кеш по ошибке попадают будущие позиции.
''')

md(r'''# Задание 3. KV-Cache — кеширование ключей и значений

При пошаговой генерации K и V для прошлых токенов не меняются, поэтому пересчитывать их на каждом шаге неэффективно. KV-cache хранит их и добавляет только новые.
''')

code(r'''class KVCache:
    """
    Кеш ключей (K) и значений (V) для пошагового инференса.
    """

    def __init__(self):
        """
        Инициализация пустого KV-кеша.
        """
        self.k_cache = None
        self.v_cache = None

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Добавляет новые ключи и значения в кеш.

        k_new : torch.Tensor
            Новый тензор ключей формы (batch, n_heads, 1, d_k).
        v_new : torch.Tensor
            Новый тензор значений формы (batch, n_heads, 1, d_v).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            k_cache: полный накопленный тензор K,
            v_cache: полный накопленный тензор V.
        """
        if self.k_cache is None:
            self.k_cache = k_new
            self.v_cache = v_new
        else:
            self.k_cache = torch.cat([self.k_cache, k_new], dim=-2)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=-2)

        return self.k_cache, self.v_cache

    def reset(self) -> None:
        """
        Сбрасывает закешированные значения.
        """
        self.k_cache = None
        self.v_cache = None

    @property
    def seq_len(self) -> int:
        """
        Текущая длина закешированной последовательности токенов.
        """
        if self.k_cache is None:
            return 0
        return self.k_cache.size(-2)


def attention_with_cache(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache: KVCache
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет внимание для нового токена с использованием KV-кеша.

    q_new : torch.Tensor
        Query тензор нового токена формы (batch, n_heads, 1, d_k).
    k_new : torch.Tensor
        Key тензор нового токена формы (batch, n_heads, 1, d_k).
    v_new : torch.Tensor
        Value тензор нового токена формы (batch, n_heads, 1, d_v).
    cache : KVCache
        Объект KV-кеша.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        output: результат внимания формы (batch, n_heads, 1, d_v),
        attn_weights: веса внимания формы (batch, n_heads, 1, seq_len_cache).
    """
    k_full, v_full = cache.update(k_new, v_new)
    output, attn_weights = scaled_dot_product_attention(q_new, k_full, v_full, mask=None)
    return output, attn_weights
''')

code(r'''try:
    cache = KVCache()
    assert cache.seq_len == 0, "Начальная длина кеша должна быть 0"

    k_1 = torch.randn(1, 2, 1, 16)
    v_1 = torch.randn(1, 2, 1, 16)
    k_full, v_full = cache.update(k_1, v_1)
    assert cache.seq_len == 1, "После одного обновления длина кеша должна быть 1"
    assert k_full.shape == (1, 2, 1, 16), f"Неверная форма K: {k_full.shape}"

    cache.update(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    assert cache.seq_len == 2, "После двух обновлений длина кеша должна быть 2"
    cache.reset()
    assert cache.seq_len == 0, "После reset длина кеша должна быть 0"

    seq_len = 6
    q_all = torch.randn(1, 2, seq_len, 16)
    k_all = torch.randn(1, 2, seq_len, 16)
    v_all = torch.randn(1, 2, seq_len, 16)
    full_out, _ = scaled_dot_product_attention(
        q_all, k_all, v_all, mask=create_causal_mask(seq_len)
    )

    cache = KVCache()
    for t in range(seq_len):
        out_t, weights_t = attention_with_cache(
            q_all[:, :, t:t + 1],
            k_all[:, :, t:t + 1],
            v_all[:, :, t:t + 1],
            cache
        )
        assert cache.seq_len == t + 1, f"Шаг {t}: длина кеша должна быть {t + 1}"
        assert out_t.shape == (1, 2, 1, 16), f"Шаг {t}: неверная форма выхода {out_t.shape}"
        assert torch.allclose(out_t, full_out[:, :, t:t + 1], atol=1e-5), \
            f"Шаг {t}: пошаговый выход с кешем не совпадает с полным causal-вниманием"
    print_ok("KVCache и attention_with_cache: все проверки пройдены")
except Exception as e:
    print_error(f"Тест KV-Cache провален: {e}")
''')

code(r'''# Бенчмарк: полный пересчёт vs KV-кеш
def benchmark_generation(n_steps=20, d_k=64, n_heads=4):
    def run(use_cache):
        cache = KVCache() if use_cache else None
        qs, ks, vs = [], [], []
        times = []

        for t in range(n_steps):
            start = time.perf_counter()

            q_new = torch.randn(1, n_heads, 1, d_k)
            k_new = torch.randn(1, n_heads, 1, d_k)
            v_new = torch.randn(1, n_heads, 1, d_k)

            if use_cache:
                _ = attention_with_cache(q_new, k_new, v_new, cache)
            else:
                qs.append(q_new)
                ks.append(k_new)
                vs.append(v_new)

                q_all = torch.cat(qs, dim=-2)
                k_all = torch.cat(ks, dim=-2)
                v_all = torch.cat(vs, dim=-2)

                out, _ = scaled_dot_product_attention(
                    q_all, k_all, v_all,
                    mask=create_causal_mask(t + 1)
                )
                _ = out[:, :, -1:, :]

            times.append(time.perf_counter() - start)

        return times

    return run(True), run(False)


torch.manual_seed(42)
times_cache, times_full = benchmark_generation(n_steps=20, d_k=128, n_heads=8)
steps = np.arange(1, len(times_cache) + 1)

plt.figure(figsize=(8, 5))
plt.plot(steps, np.cumsum(times_full), marker="o", label="Без KV-кеша")
plt.plot(steps, np.cumsum(times_cache), marker="s", label="С KV-кешем")
plt.xlabel("Номер сгенерированного токена")
plt.ylabel("Накопленное время, сек")
plt.title("Генерация 20 токенов: полный пересчёт vs KV-кеш")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
''')

md(r'''> ***Вопрос на подумать***
>
> Во сколько раз снижается вычислительная сложность генерации $N$-го токена при использовании KV-cache по сравнению с полным пересчетом? Какова цена KV-cache по памяти (VRAM)?
>
> ***Ответ:***  
> Для $N$-го токена полный пересчёт без кеша требует заново получить K и V для всех $N$ позиций. Если считать всё внимание, дополнительно возникает $O(N^2 d)$ на attention. KV-кеш позволяет вычислить K и V только для нового токена и один раз посчитать внимание нового query к $N$ закешированным K, V. В пересчёте на один слой стоимость проекций K, V падает примерно с $O(N d^2)$ до $O(d^2)$, то есть ускорение по этой части порядка $N$ раз. Attention для нового токена остаётся $O(N d)$. Итоговое ускорение тем ближе к $N$, чем больше $d$ и чем длиннее последовательность.  
>
> Цена по памяти: нужно хранить
> $$
> 2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot \text{seq\_len} \cdot d_{\text{head}} \cdot \text{batch}
> $$
> чисел. Обычно это float16 или float32. Память растёт линейно с длиной контекста, числом слоёв, числом голов и размером батча. Для длинных контекстов KV-кеш становится основным потребителем VRAM.
''')

md(r'''## Итог

Все задания выполнены:

1. Реализован `scaled_dot_product_attention` с маской и без.
2. Реализована `create_causal_mask`.
3. Реализованы `KVCache` и `attention_with_cache`.
4. Пройдены все проверки.
5. Построены визуализации внимания и бенчмарк KV-кеша.
6. Даны ответы на вопросы «на подумать».
''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("lite_homework_solution.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Готово: lite_homework_solution.ipynb")