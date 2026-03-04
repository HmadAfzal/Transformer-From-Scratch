# Transformer from Scratch: An Implementation of "Attention Is All You Need"
![Transformer Architecture](assets/transformer.webp)

Building a Transformer isn't just about importing a library, it's about understanding the delicate dance of tensors, masks, and attention scores. This repository is a ground-up PyTorch implementation of the original Transformer architecture as described in the seminal 2017 paper: **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**.

Instead of using high-level libraries, I implemented every mathematical block in PyTorch to understand the mechanics of Self-Attention, Cross-Attention, and the Encoder-Decoder interplay.

## Why I Built This?

Most modern NLP rests on the shoulders of the Transformer, yet many developers treat it as a black box. I built this repository to deconstruct that box. By implementing every layer manually from the sinusoidal positional encodings to the complex causal masking. I wanted to master the mechanics of how machines understand sequence and context.

---

## 🛠️ Architecture & Implementation Details

### 1. The Core: Multi-Head Attention (`model.py`)

Instead of a single attention pass, I implemented the Multi-Head mechanism which allows the model to "look" at the sentence from different perspectives simultaneously.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 2. Positional Encoding

Since Transformers process tokens in parallel, they lack an inherent sense of "order." I implemented sinusoidal encodings using sine and cosine functions of different frequencies to inject relative position into the embeddings.

### 3. The Data Pipeline (`dataset.py` & `config.py`)

* **Source:** Trained on the **Opus Books** dataset (English to Italian).
* **Tokenizer:** Built a custom `WordLevel` tokenizer to handle the specific vocabulary of the dataset.
* **Masking:** Implemented **Causal Masking** in the decoder to prevent "cheating" during the training phase.

---

## Interpretability & Results

A key feature of this repo is the **Attention Visualization** (`attention_visual.ipynb`). Using **Altair**, I created interactive heatmaps that allow you to see exactly which words the model is "looking at" when it generates a translation.
![Attention Heatmap](assets/attention_map.png)

### Performance Metrics

I integrated `torchmetrics` to track the model's progress beyond simple loss:

* **BLEU Score:** For linguistic quality.
* **Word Error Rate (WER):** For translation accuracy.

---

## 🚀 Quick Start

1. **Setup:**
```bash
git clone https://github.com/HmadAfzal/Transformer-From-Scratch
pip install -r requirements.txt

```


2. **Train the Model:**
```bash
python train.py

```


3. **Run Inference:**
```bash
python translate.py "Building a transformer from scratch is challenging but rewarding."

```


---

## Key Challenges

The most difficult part of this project wasn't the math it was the **Broadcasting and Reshaping**. Specifically, ensuring the attention masks correctly blocked future tokens across multiple heads while maintaining the `(Batch, Head, Seq_Len, d_k)` shape was a significant logic puzzle that required rigorous debugging.

---

> **Note:** The core technical content and architecture of this README are mine, while the English and formatting were refined for clarity by Gemini.
