# Causal Multi-Head Attention: Análise Empírica

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
---

Investigação empírica do mecanismo de Causal Multi-Head Attention, analisando o impacto do bias nas projeções QKV e da variação no número de heads através de visualizações com heatmaps.

## Implementações

O arquivo `mha.ipynb` contém três implementações de mecanismos de atenção:

| Classe | Descrição |
|--------|-----------|
| `CausalSelfAttention` | Self-attention causal com máscara triangular |
| `MultiHeadAttentionWrapper` | Multi-head usando lista de `CausalSelfAttention` |
| `MultiHeadAttention` | Multi-head eficiente com projeções compartilhadas |

Todas as implementações retornam os pesos de atenção para visualização.

## Experimentos

### Parte 1: Efeito do QKV Bias

Compara `CausalSelfAttention` com `qkv_bias=True` vs `qkv_bias=False`.

**Configuração:**
- Entrada: `(batch=5, seq_len=8, d_in=20)`
- Saída: `d_out=16`
- Dropout: `0.1`

**Resultados:**

| Configuração | Distribuição de Atenção | Vetores de Contexto |
|--------------|-------------------------|---------------------|
| Com bias     | Mais uniforme, pesos decaem gradualmente | Maior amplitude |
| Sem bias     | Mais concentrada ("spiky"), picos pronunciados | Menor amplitude |

**Conclusões:**
- O bias atua como regularizador implícito, suavizando a distribuição de atenção
- Introduz um "prior" de atenção independente do conteúdo
- Aumenta a expressividade, mas pode requerer normalização mais cuidadosa

### Parte 2: Número de Heads

Compara `MultiHeadAttention` com 2 heads vs 4 heads, mantendo `d_out` fixo.

**Configuração:**
- Entrada: `(batch=5, seq_len=6, d_in=4)`
- Saída: `d_out=4`

**Trade-off fundamental:**

```
head_dim = d_out / num_heads
```

| Configuração | num_heads | head_dim | Score de Atenção |
|--------------|-----------|----------|------------------|
| 2 Heads      | 2         | 2        | q1·k1 + q2·k2    |
| 4 Heads      | 4         | 1        | q1·k1            |

**Resultados:**
- Correlação entre heads (2 heads): ~0.95
- Vetores de contexto praticamente idênticos entre configurações
- 4 heads com `head_dim=1` não superam 2 heads com `head_dim=2`

**Conclusões:**
- Aumentar heads com `d_out` fixo reduz `head_dim` proporcionalmente
- `head_dim` muito pequeno limita a capacidade de discriminação
- Existe um trade-off entre número de "perspectivas" e riqueza de cada uma

## Dependências

```
uv sync
```


## Referências

1. [Understanding and Coding Self-Attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) - Sebastian Raschka
2. [LLMs from Scratch - Multi-Head Attention](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/multihead-attention.ipynb)