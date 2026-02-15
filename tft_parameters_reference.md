# TFT Parameter Reference — Complete Guide

## Applied to Beach Crowd Prediction (NeuralForecast Implementation)

> Every parameter of the Temporal Fusion Transformer explained in detail: where it acts
> inside the architecture, what it does to your data, why the default was chosen,
> and what to change for beach crowd forecasting.

---

## TFT Architecture Pipeline

Your beach data flows through TFT in this order:

```
Raw data
  → Scaling (scaler_type)
  → Embedding (hidden_size)
  → Variable Selection Networks (dropout, grn_activation)
  → Static Covariate Encoder (stat_exog_list, one_rnn_initial_state)
  → LSTM Encoder (input_size, n_rnn_layers, rnn_type)
  → Multi-Head Attention (n_head, attn_dropout)
  → Temporal Fusion Decoder
  → Output (h, loss)
```

Parameters are grouped by their role in this pipeline.

---

## 1. Architecture Parameters

These define the model's structure — what the network looks like.

---

### 1.1 `h` — Forecast Horizon

| | |
|---|---|
| **Type** | `int` (required) |
| **Current value** | `24` |
| **Where it acts** | Decoder output layer |

**Mechanism:** TFT is a direct multi-step forecaster. The decoder produces exactly `h` output
values simultaneously through a linear projection layer: `Linear(hidden_size, h × n_quantiles)`.
This is NOT autoregressive — it doesn't predict step 1, then use that to predict step 2. It
predicts all `h` values in a single forward pass.

**What happens with your beach data:** With `h=24`, the model sees the encoder window and
outputs a vector of 24 predictions at once. Each of those 24 positions goes through the
attention mechanism independently, but they all come from the same encoder representation.

**Why it causes smoothing when large:** When `h` is large (e.g., 168), the model must spread
its "confidence" across 168 positions. The loss function averages across all positions, so the
optimizer finds that predicting a smooth curve minimizes average error better than trying to
nail each peak. With `h=24`, there are only 24 positions to worry about, so the model can
afford to be more precise at each one.

**The math:** With `h=168` and `hidden_size=128`, the output mapping is 128→168 — more
outputs than inputs, which is underdetermined. With `h=24`, it's 128→24, which is
overdetermined and more stable.

**Rule of thumb:** `h` should be ≤ `input_size`. The ratio `input_size / h` determines how
much context per prediction step. Below 1:1 you're asking the model to predict more steps
than it observes, which almost guarantees smoothing.

**Recommended values for beach data:**
- `h=24` (1 day of daytime hours) — good balance
- `h=12` (half day) — sharper predictions, less lookahead
- `h=6` — very sharp, useful for real-time monitoring

---

### 1.2 `input_size` — Encoder Lookback Window

| | |
|---|---|
| **Type** | `int` (required) |
| **Current value** | `48` |
| **Where it acts** | LSTM encoder |

**Mechanism:** Defines the sliding window of past observations that the LSTM processes
sequentially. The LSTM reads timestep `t - input_size` through `t - 1`, updating its hidden
state at each step. The final hidden state (plus all intermediate states) becomes the
"encoded representation" that the attention layer uses.

**What happens with your beach data:** With `input_size=48` on daytime sequential data
(roughly 3.5 days of ~14h/day), the model sees about 3 daily cycles. It observes: "Monday was
busy, Tuesday less, Wednesday had rain and was empty." From this pattern, the attention layer
decides which of those 48 past steps are most relevant for predicting the next 24.

**Why too small hurts:** With `input_size=12` (less than one day), the model never sees a
complete daily cycle in its window. It can't learn "mornings start slow and peak at 2pm"
because it might only see 7am–7pm OR 10am–10pm depending on the window position.

**Why too large wastes resources:** Each LSTM step stores activations for backpropagation.
Memory usage scales linearly. And the LSTM's hidden state at step 168 has very little gradient
connection to step 0 — it effectively "forgot" what it saw at the beginning.

**Long-term knowledge doesn't come from input_size:** The exogenous features (`month`,
`is_summer`, `day_of_week`) tell the model "it's July, Saturday" without needing months of
history. The model weights themselves encode seasonal patterns learned during training.

**Recommended values:**
- `input_size=48` with `h=24` — ratio 2:1, good default
- `input_size=96` with `h=24` — captures ~1 week of daytime, good for weekly patterns
- Never below 14 (one daytime cycle)

---

### 1.3 `hidden_size` — Internal Representation Width

| | |
|---|---|
| **Type** | `int` |
| **Default** | `128` |
| **Where it acts** | Everywhere — embeddings, LSTM, GRN, attention, VSN |

**Mechanism:** Every component inside TFT operates in `hidden_size`-dimensional space:
- Each input variable gets embedded from its raw dimension (1 for continuous) into
  `hidden_size` dimensions
- The LSTM has `hidden_size` units in its hidden state
- The Gated Residual Networks (GRN) have `hidden_size` width
- The Variable Selection Networks output `hidden_size`-dimensional weighted features
- The attention Q, K, V matrices are `hidden_size × hidden_size`

**What happens with your beach data:** With `hidden_size=128`, every beach count value,
every temperature reading, every hour indicator gets projected into a 128-dimensional vector.
The LSTM maintains a 128-dimensional "memory" as it processes each timestep. The attention
layer computes similarity in 128-dimensional space.

**Why 64 was too small (the spline behavior):** With 64 dimensions, the model had to
compress ~40 features (weather + temporal) into a 64-dim space, then represent temporal
patterns in that same 64-dim space. There wasn't enough capacity to represent both "it's a
hot Saturday in July" AND "the crowd peaked yesterday at 150" simultaneously. The model
compromised by outputting a smooth average.

**Why 256+ can overfit:** With 128 dimensions, each weight matrix has 128×128 = 16,384
parameters. Going to 256 quadruples that to 65,536 per matrix, across ~20 matrices that's
over 1M parameters. With only ~40k training samples, the model memorizes training data instead
of learning generalizable patterns.

**Rule of thumb:** `hidden_size` should be 2–4× the number of input features. With ~40
features, 128 (3.2×) is well-calibrated.

---

### 1.4 `n_head` — Number of Attention Heads

| | |
|---|---|
| **Type** | `int` |
| **Default** | `4` |
| **Where it acts** | Multi-head attention layer in the Temporal Fusion Decoder |

**Mechanism:** The attention layer looks at all encoded timesteps and decides "which past
moments are most relevant for predicting each future moment?" Multi-head attention splits this
into `n_head` independent attention computations, each operating on
`hidden_size / n_head` dimensions.

With `n_head=4` and `hidden_size=128`: each head works in 32-dimensional space. Head 1 might
learn to attend to "same hour yesterday," head 2 to "weekend vs weekday pattern," head 3 to
"weather transition moments," head 4 to "recent trend."

**TFT's interpretable modification:** Standard transformers concatenate head outputs. TFT
uses shared values `V` across heads and **additive** aggregation. This means the attention
weights can be directly interpreted — you can visualize which past timesteps the model
considers important using `nf.models[0].attention_weights()`.

**Constraint:** `n_head` must divide `hidden_size` evenly. 128/4 = 32 ✓, 128/5 = 25.6 ✗.

**What your data sees:** With 4 heads and `input_size=48`, the model computes 4 separate
48-length attention weight vectors for each of the 24 future positions. That's 4 × 24 = 96
attention distributions. The model might discover: head 1 always attends to the same hour
yesterday (step t-14), head 2 spreads attention across the last 3 hours, head 3 focuses on
the exact same weekday last week.

**Recommended values:**
- `n_head=4` with `hidden_size=128` → 32 dims per head (good default)
- `n_head=8` with `hidden_size=128` → 16 dims per head (more diverse but less capacity each)
- `n_head=2` with `hidden_size=128` → 64 dims per head (fewer but richer patterns)

---

### 1.5 `n_rnn_layers` — LSTM Encoder Depth

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1` |
| **Where it acts** | LSTM/GRU encoder |

**Mechanism:** Stacks multiple LSTM layers on top of each other. With `n_rnn_layers=1`, one
LSTM reads the input sequence and produces hidden states. With `n_rnn_layers=2`, the first
LSTM's outputs become the inputs to a second LSTM, which produces a "higher-level" temporal
representation.

**What happens with your beach data:** Layer 1 learns low-level patterns (hourly fluctuations,
immediate weather effects). Layer 2 would learn higher-level patterns (multi-day trends,
weekly rhythms) from Layer 1's outputs.

**Tradeoff:** Each additional layer roughly doubles the RNN parameters and training time. With
your ~6 months of data, `n_rnn_layers=1` is appropriate. The original TFT paper used 1 layer.
Going to 2 can help with very complex multi-scale patterns, but with your data size it's more
likely to overfit.

---

### 1.6 `rnn_type` — Encoder Architecture

| | |
|---|---|
| **Type** | `str` |
| **Default** | `'lstm'` |
| **Options** | `'lstm'`, `'gru'` |
| **Where it acts** | Temporal encoder |

**LSTM** (Long Short-Term Memory): Has 3 gates (input, forget, output) plus a cell state.
The cell state acts as a separate "memory highway" that can carry information across many
timesteps with minimal gradient decay. More parameters per layer.

**GRU** (Gated Recurrent Unit): Has 2 gates (reset, update), no separate cell state. Simpler,
~25% fewer parameters, slightly faster training. Often performs comparably to LSTM on shorter
sequences.

**For beach data:** With `input_size=48` (moderate length), both perform similarly. LSTM is
the TFT paper's original choice. GRU could help if overfitting is a concern (fewer parameters).

---

### 1.7 `one_rnn_initial_state` — Shared vs Independent RNN Initialization

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `False` |
| **Where it acts** | Connection between Static Covariate Encoder and LSTM |

**Mechanism:** When you have static exogenous variables (`stat_exog_list`), the Static
Covariate Encoder produces context vectors that initialize the LSTM's hidden state and cell
state. With `False`, each RNN layer gets its own independently computed initial state. With
`True`, all layers share the same initial state.

**For your data:** You're not currently using `stat_exog_list`, so this parameter has no
effect. If you added beach-level static features (capacity, orientation, latitude), `False`
gives each LSTM layer its own beach-specific initialization.

---

### 1.8 `tgt_size` — Target Size

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1` |
| **Where it acts** | Output layer |

**Mechanism:** For multivariate forecasting — predicting multiple target variables
simultaneously. `tgt_size=1` means univariate (just predicting `count`).

**For your data:** Keep at 1. You're predicting one variable (crowd count) per beach.

---

### 1.9 `grn_activation` — GRN Non-linearity

| | |
|---|---|
| **Type** | `str` |
| **Default** | `'ELU'` |
| **Options** | `'ReLU'`, `'Softplus'`, `'Tanh'`, `'SELU'`, `'LeakyReLU'`, `'Sigmoid'`, `'ELU'`, `'GLU'` |
| **Where it acts** | Every Gated Residual Network in the architecture |

**Mechanism:** The GRN is the fundamental building block of TFT — it appears inside every
VSN, inside the static covariate encoder, and inside the temporal fusion decoder. The GRN
transforms inputs through two linear layers with a non-linear activation between them:

```
GRN(a, c) = LayerNorm(a + GLU(W₂ · activation(W₁ · a + W₃ · c + b₁) + b₂))
```

The `grn_activation` controls that inner non-linearity.

**Options explained:**
- `'ELU'` (default): Smooth, allows negative outputs, no "dead neuron" problem. Best
  general-purpose choice.
- `'ReLU'`: Simple and fast, but neurons can "die" (output always 0) if large negative inputs
  push them permanently below zero.
- `'SELU'`: Self-normalizing — maintains mean≈0 and variance≈1 across layers. Good
  theoretical properties but sensitive to weight initialization.
- `'GLU'`: Gated Linear Unit — adds extra gating on top of the GRN's own GLU gate.
  Double-gating can suppress more aggressively.
- `'Softplus'`, `'Tanh'`, `'LeakyReLU'`, `'Sigmoid'`: Standard alternatives.

**For beach data:** ELU is the recommended default. Only change during systematic
hyperparameter search.

---

## 2. Feature Parameters

These control what information the model receives.

---

### 2.1 `hist_exog_list` — Historical Exogenous Features

| | |
|---|---|
| **Type** | `list of str` |
| **Default** | `None` |
| **Where it acts** | Variable Selection Network → LSTM encoder (past only) |

**Mechanism:** Features known only for past timesteps. The encoder sees them for
t-input_size through t-1, but the decoder does NOT see them for future steps t+1 through t+h.

**What should go here:** Weather variables — you don't have reliable future weather forecasts.
`ae_temperature`, `om_humidity`, `om_wind_speed`, etc.

**Current problem:** You have ALL features here, including temporal ones (hour, month) that
are known for the future. This means the decoder has zero exogenous information about future
positions.

---

### 2.2 `futr_exog_list` — Future Exogenous Features

| | |
|---|---|
| **Type** | `list of str` |
| **Default** | `None` |
| **Where it acts** | Variable Selection Network → both encoder AND decoder |

**Mechanism:** Features known for both past and future timesteps. Both the encoder and
decoder see them. This is the critical difference — the decoder at position t+12 gets explicit
information like "it's Saturday at 2pm in July."

**What should go here:** Temporal features — `hour`, `day_of_week`, `month`, `is_weekend`,
`is_summer`. You always know what hour/day/month it will be in the future.

**Why this is critical for your data:** Without `futr_exog_list`, the decoder must guess from
the encoded past that position +12 corresponds to Saturday 2pm. With it, the decoder directly
knows "this prediction target is Saturday 2pm" and can apply the correct seasonal pattern.

**If weather forecasts are available:** Move weather variables here too. "It will rain
tomorrow" is extremely valuable for crowd prediction.

---

### 2.3 `stat_exog_list` — Static Exogenous Features

| | |
|---|---|
| **Type** | `list of str` |
| **Default** | `None` |
| **Where it acts** | Static Covariate Encoder → context for VSN, LSTM initialization, decoder enrichment |

**Mechanism:** Features constant per time series (per beach). They produce 4 context vectors:
- `c_s`: temporal variable selection context (helps VSN decide feature importance)
- `c_e`: decoder enriching context (helps attention interpretation)
- `c_h`, `c_c`: LSTM hidden/cell state initialization

**What could go here for your data:** Beach-specific attributes like maximum capacity,
beach orientation (north/south), whether it's urban or rural, latitude/longitude. You'd pass
these as a separate `static_df` to NeuralForecast.

**Currently unused** but could improve multi-beach models by telling TFT "this is a big urban
beach" vs "this is a small rural cove."

---

## 3. Regularization Parameters

These prevent overfitting.

---

### 3.1 `dropout` — VSN Input Dropout

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.1` |
| **Where it acts** | Variable Selection Networks — applied to input embeddings |

**Mechanism:** Randomly zeros 10% of the embedded feature values during training. Each
feature embedding goes through a GRN, and dropout is applied within those GRNs.

**What happens with your beach data:** With 40+ features (weather + temporal), some features
are noisy or redundant. Dropout forces the VSN to not rely too heavily on any single feature.
If `om_temperature` is dropped in one pass, the model must learn to use `ae_temperature` or
`hour` as alternatives.

**Interaction with data size:** Small data (~40k samples) benefits from moderate dropout
(0.1–0.2). Large datasets (millions) can use lower dropout (0.0–0.05).

---

### 3.2 `attn_dropout` — Attention Dropout

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.0` |
| **Where it acts** | Multi-head attention layer |

**Mechanism:** Randomly zeros out attention weights before they're applied to value vectors.
If `attn_dropout=0.2`, 20% of attention connections are randomly dropped during training.

**What happens with your beach data:** Without it, the model might always attend heavily to
"same hour yesterday" (step t-14). With `attn_dropout=0.2`, that connection is randomly broken
20% of the time, forcing the model to also learn backup temporal patterns.

**Difference from `dropout`:** `attn_dropout` regularizes only the attention mechanism.
`dropout` regularizes the Variable Selection Networks (input gates). They act on different
parts of the architecture.

**For beach data:** Default 0.0 is fine with ~40k samples. Try 0.1 if `hidden_size` > 128.

---

### 3.3 `scaler_type` — Per-Series Normalization

| | |
|---|---|
| **Type** | `str` |
| **Default** | `'robust'` |
| **Options** | `'robust'`, `'standard'`, `'minmax'`, `'identity'` |
| **Where it acts** | Applied before any neural network processing, reversed after prediction |

**Mechanism:** Each time series (each beach) is independently normalized before feeding to
the model. This ensures that a beach with max 200 people and a beach with max 20 people both
produce similar-magnitude values for the network.

- `'robust'`: Uses median and IQR. `scaled = (x - median) / IQR`. Insensitive to outliers.
- `'standard'`: Uses mean and std. `scaled = (x - mean) / std`. Outliers inflate std,
  compressing normal values.
- `'minmax'`: Uses min and max. `scaled = (x - min) / (max - min)`. A single outlier
  (festival day with 500 people) compresses the entire series to near-zero.
- `'identity'`: No scaling at all.

**Why `'robust'` is best for beach data:** Beach counts have heavy-tailed distributions. Most
hours have 0–50 people, but festival days might have 300+. With `'standard'`, those festival
days inflate the std, making normal variations look like noise. With `'robust'`, the median
and IQR capture the typical range, and outliers just get larger scaled values without
distorting everything else.

---

## 4. Training Parameters

These control the optimization process.

---

### 4.1 `max_steps` — Total Gradient Updates

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1000` |
| **Current value** | `1500` |
| **Where it acts** | Training loop (PyTorch Lightning Trainer) |

**Mechanism:** One "step" = one batch of data → forward pass → compute loss → backward pass →
update weights. This is NOT the same as epochs. If you have 40,000 samples and
`batch_size=64`, one epoch ≈ 625 steps. So `max_steps=1500` ≈ ~2.4 epochs.

**What happens during training:** At step 0, all weights are random — predictions are
garbage. As steps increase, the model gradually learns. First it learns the mean (stop
predicting negatives). Then the daily pattern (peaks at 2pm). Then weekly patterns
(Saturday > Wednesday). Then weather effects (rain → low counts). Each pattern requires more
steps because it's a more subtle signal.

**Why 500 was too few:** With 500 steps and batch_size=64, the model saw only 32,000 samples
total. It barely learned the daily pattern. Result: smooth mean-like predictions.

**Why 3000+ may overfit:** After enough steps, training loss keeps decreasing but validation
loss starts increasing — the model memorizes noise. Without early stopping, training continues
past the optimal point.

**Diagnostic:** If training loss is still clearly dropping at max_steps, increase. If it
plateaued by step 800, you have enough.

---

### 4.2 `learning_rate` — Optimizer Step Size

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.001` |
| **Where it acts** | Adam optimizer |

**Mechanism:** After each backward pass, every weight `w` is updated:
`w = w - lr × gradient`. The learning rate scales how large each update is. Adam also
maintains per-weight adaptive rates, but the global `lr` multiplies all of them.

- `lr=1e-2` (too high): Huge steps, model oscillates, can't settle into a minimum.
- `lr=1e-3` (default): Steady convergence, each step makes small reliable improvement.
- `lr=1e-5` (too low): Barely changes anything. 1500 steps accomplish what 15 steps at 1e-3
  would.

**Interaction with `max_steps`:** Halving the lr roughly requires 2× the steps to reach the
same point. `lr=1e-4` with `max_steps=1500` is equivalent to `lr=1e-3` with `max_steps=150`.

---

### 4.3 `batch_size` — Samples Per Gradient Step

| | |
|---|---|
| **Type** | `int` |
| **Default** | `32` |
| **Current value** | `64` |
| **Where it acts** | Data loader |

**Mechanism:** Each training step samples `batch_size` windows from the dataset. The gradient
is the average gradient across these samples. Larger batches produce more stable (less noisy)
gradient estimates.

**Tradeoff:**
- Large batch (128): each step takes longer but gradient is accurate. Training path is
  smooth and direct. May generalize worse (converges to "sharp" minima).
- Small batch (32): each step is fast and noisy. Noise can help escape shallow local minima
  (regularization). But can wander if lr is too high.

**VRAM impact:** Directly controls GPU memory. Your A6000 (48GB) can easily handle batch=256+,
but the generalization tradeoff matters more than memory.

---

### 4.4 `loss` — Training Objective

| | |
|---|---|
| **Type** | PyTorch module |
| **Default** | `MAE()` |
| **Options** | `MAE()`, `MSE()`, `RMSE()`, `QuantileLoss()`, `SMAPE()`, `DistributionLoss()` |
| **Where it acts** | Loss computation during training |

**Mechanism:** Defines what "good prediction" means mathematically.

- `MAE()` (L1): `|actual - predicted|` averaged. All errors weighted equally.
- `MSE()` (L2): `(actual - predicted)²` averaged. Quadratic penalty — being off by 50 counts
  25× more than being off by 10.
- `RMSE()`: `√(MSE)`. Same optimization as MSE but in original units.
- `QuantileLoss()`: Predicts multiple quantiles (10th, 50th, 90th), producing prediction
  intervals.
- `DistributionLoss(distribution='StudentT', level=[80, 90])`: Full probabilistic output —
  predicts distribution parameters.

**For beach data:** MAE is preferred because it doesn't disproportionately penalize peak-hour
errors. If you care more about getting peaks right, try MSE.

---

### 4.5 `valid_loss` — Validation Objective

| | |
|---|---|
| **Type** | PyTorch module |
| **Default** | `None` (same as `loss`) |
| **Where it acts** | Validation evaluation |

**Mechanism:** When `None`, validation uses the same loss as training. You can set a different
loss — for example, train with `MSE()` (helps with peaks) but validate with `MAE()` (matches
your RelMAE metric). Model selection picks the checkpoint minimizing validation loss.

---

### 4.6 `early_stop_patience_steps` — Early Stopping

| | |
|---|---|
| **Type** | `int` |
| **Default** | `-1` (disabled) |
| **Where it acts** | Training loop |

**Mechanism:** When positive (e.g., 50), training stops if validation loss doesn't improve
for that many consecutive validation checks. With `val_check_steps=50` and patience=5,
training stops if no improvement for 250 training steps.

**Currently disabled** because `cross_validation` handles data splits internally and NF's
early stopping can conflict. If using `fit/predict`, enable with patience=50.

---

### 4.7 `val_check_steps` — Validation Frequency

| | |
|---|---|
| **Type** | `int` |
| **Default** | `100` |
| **Current value** | `50` |
| **Where it acts** | Training loop |

**Mechanism:** Runs validation every N training steps. Used for logging, early stopping, and
LR scheduling. 50 with max_steps=1500 → ~30 checks, which is enough to track convergence.

---

### 4.8 `num_lr_decays` — Learning Rate Schedule

| | |
|---|---|
| **Type** | `int` |
| **Default** | `-1` (no decay) |
| **Where it acts** | StepLR scheduler |

**Mechanism:** When positive, the learning rate is halved at evenly spaced intervals across
`max_steps`. With `num_lr_decays=3` and `max_steps=1500`:

- Steps 0–375: lr = 1e-3
- Steps 375–750: lr = 5e-4
- Steps 750–1125: lr = 2.5e-4
- Steps 1125–1500: lr = 1.25e-4

**Why this helps:** Large initial LR gets near the optimum quickly. Smaller LR fine-tunes.
For `max_steps=1500`, constant LR is usually fine. For 3000+ steps, `num_lr_decays=2` helps.

---

### 4.9 `optimizer` / `optimizer_kwargs` — Optimizer Choice

| | |
|---|---|
| **Type** | Subclass of `torch.optim.Optimizer` / `dict` |
| **Default** | `None` (Adam) |
| **Where it acts** | Training loop |

**Mechanism:** When `None`, uses Adam with the specified `learning_rate`. You can pass any
PyTorch optimizer:

```python
optimizer=torch.optim.AdamW,
optimizer_kwargs={'weight_decay': 0.01}
```

AdamW adds L2 regularization (weight decay) which prevents overfitting.

---

### 4.10 `lr_scheduler` / `lr_scheduler_kwargs` — LR Schedule

| | |
|---|---|
| **Type** | Subclass of `torch.optim.lr_scheduler.LRScheduler` / `dict` |
| **Default** | `None` (StepLR controlled by `num_lr_decays`) |
| **Where it acts** | Training loop, after each step |

**Mechanism:** Custom LR schedules. For example, cosine annealing:

```python
lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
lr_scheduler_kwargs={'T_max': 1500, 'eta_min': 1e-5}
```

Starts at `learning_rate=1e-3` and smoothly decreases to `1e-5` following a cosine curve.

---

## 5. Data Pipeline Parameters

These control how data is windowed, batched, and fed to the model.

---

### 5.1 `windows_batch_size` — Training Window Sampling

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1024` |
| **Where it acts** | NeuralForecast's internal windowing |

**Mechanism:** NeuralForecast creates training windows by sliding across each time series.
With `input_size=48` and `h=24`, each window is 72 timesteps. For 16 beaches × ~2000 steps,
you get ~30,000 windows. `windows_batch_size=1024` means each training step samples 1024 of
these, then creates mini-batches of `batch_size=64` from them.

**Two-level batching:**
1. Sample 1024 windows from all available windows (`windows_batch_size`)
2. Feed them in groups of 64 (`batch_size`)

**For your data:** With ~30k windows, 1024 means each step sees 3.4% of windows. Setting
`windows_batch_size=None` loads all windows — feasible with A6000 GPUs and your data size.
This gives better gradient estimates.

---

### 5.2 `inference_windows_batch_size` — Prediction Window Sampling

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1024` |
| **Where it acts** | Data pipeline during prediction/cross_validation |

**Mechanism:** Same as `windows_batch_size` but for inference. `-1` means use all windows.

**For your data:** Set to `-1` for fastest inference since your data fits in memory.

---

### 5.3 `valid_batch_size` — Validation Batch Size

| | |
|---|---|
| **Type** | `int` |
| **Default** | `None` (same as `batch_size`) |
| **Where it acts** | Validation data loader |

**Mechanism:** Validation doesn't compute gradients, so it uses less VRAM per sample. You can
set this larger (e.g., 256) to speed up validation without affecting model quality.

---

### 5.4 `step_size` — Window Stride

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1` |
| **Where it acts** | Window sliding during data preparation |

**Mechanism:** Controls how many timesteps the window moves between consecutive windows.

- `step_size=1`: overlapping windows [0,72), [1,73), [2,74)... Maximum data augmentation
  (~30k windows) but heavily correlated.
- `step_size=24`: non-overlapping predictions [0,72), [24,96)... Fewer windows but each is
  independent. During `cross_validation`, `step_size=h` means each test timestep is predicted
  exactly once.

---

### 5.5 `start_padding_enabled` — Zero-Padding at Series Start

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `False` |
| **Where it acts** | Beginning of each time series |

**Mechanism:** When `False`, the model can't predict the first `input_size` timesteps (not
enough history). When `True`, the series is padded with zeros at the start.

**For beach data:** Keep `False`. Padding with zeros would tell the model "there were 0 people
for 48 hours before data starts," which is misleading. You have enough data (~2000 steps per
beach) that losing 48 at the start is negligible.

---

### 5.6 `training_data_availability_threshold` — Missing Data Tolerance

| | |
|---|---|
| **Type** | `float` or `list[float, float]` |
| **Default** | `0.0` |
| **Where it acts** | Window filtering |

**Mechanism:** Minimum fraction of non-missing data required in a training window. With `0.0`,
even a window with 1 valid point (rest NaN) is included. With `0.5`, at least half must be
valid. A list `[0.8, 0.5]` sets 80% for insample and 50% for outsample.

**For your data:** No effect — your cleaned sequential data has no NaNs.

---

### 5.7 `drop_last_loader` — Drop Incomplete Batches

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `False` |
| **Where it acts** | PyTorch DataLoader |

**Mechanism:** Drops the last mini-batch if it's smaller than `batch_size`. The smaller batch
has different gradient statistics which can cause minor instability.

**For your data:** Keep `False`. Minimal effect with your data sizes.

---

### 5.8 `dataloader_kwargs` — Extra DataLoader Arguments

| | |
|---|---|
| **Type** | `dict` |
| **Default** | `None` |
| **Where it acts** | PyTorch DataLoader |

**Mechanism:** Passes extra arguments like `num_workers` (parallel data loading) or
`pin_memory` (faster GPU transfer).

```python
dataloader_kwargs={'num_workers': 4, 'pin_memory': True}
```

---

### 5.9 `freq` — Data Frequency (NeuralForecast Constructor)

| | |
|---|---|
| **Type** | `str` or `int` |
| **Where it acts** | NeuralForecast's internal data handling |

**Mechanism:** Tells NF how to interpret the `ds` column.
- `freq='h'` → expects datetime with hourly spacing, uses `fill_gaps` for missing hours
- `freq=1` → expects integers, each row is one step

**Why `freq=1` for daytime data:** With daytime-only data, there's a 12-hour gap every night
(8pm to 7am). `freq='h'` would insert 12 rows of interpolated data per night — recreating
the night bias. `freq=1` treats daytime steps as continuous: step 13 (Mon 8pm) → step 14
(Tue 7am) with no gap. Temporal features (hour, day_of_week) carry the real calendar
information.

---

## 6. Reproducibility & Output Parameters

---

### 6.1 `random_seed` — Deterministic Training

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1` |

Sets PyTorch, NumPy, and Python seeds. Same seed + same hardware = same results. Essential
for the sensitivity analysis notebook where differences must come from parameter changes, not
randomness.

---

### 6.2 `alias` — Custom Model Name

| | |
|---|---|
| **Type** | `str` |
| **Default** | `None` → `'TFT'` |

The prediction column in cross_validation output is named after the model. Setting
`alias='TFT_v2'` renames it. Useful when comparing multiple TFT configurations.

---

### 6.3 `**trainer_kwargs` — PyTorch Lightning Trainer

| | |
|---|---|
| **Type** | keyword arguments |
| **Where it acts** | Lightning Trainer |

Passes directly to PL's Trainer. Important ones for your setup:

- `accelerator='gpu'` — use GPU
- `devices=[0]` — which GPU(s)
- `gradient_clip_val=1.0` — clips gradients to prevent explosion
- `enable_progress_bar=True` — shows training progress
- `enable_checkpointing=True` — saves model checkpoints (default False to save disk)
- `precision='16-mixed'` — mixed FP16, ~2× faster and half VRAM on A6000s

---

## 7. Interpretability Methods

TFT provides three built-in interpretability tools accessible after training:

### 7.1 `attention_weights()`

Returns a numpy array of shape `(input_size + h, input_size + h)` showing which past
timesteps the model attends to for each prediction position. Visualizable as heatmap,
per-horizon lines, or averaged over time.

### 7.2 `feature_importances()`

Returns a dict with three DataFrames:
- `'Past variable importance over time'` — how important each hist feature is at each past step
- `'Future variable importance over time'` — same for futr features
- `'Static covariates'` — importance of static features

These come from the VSN weights (softmax over GRN outputs).

### 7.3 `feature_importance_correlations()`

Shows which features gain/lose importance at the same time — reveals feature interactions.

---

## 8. Parameter Interaction Map

| Parameter | Primary Effect | Key Interactions |
|---|---|---|
| `h` | Prediction sharpness | `input_size` (ratio), `hidden_size` (output layer width) |
| `input_size` | Context richness | `h` (ratio), `hidden_size` (capacity to use context) |
| `hidden_size` | Model capacity | `n_head` (must divide), data size (overfitting risk), all components |
| `n_head` | Attention diversity | `hidden_size` (dim per head) |
| `n_rnn_layers` | Encoder depth | `hidden_size` (params per layer), data size |
| `max_steps` | Convergence | `learning_rate` (lr × steps = learning), `batch_size` |
| `learning_rate` | Convergence speed | `max_steps`, `batch_size`, `num_lr_decays` |
| `batch_size` | Gradient stability | `learning_rate`, VRAM, `windows_batch_size` |
| `dropout` | VSN regularization | Data size, number of features |
| `attn_dropout` | Attention regularization | `hidden_size`, `n_head` |
| `scaler_type` | Normalization | Outlier distribution in data |
| `loss` | Optimization target | Metric you care about (MAE→RelMAE, MSE→peak accuracy) |
| `hist/futr_exog` | Decoder information | `input_size` (less important if futr gives calendar info) |
| `windows_batch_size` | Memory / gradient quality | `batch_size`, total windows count |

---

## 9. Recommended Configuration for Beach Crowd Prediction

```python
TFT(
    h=24,                        # 1 day of daytime predictions
    input_size=48,               # ~3.5 days of context
    hidden_size=128,             # 3.2× feature count
    n_head=4,                    # 32 dims per head
    n_rnn_layers=1,              # sufficient for 6 months data
    rnn_type='lstm',             # original TFT choice
    attn_dropout=0.0,            # no attention regularization needed
    grn_activation='ELU',        # smooth, no dead neurons
    dropout=0.1,                 # light VSN regularization
    loss=MAE(),                  # matches RelMAE metric
    max_steps=1500,              # ~2.4 epochs, check if still learning
    learning_rate=1e-3,          # standard Adam starting point
    batch_size=64,               # stable gradients
    scaler_type='robust',        # handles outlier festival days
    hist_exog_list=WEATHER_COLS, # ← weather only (past)
    futr_exog_list=TEMPORAL_COLS,# ← temporal to future (CRITICAL FIX)
    early_stop_patience_steps=-1,# disabled for cross_validation
    val_check_steps=50,          # 30 checks over training
    windows_batch_size=None,     # use all windows (A6000 can handle it)
    random_seed=42,              # reproducibility
)
```

**Key change from current:** Move temporal features from `hist_exog_list` to `futr_exog_list`.
This single change gives the decoder explicit calendar information for future positions and
should noticeably improve predictions.
