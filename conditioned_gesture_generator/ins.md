**Instruction for Action-to-Gesture Autoregressive Model**

---

### Model Architecture

1. **Inputs**

   * Action latent sequence: `H ∈ ℝ^{B×T×d_action}`
   * Target gesture token sequence: `Y ∈ ℤ^{B×(T·U)}` where `U=250` (tokens quantized into `n_classes = 3000`)

2. **Action Encoder**

   * Linear projection: `Linear(d_action → d_model)`
   * Positional encoding over action index `t ∈ [0,T)`
   * TransformerEncoder: `num_layers = nl_enc`, `d_model`, `nhead`
   * Output: `E ∈ ℝ^{B×T×d_model}`

3. **Segment-Aware Memory Construction**

   * For each gesture step `ℓ ∈ [0,L)` where `L = T·U`:

     * Action index: `t = ℓ // U`
     * Intra-step index: `u = ℓ % U`
   * Intra-step embedding: `Embedding(U, d_model)`
   * Memory vector: `C_ℓ = Linear([E[:,t,:] ⊕ Emb(u)] → d_model)`
   * Output: `C ∈ ℝ^{B×L×d_model}`

4. **Gesture Decoder (Autoregressive)**

   * Input tokens: previous gesture tokens (teacher-forced during training)
   * Token embedding: `Embedding(n_classes, d_model)`
   * Positional encoding over gesture index `ℓ ∈ [0,L)`
   * TransformerDecoder: `num_layers = nl_dec`, `d_model`, `nhead`, with **causal mask**
   * Cross-attention over memory `C`
   * Output heads:

     * Vocabulary logits: `Linear(d_model → n_classes)` → `p_vocab`
     * Copy-gate: `Linear(d_model → 1)` → `α ∈ (0,1)`

5. **Copy-Kernel Distribution**

   * For each step `ℓ`, build `p_copy(· | y_{ℓ-1})` as a discrete kernel centered on previous token `y_{ℓ-1}`, width `σ`, radius `r`.
   * Final distribution:

     ```
     p(y_ℓ) = α_ℓ * p_vocab + (1 - α_ℓ) * p_copy
     ```

---

### Training Method

1. **Input Preparation**

   * Quantize continuous gesture values to token IDs `[0, n_classes-1]`.
   * Build `y_in` by shifting `Y` right with `<bos>` token at index 0.
   * Use `y_prev = y_in` as previous-token reference for copy-kernel.

2. **Forward Pass**

   * Encode actions → `E`.
   * Build memory `C`.
   * Decode with `y_in` and causal mask → logits, gate.
   * Construct `p_vocab = softmax(logits)`.
   * Construct `p_copy` using `y_prev`.
   * Compute `p_mix = α * p_vocab + (1-α) * p_copy`.

3. **Loss Function**

   * Use negative log-likelihood:

     ```
     L = - (1 / (B·L)) ∑ log p_mix(y_true)
     ```
   * Implement as `NLLLoss` over `log(p_mix)`.
   * No label smoothing.
   * Optional: class weights or focal loss if imbalance is severe.

4. **Optimization**

   * Optimizer: AdamW.
   * Gradient clipping: 1.0.
   * Learning rate schedule: linear warmup, cosine decay.
   * Mixed precision recommended.

---

**End of Instruction**
