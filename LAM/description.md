# Requirement

# Model Design Summary
Mathematically describing the model, we have three parts.

- Action Encoder: $E_{\alpha} : \tau_0^t\rightarrow a_0^t$
- World Encoder: $E_{\beta}: \tau_0^T \rightarrow h_{\text{world}}$
- Dynamics Predictor: $\text{P}_{\phi}:(\tau_0^t, a_0^t,h_{\text{world}}) \rightarrow \tau_{t+1}$

Coming down for more specific details.

## Action Encoder: $z_0^{t+1}\rightarrow z_0^{t}$

- ST-Transformer (A).
- Shifted Temporal Causal Masking (i.e. Diagonal = 2) to see one step ahead. The action encoder needs to see present and future together to summarize a transition token i.e. action.
- Mean-pooling spatially to obtain a finite codebook over time.
- Input shape: [B, t, n_patches, d_patches] (states)
- Output shape: [B, t-1, d_code_a]
- Output obtained via RVQ.
  - With L_a many RVQ level, and takes in a tuple of (l1, ..., l_{L_a}) for Codebook size per RVQ level.
  - codebook_sizes_a=(12, 64, 256) by default. (Tho a better option is up for testing)
- Number of ST-Transformers: 3 layers by default.

## World Encoder: ${z}_0^T \rightarrow h_{\text{world}}$

- ST-Transformer (H).
- No Causal Masking, bi-directional.
- Mean-pooling spatially and temporally to obtain a finite codebook.
- Input shape: [B, t, n_patches, d_patches] (states)
- Output shape: [B, d_code_h]
- Output $h$ (hypothesis) obtained via RVQ.
  - With L_h many RVQ level, and takes in a tuple of (l1, ..., l_{L_h}) for Codebook size per RVQ level.
  - codebook_sizes_h=(12, 24, 48, 256, 256, 256), (Tho a better option is up for testing)
- Number of ST-Transformers: 3 layers by default.

## Dynamics Predictor: $(\{z\}_0^t, \{a\}_0^t, h_{\text{world}}) \rightarrow z_{t+1}$

- CNN Tokenizer (X) + ST-Transformer (P) + CNN De-tokenizer (X).
- Input sources:
  - State: [B, t-1, n_patches, d_patches] (need to drop [-1])
  - Action: [B, t-1, n_code_a * d_code_a] (already output this shape)
  - Hypothesis: [B, t-1, n_code_h, d_code_h] (duplicates of the same over time)
- Action and Hypothesis goes through a linear transformation and  are added directly on top of each state patches through broadcasting. So ST-Transformer (P) handles [B, t-1, N_patches, d_patches] as input.
- Temporal Causal Masking (i.e. Diagonal = 1) to see only the past.
- Output shape: [B, t-1, N_patches, d_patches]. Model predicts one step ahead. i.e. Output[t] = Input[t-1];  
- Teacher forcing: Model takes in GT[0:t-1] of temporal shape [t-1], output PRED[1:t] of temporal shape [t-1]. i.e. [A,B,C] in, [B,C,D] out, and compare loss with GT[1:t].
- Number of ST-Transformers: 3 layers by default.

The CNN Tokenizer (X): $o_t\rightarrow z_t$

- Input [B, t, C=16, w=64, h=64] (reshaped to [B*t, C=16, w=64, h=64] internally)
- Output [B*t, d_model, w/4=16, h/4=16] (spatial feature maps with merged batch-time dimension)

The CNN De-Tokenizer (X): $z_t\rightarrow o_t$

- Input [B*t, d_model, w/4=16, h/4=16] (spatial feature maps with merged batch-time dimension)
- Output [B*t, C=16, w=64, h=64] (reshaped to [B, t, C=16, w=64, h=64] as needed)

## Overall Note:

- CNN Tokenizer is shared among the three models, i.e. the same model with the same weight is used. You only need to pass the entire sequence through it once.
- The CNN Tokenizer outputs spatial feature maps with merged batch-time dimension [B*t, d_model, H', W']. The ST-Transformers receive this format along with the batch size 'b' as metadata, which allows them to internally separate and perform temporal-spatial attention.
- Input to the CNN Tokenizer are output (preprocessed) from Video VAE in h5 format of shape [B, t, C=16, w=64, h=64] in the range of [-1,1]. No normalization is needed in our own code.
- Each model has their own ST-Transformer.
- Spatial Positional Embedding, and Temporal Positional Embedding, should be added after CNN tokenization, before the sequence is passed into any ST-Transformers (World Encoder, Action Encoder, and Dynamics Predictor). Use Sinusoidal PE.
- Action token and World Token should be added together (**Additive**), broadcasted over the visual tokens before feeding into the dynamics predictor. Two linear matrix $W_a^z$ and $W_h^z$ is used to transform action into dimension for latent state, and having world embedding transformed into the dimension of latent state.
- The ST-Transformer and CNN Tokenizer/De-Tokenizer can both be found in `st_transformer.py` . We must implement the RVQ Process oursleves. 

## Loss Function:

- Action Encoder RVQ Loss: Commitment. (Only affects action encoder)
- World Encoder RVQ Loss: Commitment. (Only affects world encoder)
- Prediction & Rollout MSE Loss. (Propagate from CNN De-tokenizer, to Every Model)

$$
L = J_{RVQ}(\alpha) + J_{RVQ}(\beta) + J_{TF}(\phi,\alpha,\beta)
$$

Specifically:
$$
\begin{aligned}
L_{\text{total}}
&= \underbrace{\sum_{t}\big\|\hat{\tau}_{t+1} - \tau_{t+1}\big\|_2^2}_{\text{Reconstruction / Rollout Loss}} \\[4pt]
&\quad + \underbrace{\beta_a \sum_i \big\| z^{(a)}_{e,i} - \operatorname{sg}[e^{(a)}_i] \big\|_2^2}_{\text{Action Encoder Commitment Loss}} \\[4pt]
&\quad + \underbrace{\beta_h \sum_j \big\| z^{(h)}_{e,j} - \operatorname{sg}[e^{(h)}_j] \big\|_2^2}_{\text{World Encoder Commitment Loss}}
\end{aligned}
$$

Note, the codebook is updated via EMA (see EMA section).

# Training:

- Random masking of frame tokens post tokenizer before the dynamics model and world encoder, probability default to 10%, using Bernoulli distribution. This should not be applied to the action encoder.
- Predictor uses previous prediction as context (input), to simulate rollout, replacing [1,T] each time. Simulated rollout for 2 steps by default. Adding up the loss with teacher forcing using weight (TF, Roll 1, Roll 2) = (1, 0.8, 0.5).
- Temporal Causal Masking is used for both dynamics predictor and action encoder (shifted) to train under teacher forcing without sequential iteration (efficiency). 
- Both RVQ Encoder (Action and World) uses Straight through and EMA for their codebook update.
- Both RVQ Encoder should use a linear EMA scheduler, we monitor dead codes, and use encoder output to replace them.
- The training data consists of h5 files, each of shape [T, C=16, W=64, H=64]
- Dataloader: Should choose a window from a h5 file for each batch element. However, each batch should sample window (sequence) from a variety of h5 files, not sliding window from the same.
- **Positional Embedding**: 
  - Our Tokenizer adds PE for us, so even in rollout we have to go by the tokenizer and detokeniser.
  - To avoid exposure bias, we randomly sample starting point of the temporal PE and train model with it, instead of using a fixed starting point of 0. This helps the model to generalize beyond it's trained context window.


### EMA

$$
\begin{aligned}
N^{(l)}_k &\leftarrow \gamma\, N^{(l)}_k + (1 - \gamma)\, n^{(l)}_k \\[4pt]
M^{(l)}_k &\leftarrow \gamma\, M^{(l)}_k + (1 - \gamma)\, \sum_{x_i \in \mathcal{A}^{(l)}_k} z_{e,i}^{(l)} \\[6pt]
e^{(l)}_k &\leftarrow \frac{M^{(l)}_k}{N^{(l)}_k + \varepsilon}
\end{aligned}
$$

$$
\text{where: }
\begin{cases}
\gamma & \text{is the EMA decay rate (e.g. 0.99)}\\
n^{(l)}_k & \text{is the number of encoder vectors assigned to code } k \text{ at level } l\\
\mathcal{A}^{(l)}_k & \text{is the set of encoder vectors assigned to code } k\\
M^{(l)}_k,\, N^{(l)}_k & \text{are running EMA accumulators for sums and counts}\\
\varepsilon & \text{is a small constant for numerical stability.}
\end{cases}
$$



### Free Running (Rollout)

During free running in training, we do:

1. $S_{k+1} = F(S_k)$  where $S_k$ is the previous sequence output, with $S_k[0] = GT[0]$ for all t.
2. We then concatenate $GT[0] = S_k[0]$ onto $S_{k+1}$ to form a full sequence once again, post teacher forcing.
3. Then, we repeate the above step.

If we take a matrix $S \in \mathbb R^{K\times T}$ where $K$ is the maximum iteration. The lower triangle of it would be redudent to the upper-triangle. Here is a simple demo:

- $S_1[0] = GT[0]$, and due to causal masking $S_1[1] = F(GT[0])$.
- $S_2[1] = F(S_1[0]) = F(GT[0]) = S_1[1]$. Hence, model didn't predict anything new here.
- We can form an inductive hypothesis that at $ k $, we have $S_{k}[i] = S_{k-1}[i]$ for all $i \lt k$ (lower triangular without the main diagnal). 
- Then, $S_{k+1}[i+1] = F(S_k[0],...,S_k[i]) = F(GT[0], S_{k-1}[1],...,S_{k-1}[i]) = S_{k}[i+1]$ 
- We have the initial condition, and inductive step proven. Therefore the claim $\forall i\le k:S_k[i] = S_{k-1}[i]$ is true.

**Training trick**:

- During the computation of training/val running rollout MSE recon loss, apply a mask from 0 to k-1 (inclusive), for each k iteration rollout. (Note, this is not autoregressive MSE recon loss)
- Note, we consider the first step of rollout to be the teacher forcing.




## Overfit test in `overfit.py`
- Use the same data loading method as training: h5 and manifest.
- Take a single datapoint from the dataset.
- Log all the losses.
- -steps default to 1000, we run over one dataset with steps many times to see if the model overfits.

Generation Process:

- We record a dictionary of codes for the world codebook as well as the action codebook. These are used for generation.
- We can then use only the Predictor, giving it random sample from the action dictionary, and a fixed world hypothesis. Starting with one image, following multiple steps of predictions.
- Then, passing those results into the video decoder to see the full video.

# Monitoring

## General Training and Validation Loss Monitoring Candidates

- TF Loss, must be monitored for both train and validation.
- Rollout loss 1 step. This is done using context replacement, it is performed every batch.
- Rollout loss 2 step. This is done using context replacement to, performed every batch.
- Action Encoder RVQ Commitment loss.
- Action Encoder RVQ Codebook loss.
- World Encoder RVQ Commitment loss.
- World Encoder RVQ Codebook loss.
- Total Loss.

Monitoring frequency and datasize:

- We run 4 validation runs per epoch at 25% points.
- Each time, we use val_size_percent of the validation set. This value can be from 0 to 1. 0 for no validation, 1 for full validation set.
- All monitoring should be done for both training and validation.

## Action Encoder Diagnal Attention

We compute diagnalization by adding up the main diagnal and second diagnal of all attentions heads of the Action Encoder. And divided by the number of main diagnal and second diagnals. If the matrix is fully focusing on the main and second diagnal, this value with be 1. 

We plot a wandb native distribution graph for each transformer block.

## Codebook Usage

Among the N_code we have, in each batch, how many are used?
Returns int

log on wandb native distribution: height is N_code used / N_code total over steps per Layer, x_axis is steps as usual, y_axis (cateogry for distribution plot) is ratio of code used.

This should be done for both World Encoder, and Action Encoder.

## Codebook diversity

Measure code to code similarity matrix, using MSE between two codes. 

- Log Min and Maximum similarity score, and mean similarity score for each codebook.

## Action Sensitivity

dPSNR = PSNR_seq - PSNR_rand
$$
\Delta_t \mathrm{PSNR}
= \frac{1}{B} \sum_{b=1}^{B}
\left[
10 \log_{10} \!\left( \frac{1}{\mathrm{MSE}(x_{b,t}, \hat{x}_{b,t})} \right)
-
10 \log_{10} \!\left( \frac{1}{\mathrm{MSE}(x_{b,t}, \hat{x}'_{b,t})} \right)
\right]
$$
**Where:**

- \( B \): batch size (number of video samples)
- \( t \): evaluation frame index (Genie uses \( t = 4 \))
- \( x_{b,t} \): ground-truth frame for sample \( b \) at time \( t \)
- \( \hat{x}_{b,t} \): frame predicted using **inferred latent actions** from ground truth
- \( \hat{x}'_{b,t} \): frame predicted using **random latent actions** sampled from a categorical distribution
- \( \mathrm{MSE}(x, \hat{x}) \): mean-squared-error between pixel intensities of two frames, typically over \( C,H,W \)
- \( \Delta_t \mathrm{PSNR} \): the **controllability score** — how much the correct actions affect the predicted frame quality (larger = more action-sensitive)

log all three: dPSNR, PSNR_seq and PSNR_rand during validation, under Val_Action_Encoder/

## World Sensitivity

dPSNR = PSNR_seq - PSNR_rand
$$
\Delta_t \mathrm{PSNR}
= \frac{1}{B} \sum_{b=1}^{B}
\left[
10 \log_{10} \!\left( \frac{1}{\mathrm{MSE}(x_{b,t}, \hat{x}_{b,t})} \right)
-
10 \log_{10} \!\left( \frac{1}{\mathrm{MSE}(x_{b,t}, \hat{x}'_{b,t})} \right)
\right]
$$
**Where:**

- \( B \): batch size (number of video samples)
- \( t \): evaluation frame index (Genie uses \( t = 4 \))
- \( x_{b,t} \): ground-truth frame for sample \( b \) at time \( t \)
- \( \hat{x}_{b,t} \): frame predicted using **inferred world embedding** from ground truth
- \( \hat{x}'_{b,t} \): frame predicted using **random world embedding** sampled from a categorical distribution
- \( \mathrm{MSE}(x, \hat{x}) \): mean-squared-error between pixel intensities of two frames, typically over \( C,H,W \)
- \( \Delta_t \mathrm{PSNR} \): the **controllability score** — how much the correct actions affect the predicted frame quality (larger = more action-sensitive)

log all three: dPSNR, PSNR_seq and PSNR_rand during validation, under Val_World_Encoder/



## TFLOPs of Compute

We also record TFLOPS of compute used and log it, so we can later create plot of Val Loss against TFLOPS to monitor scaling.

## Logging Wandb Format:

We split with:

- Train_Total/
- Train_Dynamics_Predictor/
- Train_World_Encoder/
- Train_Action_Encoder/
- Val_Total/
- Val_Dynamics_Predictor/
- Val_World_Encoder/
- Val_Action_Encoder/

---

# Implementation Dependencies

` st_transformer.py` :

- Tokenizer
- Detokenizer
- TransformerBlock

`world_model.py`:

- ResidualVectorQuantizer
- ActionEncoder
  - wraps: TransformerBlock (shifted_causal) x3 by default
  - uses: RVQ (with Layered Codebook for Actions)
  - receives: tokenized input from LAM

WorldEncoder:

- wraps: TransformerBlock (none) x3 by default
- uses: RVQ (with Layered Codebook for World)
- receives: pre-tokenized input from LAM

DynamicsPredictor:

- wraps: TransformerBlock (causal) x3 by default
- receives: tokenized input from LAM

World Model:

- owns: Tokenizer, Detokenizer
- owns: ActionEncoder, WorldEncoder, DynamicsPredictor
- calls tokenizer ONCE
- distributes tokens to all three components
- calls detokenizer ONCE (at the end)



---

# Questions remain

**World Token Information bottleneck**: Are we really providing sufficient information? what about longer sequences or more details in a short sequence?

**World Token as alternative to stochastic sampling**: is this really good enough? 

**World Token tricks**: We used world token as another additive embedding. Are there other tricks done by the original paper to train this world encoder? And are there better ways to fuse the information?

- They had a three phase training: No world token -> world token without KL -> with KL. Says this stablizes the training and helps the model learn to use the world token.
- I don't know if my model is currently even world token sensitive. Gotta add that.

