## Training instructions:

### Get dataset and split
- Download or git clone this repo
- The entire dataset consists of Harris' google drive under `vvae_embeddings/mother` and place in an appropriate place such as is `./mother`

`./training/get_subset_h5.sh <remote name>:vvae_embeddings/mother ./mother all`

### Environment setup
- Will supply `WANDB_API_KEY` through whatsapp or email upon request.
- Ensure you pip install wandb: 
```
export WANDB_API_KEY=<your_api_key>
pip install wandb
wandb login
```
- Must have `torch`. Install with `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` or other appropriate version given the hardware. 
- Other dependencies:
`pip install timm einops h5py plotly matplotlib scipy opencv-python psutil omegaconf decord av taming-transformers`

### Run training
Choose `<number of devices>` and proceed with command:

`PYTHONPATH=. torchrun --nproc_per_node=<number of devices> latent_action_model/train_vvae_lam.py --data_dir ./mother --manifest_path ./mother/manifests/experiment_A.json --num_epochs 100 --batch_size 7 --context_length 8 --mixed_precision --use_wandb --project_name vvae-lam --validation_fraction 0.1`

---

## LEGACY: DO NOT MIND BELOW
### Run training
Must choose a model with slug in the registry. Choose between vanilla, cross_attention and film.
```
MODEL_REGISTRY = {
    "vanilla": VJ2GUIPredictor,
    "cross_attention": VJ2GUIPredictorCrossAttention,
    "film": VJ2GUIPredictorFiLM,
}
```
#### Vanilla
`python -m training.train_v2 --model_type vanilla --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./mother --manifest ./mother/manifests/experiment_A.json --depth 24 --num_heads 8 --rollout_horizon 2`

You may increase batch size depending on the GPU constrants. Adjust `num_workers` as appropriate. Safe to control+c any time.

#### Cross attention training
`python -m training.train_v2 --model_type cross_attention --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./mother --manifest ./mother/manifests/experiment_A.json --depth 24 --num_heads 8 --rollout_horizon 2`

#### FiLM training
`python -m training.train_v2 --model_type film --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./mother --manifest ./mother/manifests/experiment_A.json --depth 18 --num_heads 8 --rollout_horizon 2 --film_layers_modulated 6`

#### Noop training
`python -m training.train_multi_gpu_noop --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./mother --manifest ./mother/manifests/experiment_A.json --depth 24 --num_heads 8 --rollout_horizon 2`

#### Latent action model training

`python -m latent_action_model.training --data_dir ./mother --manifest_path ./mother/manifests/experiment_A.json --num_epochs 60 --batch_size 80 --min_context 3 --max_context 5 --encoder_depth 16 --decoder_depth 16 --encoder_heads 16 --decoder_heads 16 --embed_dim 512 --context_schedule linear --learning_rate 1e-4 --kl_weight 0.0005 --num_workers 6  --detach_rollout_first --rollout_prob 1`