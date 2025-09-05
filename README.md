## Training instructions:

### Get dataset
- Download or git clone this repo
- The entire dataset consists of Harris' google drive under `dataset/final_data/mother` and place in an appropriate place such as is `./mother`

`rclone copy -P --transfers 8 --checkers 8 --drive-chunk-size 128M --fast-list <remote name>:vj2gui/dataset/final_data/mother ./mother`

### Dataset split
Run:
`python preprocess/generate_split_manifest.py --data-dir ./mother --output-dir ./mother/manifests --name experiment_A`

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
`pip install timm einops h5py plotly matplotlib scipy`

## Run training
Must choose a model with slug in the registry. Choose between vanilla, cross_attention and film.
```
MODEL_REGISTRY = {
    "vanilla": VJ2GUIPredictor,
    "cross_attention": VJ2GUIPredictorCrossAttention,
    "film": VJ2GUIPredictorFiLM,
}
```
### Vanilla
`python -m training.train_v2 --model_type vanilla --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./mother --manifest ./mother/manifests/experiment_A.json --depth 24 --num_heads 8 --rollout_horizon 2`

You may increase batch size depending on the GPU constrants. Adjust `num_workers` as appropriate. Safe to control+c any time.

### Cross attention training
`python -m training.train_v2 --model_type cross_attention --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./mother --manifest ./mother/manifests/experiment_A.json --depth 24 --num_heads 8 --rollout_horizon 2`

### FiLM training
`python -m training.train_v2 --model_type film --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./mother --manifest ./mother/manifests/experiment_A.json --depth 18 --num_heads 8 --rollout_horizon 2 --film_layers_modulated 6`



### Noop training
`python -m training.train_multi_gpu_noop --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./mother --manifest ./mother/manifests/experiment_A.json --depth 24 --num_heads 8 --rollout_horizon 2`
