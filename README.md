## Training instructions:

### Get dataset
- Download or git clone this repo
- The training dataset consists of Harris' google drive under `dataset/final_data/train` and place in an appropriate place such as is `./train`
- The training requires a separate directory for validation to keeps consistency across sessions (instead of a different random split each session). Download the `dataset/final_data/validate` folder from Harris' google drive and place into `./validate`

### Dataset Migration
- run 
`python -m training.migrate_npz_to_npy --input-dir ./train --output-dir ./Train --cleanup`
`python -m training.migrate_npz_to_npy --input-dir ./validate --output-dir ./Validate --cleanup`

### Environment setup
- Ensure you pip install wandb: 
`export WANDB_API_KEY=<your_api_key>`  # (will supply through whatsapp or email upon request)
`pip install wandb`. Then run
`wandb login`
- Must have `torch`. Other dependencies:
`pip install timm einops`

### Run training
`python -m training.train_multi_gpu --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./Train --validation_data_dir ./Validate --depth 48 --num_heads 25 --rollout_horizon 5`
- You may increase batch size depending on the GPU constrants. Adjust `num_workers` as appropriate. Safe to control+c any time.

### Noop training
`python -m training.train_multi_gpu_noop --num_epochs 40 --batch_size 64 --num_workers 12 --processed_data_dir ./Train --validation_data_dir ./Validate --depth 24 --num_heads 8 --rollout_horizon 1`