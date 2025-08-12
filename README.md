Training instructions:
- Download or git clone this repo
- The training dataset consists of Harri's google drive under `final_data/train` and place in an appropriate place. Let us say it is ./Train
- The training requires a separate directory for validation. This keeps consistency across sessions instead of a different random split each session. Download the `final_data/validate` folder from Harris' google drive.
- Ensure you pip install wandb: 
`export WANDB_API_KEY=<your_api_key>`  # (will supply through whatsapp or email upon request)
`pip install wandb`. Then run
`wandb login`
- Must have `torch`. Other dependencies:
`pip install timm einops`
- run training:
`python -m training.train_multi_gpu --num_epochs 30 --batch_size 32 --num_workers 12 --processed_data_dir ./Train --validation_data_dir ./Validate`
- You may increase batch size depending on the GPU constrants. Adjust `num_workers` as appropriate. Safe to control+c any time.
