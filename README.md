Training instructions:
- Download or git clone this repo
- The training dataset consists of Harri's onedrive under `train_v2` and place in an appropriate place. Let us say it is ./train_v2
- The training requires a separate directory for validation. This keeps consistency across sessions instead of a different random split each session. Download the `validate_v2` folder from Harris' onedrive.
- Ensure you pip/conda install wandb: 
`export WANDB_API_KEY=<your_api_key>`  # (will supply through whatsapp or email upon request)
`pip install wandb`
`wandb login`
- Must have `torch`. Other dependencies:
`pip install timm einops`
- run training:
`python vj2ac_train --num_epochs 30 --batch_size 32 --num_workers 12 -t ./train_v2 -v ./validate_v2 --save_every_epochs 1`
- (model is small about 1GB so every epoch save is fine).
- You may increase batch size depending on the GPU constrants. Adjust `num_workers` as appropriate. Safe to control+c any time.
