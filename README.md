Training instructions:
- Download or git clone this repo
- The training dataset consists of Kevin's onedrive under `vj2gui/train` as well as Harris's onedrive folder also shared with you called `train`. Download contents and place all .npz under one folder in an appropriate place. Let us say it is ./Train
- The training requires a separate directory for validation. This keeps consistency across sessions instead of a different random split each session. Download the `Validate` folder from Harris' onedrive.
- Ensure you pip/conda install wandb: 
`export WANDB_API_KEY=<your_api_key>`  # (will supply through whatsapp or email upon request)
`pip install wandb`
`wandb login`
- Must have `torch`. Other dependencies:
`pip install timm einops`
- run training:
`python vj2ac_train.py --num_epochs 30 --batch_size 32 --num_workers 12 -t ./Train -v ./Validate --save_every_epochs 1`
- (model is small about 1GB so every epoch save is fine).
- You may increase batch size depending on the GPU constrants. Adjust `num_workers` as appropriate. Safe to control+c any time.
