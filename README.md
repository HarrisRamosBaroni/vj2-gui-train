Training instructions:
- Download or git clone this repo
- The training dataset is in Kevin's onedrive under vj2gui/train. Place it in an appropriate place. Let us say it is ./train
- The training requires a separate directory for validation. This keeps consistency across sessions instead of a different random split each session. Move 15% of the .npz from the ./trian into a folder ./validate. Time order or any sort is random enough.
- Ensure you pip/conda install wandb: 
`export WANDB_API_KEY=<your_api_key>`
`pip install wandb`
`wandb login`
- run training:
`python vj2ac_train --num_epochs 30 --batch_size 32 --num_workers 12 -t ./trian -v ./validate --save_every_epochs 1`
- (model is small about 1GB so every epoch save is fine).
- You may increase batch size depending on the GPU constrants