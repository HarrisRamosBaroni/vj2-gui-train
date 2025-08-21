
# training/utils.py

import torch
from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

# Matches V-JEPA2 training setup (256x256, normalized to [-1, 1])
# _preprocess = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # assumes RGB image
# ])
_preprocess = transforms.Compose([
transforms.Lambda(lambda x: x.permute(2, 0, 1)       # HWC ➜ CHW
                         .float()
                         .div(255)),                 # 0-255 ➜ 0-1
transforms.Resize((256, 256)),
transforms.Normalize(mean=[0.5, 0.5, 0.5],
                     std=[0.5, 0.5, 0.5]),

])
def encode_frame_from_adb(pil_img, encoder):
    if pil_img is None:
        return None
    device = encoder.device if hasattr(encoder, "device") else next(encoder.parameters()).device
    x = _preprocess(pil_img).unsqueeze(0).unsqueeze(1).to(device)  # shape: [1, 1, 3, 256, 256]
    with torch.no_grad():
        return encoder(x)  # returns [1, D] or [1, T, D]

def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = []

    if encoder is not None:
        param_groups.extend([
            {
                "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1) and p.requires_grad),
                "lr_scale": enc_lr_scale,
            },
            {
                "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1) and p.requires_grad),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
                "lr_scale": enc_lr_scale,
            },
        ])
    
    param_groups.extend([
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ])

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    return optimizer,scheduler, wd_scheduler
    
