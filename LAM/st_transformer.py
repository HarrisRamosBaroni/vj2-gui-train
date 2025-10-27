''' 
Spatial-Temporal Transformer Networks (by Xu et al)
From: https://github.com/researchmm/STTN/blob/master/model/sttn.py
Reference: https://arxiv.org/pdf/2001.02908
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from LAM.spectral_norm import spectral_norm as _spectral_norm


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        channel = 256
        stack_num = 8
        patchsize = [(8, 8), (4, 4), (2, 2), (1, 1)]
        blocks = []
        for _ in range(stack_num):
            blocks.append(TransformerBlock(patchsize, hidden=channel))
        self.transformer = nn.Sequential(*blocks)

        # Use standalone Tokenizer and Detokenizer modules
        self.tokenizer = Tokenizer(in_channels=3, channel=channel)
        self.encoder = self.tokenizer.encoder  # For backward compatibility

        self.detokenizer = Detokenizer(channel=channel, out_channels=3)
        self.decoder = self.detokenizer.decoder  # For backward compatibility

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames, masks):
        # extracting features
        b, t, c, h, w = masked_frames.size()
        masks = masks.view(b*t, 1, h, w)
        enc_feat = self.encoder(masked_frames.view(b*t, c, h, w))
        _, c, h, w = enc_feat.size()
        masks = F.interpolate(masks, scale_factor=1.0/4)
        enc_feat = self.transformer(
            {'x': enc_feat, 'm': masks, 'b': b, 'c': c})['x']
        output = self.decoder(enc_feat)
        output = torch.tanh(output)
        return output

    def infer(self, feat, masks):
        t, c, h, w = masks.size()
        masks = masks.view(t, c, h, w)
        masks = F.interpolate(masks, scale_factor=1.0/4)
        t, c, _, _ = feat.size()
        enc_feat = self.transformer(
            {'x': feat, 'm': masks, 'b': 1, 'c': c})['x']
        return enc_feat


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ###################### Tokenizer & Detokenizer ##############################
# #############################################################################


class Tokenizer(nn.Module):
    """
    CNN Tokenizer with Positional Embeddings - Standalone module extracted from InpaintGenerator.encoder

    Converts spatial-temporal frames into feature tokens and adds spatial + temporal positional embeddings.
    Default architecture matches InpaintGenerator from st_transformer_legacy.py:79-88

    Positional embeddings are ALWAYS added and cannot be disabled.

    Args:
        in_channels: Input channels (default 16 for VVAE latents)
        channel: Output feature dimension (default 256)
        use_random_temporal_pe: Enable random temporal PE offset during training (default False)
        max_pe_offset: Maximum random offset for temporal PE (default 120)

    Input: [B, T, in_channels, H, W] or [B*T, in_channels, H, W] (if B, T provided)
    Output: [B*T, channel, H/4, W/4] (with positional embeddings always included)
    """
    def __init__(self, in_channels=16, channel=256, use_random_temporal_pe=False, max_pe_offset=120):
        super().__init__()
        self.in_channels = in_channels
        self.channel = channel
        self.use_random_temporal_pe = use_random_temporal_pe
        self.max_pe_offset = max_pe_offset

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def sincos_1d_pos_embed(self, L, dim, device, offset=0):
        """
        Generate 1D sinusoidal positional embeddings with optional offset.

        Args:
            L: Sequence length
            dim: Embedding dimension
            device: torch device
            offset: Starting position offset (default 0)

        Returns:
            pe: [L, dim] positional embeddings
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        # Generate positions with offset: [offset, offset+1, ..., offset+L-1]
        pos = torch.arange(offset, offset + L, device=device).float()
        sinus = torch.einsum('p,d->pd', pos, inv_freq)  # [L, dim/2]
        pe = torch.cat([sinus.sin(), sinus.cos()], dim=1)  # [L, dim]
        return pe

    def sincos_2d_pos_embed(self, h, w, dim, device):
        """
        Generate 2D separable sinusoidal positional embeddings.

        Args:
            h: Height
            w: Width
            dim: Embedding dimension
            device: torch device

        Returns:
            pe: [1, dim, h, w] positional embeddings
        """
        assert dim % 2 == 0, \
            f"Channel dimension must be even for 2D positional embeddings, got dim={dim}"
        dim_y = dim // 2
        dim_x = dim - dim_y
        pe_y = self.sincos_1d_pos_embed(h, dim_y, device)  # [h, dim_y]
        pe_x = self.sincos_1d_pos_embed(w, dim_x, device)  # [w, dim_x]
        pe_y = pe_y.view(h, 1, dim_y)  # [h, 1, dy]
        pe_x = pe_x.view(1, w, dim_x)  # [1, w, dx]
        pe = torch.cat([pe_y.expand(h, w, dim_y), pe_x.expand(h, w, dim_x)], dim=2)  # [h, w, dim]
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, dim, h, w]
        return pe

    def add_positional_embeddings(self, tokens, B, T):
        """
        Add spatial and temporal positional embeddings to tokens.

        During training with use_random_temporal_pe=True:
            Temporal PE uses random offset in [0, max_pe_offset]
        During inference or use_random_temporal_pe=False:
            Temporal PE uses standard offset=0

        Spatial PE always uses offset=0 (no randomization).

        Args:
            tokens: [B*T, C, Hp, Wp] - tokenized features
            B: batch size
            T: temporal length

        Returns:
            tokens: [B*T, C, Hp, Wp] - tokens with positional embeddings added
        """
        BT, C, Hp, Wp = tokens.shape
        device = tokens.device

        # Reshape to [B, T, C, Hp, Wp] for PE addition
        tokens = tokens.view(B, T, C, Hp, Wp)

        # Generate spatial positional embeddings (no offset)
        spatial_pe = self.sincos_2d_pos_embed(Hp, Wp, C, device)  # [1, C, Hp, Wp]

        # Generate temporal positional embeddings (with optional random offset)
        if self.use_random_temporal_pe and self.training:
            # Sample random offset in [0, max_pe_offset]
            temporal_offset = torch.randint(0, self.max_pe_offset + 1, (1,), device=device).item()
        else:
            # Standard PE during inference or when disabled
            temporal_offset = 0

        temporal_pe = self.sincos_1d_pos_embed(T, C, device, offset=temporal_offset)
        temporal_pe = temporal_pe.view(1, T, C, 1, 1)  # [1, T, C, 1, 1]

        # Add positional embeddings (broadcasting: spatial over B,T; temporal over B,Hp,Wp)
        tokens = tokens + spatial_pe + temporal_pe

        # Flatten back to [B*T, C, Hp, Wp]
        tokens = tokens.view(BT, C, Hp, Wp)

        return tokens

    def forward(self, x, B=None, T=None):
        """
        Args:
            x: [B, T, in_channels, H, W] or [B*T, in_channels, H, W]
            B: batch size (required if x is [B*T, ...])
            T: temporal length (required if x is [B*T, ...])

        Returns:
            features: [B*T, channel, H/4, W/4] (with positional embeddings always included)
            B: batch size (returned for convenience)
            T: temporal length (returned for convenience)
        """
        # Handle input format
        if x.dim() == 5:  # [B, T, C, H, W]
            B_input, T_input, C, H, W = x.shape
            x = x.view(B_input * T_input, C, H, W)
            if B is None:
                B = B_input
            if T is None:
                T = T_input
        elif x.dim() == 4:  # [B*T, C, H, W]
            assert B is not None and T is not None, \
                "B and T must be provided when input is [B*T, C, H, W]"
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        # Apply CNN encoder
        tokens = self.encoder(x)  # [B*T, channel, H/4, W/4]

        # Add positional embeddings (ALWAYS - cannot be disabled)
        tokens = self.add_positional_embeddings(tokens, B, T)

        return tokens, B, T


class Detokenizer(nn.Module):
    """
    CNN Detokenizer - Standalone module extracted from InpaintGenerator.decoder

    Converts feature tokens back into spatial-temporal frames.
    Default architecture matches InpaintGenerator from st_transformer_legacy.py:91-99

    Args:
        channel: Input feature dimension (default 256)
        out_channels: Output channels (default 16 for VVAE latents)

    Input: [B*T, channel, H/4, W/4]
    Output: [B*T, out_channels, H, W]
    """
    def __init__(self, channel=256, out_channels=16):
        super().__init__()
        self.channel = channel
        self.out_channels = out_channels

        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        """
        Args:
            x: [B*T, channel, H/4, W/4]
        Returns:
            frames: [B*T, out_channels, H, W]
        """
        return self.decoder(x)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, m):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.

    Args:
        patchsize: List of (width, height) tuples for multi-scale patching
        d_model: Model dimension
        temporal_causal_mask: Type of temporal causality
            - 'none': No temporal masking (full attention across time) - default for backward compatibility
            - 'causal': Standard causal mask (position t can only see ≤ t)
            - 'shifted_causal': Shifted causal mask (position t can see ≤ t+1)
    """

    def __init__(self, patchsize, d_model, temporal_causal_mask='none'):
        super().__init__()
        self.patchsize = patchsize
        self.temporal_causal_mask_type = temporal_causal_mask
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def _create_temporal_causal_mask(self, t, out_h, out_w, device):
        """
        Create temporal causal mask for attention.

        Args:
            t: Temporal length
            out_h, out_w: Number of spatial patches
            device: torch device

        Returns:
            mask: [1, t*out_h*out_w, t*out_h*out_w] boolean mask (True = masked out)
        """
        if self.temporal_causal_mask_type == 'none':
            return None

        # Create base temporal mask [t, t]
        if self.temporal_causal_mask_type == 'causal':
            # Standard causal: position t can see positions 0...t
            # torch.triu with diagonal=1 masks out positions > t
            temporal_mask = torch.triu(torch.ones(t, t, device=device), diagonal=1).bool()
        elif self.temporal_causal_mask_type == 'shifted_causal':
            # Shifted causal: position t can see positions 0...t+1
            # torch.triu with diagonal=2 masks out positions > t+1
            temporal_mask = torch.triu(torch.ones(t, t, device=device), diagonal=2).bool()
        else:
            raise ValueError(f"Unknown temporal_causal_mask_type: {self.temporal_causal_mask_type}")

        # Expand temporal mask to spatial-temporal tokens
        # Each temporal position has out_h*out_w spatial tokens
        # Result shape: [t*out_h*out_w, t*out_h*out_w]
        mask = temporal_mask.repeat_interleave(out_h * out_w, dim=0).repeat_interleave(out_h * out_w, dim=1)

        # Add batch dimension for compatibility: [1, t*out_h*out_w, t*out_h*out_w]
        mask = mask.unsqueeze(0)

        return mask

    def forward(self, x, m, b, c, return_attention=False):
        bt, _, h, w = x.size()
        t = bt // b

        # Validate channel dimension divisibility
        assert c % len(self.patchsize) == 0, \
            f"Channel dimension c={c} must be divisible by len(patchsize)={len(self.patchsize)}. " \
            f"Got c % len(patchsize) = {c % len(self.patchsize)}"

        d_k = c // len(self.patchsize)
        output = []
        attention_weights = []  # Collect attention weights if requested
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            # Validate spatial dimension divisibility for this patch size
            assert w % width == 0, \
                f"Spatial width w={w} must be divisible by patch width={width}. " \
                f"Got w % width = {w % width}. " \
                f"Valid patch widths that divide {w}: {[i for i in range(1, w+1) if w % i == 0]}"
            assert h % height == 0, \
                f"Spatial height h={h} must be divisible by patch height={height}. " \
                f"Got h % height = {h % height}. " \
                f"Valid patch heights that divide {h}: {[i for i in range(1, h+1) if h % i == 0]}"

            out_w, out_h = w // width, h // height

            # Process spatial inpainting mask
            mm = m.view(b, t, 1, out_h, height, out_w, width)
            mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, height*width)
            mm_spatial = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t*out_h*out_w, 1)  # [b, t*out_h*out_w, height*width]

            # Create temporal causal mask if needed
            temporal_mask = self._create_temporal_causal_mask(t, out_h, out_w, x.device)  # [1, t*out_h*out_w, t*out_h*out_w] or None

            # Combine masks: spatial mask applies within patches, temporal mask applies across patches
            if temporal_mask is not None:
                # Spatial mask: [b, t*out_h*out_w, height*width] - masks within each patch
                # Temporal mask: [1, t*out_h*out_w, t*out_h*out_w] - masks across patches/time
                # We need to combine them into a single mask: [b, t*out_h*out_w, t*out_h*out_w * height*width]
                # However, the attention is computed as [b, t*out_h*out_w, d_k*height*width] @ [b, t*out_h*out_w, d_k*height*width].T
                # which gives [b, t*out_h*out_w, t*out_h*out_w]
                # So we apply temporal mask at the token level (after spatial aggregation)
                mm = temporal_mask.expand(b, -1, -1)  # [b, t*out_h*out_w, t*out_h*out_w]
            else:
                # No temporal mask, use spatial mask as before
                mm = mm_spatial
            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, attn = self.attention(query, key, value, mm)

            # Store attention weights if requested
            if return_attention:
                attention_weights.append(attn)

            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)

        if return_attention:
            # Return attention weights from all heads (one per patch scale)
            return x, attention_weights
        return x


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection

    Args:
        patchsize: List of (width, height) tuples for multi-scale patching
        hidden: Hidden dimension
        temporal_causal_mask: Type of temporal causality ('none', 'causal', 'shifted_causal')
    """

    def __init__(self, patchsize, hidden=128, temporal_causal_mask='none'):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden, temporal_causal_mask=temporal_causal_mask)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x, return_attention=False):
        x_data, m, b, c = x['x'], x['m'], x['b'], x['c']

        if return_attention:
            attn_out, attn_weights = self.attention(x_data, m, b, c, return_attention=True)
            x_data = x_data + attn_out
            x_data = x_data + self.feed_forward(x_data)
            return {'x': x_data, 'm': m, 'b': b, 'c': c, 'attention': attn_weights}
        else:
            x_data = x_data + self.attention(x_data, m, b, c, return_attention=False)
            x_data = x_data + self.feed_forward(x_data)
            return {'x': x_data, 'm': m, 'b': b, 'c': c}


# ######################################################################
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 64

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=nf*1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf*1, nf*2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)  # B, C, T, H, W
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
