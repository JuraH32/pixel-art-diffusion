import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Basic Building Blocks ---

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.act(self.bn(h))
        # Add time/label embedding
        time_emb = self.act(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.transform(h)
        h = self.act(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1])

# --- 2. Conditional Tiny U-Net ---

class ConditionalTinyPixelUNet(nn.Module):
    def __init__(self, num_classes, img_size=16, c_in=3, c_out=3, base_c=32, time_dim=128):
        super().__init__()
        self.img_size = img_size
        
        # 1. Label Embedding (Maps class ID to a vector)
        # num_classes + 1 covers the actual classes + the "null/unconditional" token
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)
        
        # Time Embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Initial Projection
        self.conv_in = nn.Conv2d(c_in, base_c, 3, padding=1)
        
        # Downsampling
        self.down1 = Block(base_c, base_c * 2, time_dim)
        self.pool1 = nn.Conv2d(base_c * 2, base_c * 2, 4, 2, 1)
        
        self.down2 = Block(base_c * 2, base_c * 4, time_dim)
        self.pool2 = nn.Conv2d(base_c * 4, base_c * 4, 4, 2, 1)
        
        # Bottleneck
        self.bot1 = Block(base_c * 4, base_c * 4, time_dim)
        self.attn = SelfAttention(base_c * 4)
        self.bot2 = Block(base_c * 4, base_c * 4, time_dim)
        
        # Upsampling
        self.up1 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 4, 2, 1)
        self.up_block1 = Block(base_c * 4 + base_c * 2, base_c * 2, time_dim)
        
        self.up2 = nn.ConvTranspose2d(base_c * 2, base_c, 4, 2, 1)
        self.up_block2 = Block(base_c * 2 + base_c, base_c, time_dim)
        
        self.conv_out = nn.Conv2d(base_c, c_out, 3, padding=1)

    def forward(self, x, t, labels):
        # Embed Time
        t_emb = self.time_mlp(t)
        
        # Embed Label and Add to Time
        # This is a lightweight way to condition the entire network
        l_emb = self.label_emb(labels)
        
        # Combine (Simple addition works surprisingly well for small models)
        # t_emb becomes the carrier for both "how noisy is it?" and "what object is it?"
        cond_emb = t_emb + l_emb
        
        x = self.conv_in(x)
        
        x1 = self.down1(x, cond_emb)
        x_down = self.pool1(x1)
        
        x2 = self.down2(x_down, cond_emb)
        x_down = self.pool2(x2)
        
        x_bot = self.bot1(x_down, cond_emb)
        x_bot = self.attn(x_bot)
        x_bot = self.bot2(x_bot, cond_emb)
        
        x_up = self.up1(x_bot)
        if x_up.shape != x2.shape: x_up = F.interpolate(x_up, size=x2.shape[2:])
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.up_block1(x_up, cond_emb)
        
        x_up = self.up2(x_up)
        if x_up.shape != x1.shape: x_up = F.interpolate(x_up, size=x1.shape[2:])
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.up_block2(x_up, cond_emb)
        
        return self.conv_out(x_up)

# --- 3. Diffusion Manager ---

class PixelDiffusion:
    def __init__(self, model, image_size=16, device="cpu", n_steps=1000):
        self.model = model.to(device)
        self.image_size = image_size
        self.device = device
        self.n_steps = n_steps
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.n_steps, size=(n,)).to(self.device)

    def train_step(self, images, labels, optimizer, loss_fn, unconditional_prob=0.1):
        """
        Train with Random Null-Label Masking for CFG.
        """
        self.model.train()
        n = images.shape[0]
        t = self.sample_timesteps(n)
        x_t, noise = self.noise_images(images, t)
        
        # Randomly mask labels with 0 (null token)
        # This teaches the model to generate even without a specific prompt
        if torch.rand(1).item() < unconditional_prob:
            labels = torch.zeros_like(labels).to(self.device)
            
        predicted_noise = self.model(x_t, t, labels)
        loss = loss_fn(noise, predicted_noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
        
    def snap_to_palette(self, images, levels=16):
        return (images * (levels - 1)).round() / (levels - 1)

    @torch.no_grad()
    def sample_with_guidance(self, labels, steps=20, cfg_scale=3.0, snap_colors=True):
        """
        Classifier-Free Guidance Sampling.
        labels: Tensor of class IDs [N]
        cfg_scale: How hard to force the prompt (3.0 is usually good for pixel art)
        """
        self.model.eval()

        labels = labels.to(device=self.device, dtype=torch.long)
        n = labels.shape[0]

        x = torch.randn((n, 3, self.image_size, self.image_size), device=self.device)

        full_steps = torch.linspace(
            0, self.n_steps - 1, steps + 1, device=self.device
        ).round().to(dtype=torch.long)
        full_steps = torch.flip(full_steps, dims=[0])

        for i in range(steps):
            t_now = int(full_steps[i].item())
            t_next = int(full_steps[i + 1].item())

            t_batch = torch.full((n,), t_now, device=self.device, dtype=torch.long)

            # Combine inputs to batch them together for speed
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            l_in = torch.cat([labels, torch.zeros_like(labels)], dim=0)  # [Label, Null]

            noise_pred = self.model(x_in, t_in, l_in)
            noise_cond, noise_uncond = noise_pred.chunk(2)

            final_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            alpha_now = self.alpha_hat[t_now]
            alpha_next = self.alpha_hat[t_next] if t_next >= 0 else torch.tensor(1.0, device=self.device)

            x0_pred = (x - torch.sqrt(1 - alpha_now) * final_noise) / torch.sqrt(alpha_now)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * final_noise

        x = (x.clamp(-1, 1) + 1) / 2
        if snap_colors:
            x = self.snap_to_palette(x)
        return x