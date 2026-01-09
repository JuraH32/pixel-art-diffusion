import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import torchvision
from dataloader import get_dataloader
from model import ScalableUNet, PixelDiffusion

def train(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Logging hyperparameters to TensorBoard
    hparams = {
        "base_channels": args.base_channels,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "save_every": args.save_every,
    }
    writer.add_text("Hyperparameters", str(hparams))

    dataloader, tokenizer = get_dataloader(root_dir=args.data_path, batch_size=args.batch_size, image_size=args.image_size)
    
    # We ensure base_c matches the expected architecture width (128).
    # t_dim is explicitly set to 256 to match the internal logic of Cross-Attention layers.
    model = ScalableUNet(
        num_classes=5, 
        img_size=args.image_size, 
        base_c=args.base_channels,
        t_dim=256
    ).to(device)
    
    diffusion = PixelDiffusion(model, image_size=args.image_size, device=device)
    
    # AdamW with specific Weight Decay prevents label embedding collapse
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        epoch_losses = []
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            loss = diffusion.train_step(images, labels, optimizer, loss_fn)
            epoch_losses.append(loss)
            pbar.set_postfix(loss=f"{sum(epoch_losses)/len(epoch_losses):.4f}")

        # Visualization every 'save_every' epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"model_e{epoch+1}.pt"))
            
            # Generate exactly 5 images: one for each class (1 through 5)
            test_labels = torch.tensor([1, 2, 3, 4, 5]).to(device)
            # High CFG (7.0+) forces the classes to differentiate visually
            samples = diffusion.sample(test_labels, cfg_scale=7.0)
            
            grid = torchvision.utils.make_grid(samples, nrow=5, normalize=False)
            writer.add_image("Validation/ByClass", grid, epoch)

    final_path = os.path.join(args.checkpoint_dir, "final_model_cond.pt")
    torch.save(model.state_dict(), final_path)

    print("Training Complete!")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to your numpy files or images")
    parser.add_argument("--image_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4) # Lower LR for Cross-Attention stability
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--base_channels", type=int, default=128, help="Model width (should be 128 for current UNet structure)")
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    train(args)