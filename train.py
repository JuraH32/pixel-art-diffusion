import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import torchvision

# Import our modules
from dataloader import get_dataloader
from model import ConditionalTinyPixelUNet, PixelDiffusion

def train(args):
    # 1. Setup Environment
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # 2. Prepare Data
    print(f"Loading dataset from {args.data_path}...")
    dataloader, tokenizer = get_dataloader(
        root_dir=args.data_path, 
        batch_size=args.batch_size, 
        image_size=args.image_size
    )
    print(f"Dataset loaded: {len(dataloader.dataset)} images.")
    # Expecting 6 classes (5 real + 1 null)
    print(f"Model initialized for {tokenizer.num_classes} classes (incl. Null).")

    # 3. Initialize Model
    # Note: Tokenizer classes include the real classes + 1 null token.
    model = ConditionalTinyPixelUNet(
        num_classes=tokenizer.num_classes,
        img_size=args.image_size, 
        base_c=args.base_channels
    ).to(device)
    
    diffusion = PixelDiffusion(
        model, 
        image_size=args.image_size, 
        device=device, 
        n_steps=args.n_steps
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # 4. Training Loop
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Train step with random null-label masking for Classifier-Free Guidance
            loss = diffusion.train_step(images, labels, optimizer, loss_fn, unconditional_prob=0.1)
            
            writer.add_scalar("Loss/train", loss, global_step)
            pbar.set_postfix(loss=f"{loss:.4f}")
            epoch_loss += loss
            global_step += 1

        # End of Epoch
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
        
        # 5. Validation / Visualization (Every 5 epochs)
        if (epoch + 1) % args.save_every == 0:
            print(f"Snapshotting at epoch {epoch+1}...")
            
            # Save Checkpoint
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"ckpt_epoch_{epoch+1}.pt"))
            
            # Generate Samples for exactly the 5 classes
            print(f"Generating samples for Classes 1-5...")
            
            # Create a tensor for classes [1, 2, 3, 4, 5]
            # We will generate 2 samples for each class to fill a row? 
            # Let's just do 1 sample per class, total 5 images, or maybe 2 per class = 10 images.
            # Let's do 2 per class to fill a nice grid.
            test_labels = []
            for class_id in range(1, 6): # Classes 1 to 5
                test_labels.extend([class_id] * 2) # Add 2 of each
            
            label_tensor = torch.tensor(test_labels).long().to(device)
            
            # Use CFG Sampling
            # We use steps=20 to ensure it matches the game-time quality
            sampled_images = diffusion.sample_with_guidance(
                label_tensor, 
                steps=20, 
                cfg_scale=3.0, 
                snap_colors=True
            )
            
            # Grid: 5 columns (classes), 2 rows (samples) if we arrange correctly
            # nrow=2 means 2 images per row. 
            # nrow=5 is better: one row per pair, or just one long row.
            grid = torchvision.utils.make_grid(sampled_images, nrow=2, normalize=False)
            writer.add_image("Generated/Class_Samples", grid, epoch)

    # 6. Final Save
    final_path = os.path.join(args.checkpoint_dir, "final_model_cond.pt")
    torch.save(model.state_dict(), final_path)
    writer.close()
    print("Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data", help="Path containing images/ and labels.csv")
    parser.add_argument("--image_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--log_dir", type=str, default="./runs_cond")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_cond")
    
    # Updated default to 5 as requested
    parser.add_argument("--save_every", type=int, default=5, help="Save interval (in epochs)")
    
    args = parser.parse_args()
    train(args)