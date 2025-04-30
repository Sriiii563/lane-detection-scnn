import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from datasets.culane import CULaneDataset
from models.scnn import SCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset and Dataloader
    train_dataset = CULaneDataset(config.train_list, config)
    val_dataset = CULaneDataset(config.val_list, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 2. Model
    model = SCNN(backbone=config.backbone, num_classes=config.num_classes).to(device)

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        print(f"\nEpoch [{epoch+1}/{config.epochs}]")

        # ----- TRAIN -----
        model.train()
        total_train_loss = 0

        for images, masks in tqdm(train_loader, desc="Training"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ----- VALIDATION -----
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ----- Save best model -----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(config.save_dir, exist_ok=True)
            save_path = os.path.join(config.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")

if __name__ == "__main__":
    train()
