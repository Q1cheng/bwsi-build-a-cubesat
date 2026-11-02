import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Dataset import TerrainDataset
from Models import UNetSmall


def set_seed(seed=67):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_batch(batch):
    xs, ys, paths = zip(*batch)
    xbatch = torch.stack(xs, dim=0)
    ybatch = torch.stack(ys, dim=0)
    return xbatch, ybatch, list(paths)


def save_checkpoint(model, optimizer, epoch, path):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        },
        path
    )


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for xbatch, ybatch, _ in loader:
        xbatch = xbatch.to(device)
        ybatch = ybatch.to(device)

        optimizer.zero_grad()

        pred = model(xbatch)
        loss = F.mse_loss(pred, ybatch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    for xbatch, ybatch, _ in dataloader:
        xbatch = xbatch.to(device)
        ybatch = ybatch.to(device)

        pred = model(xbatch)
        loss = F.mse_loss(pred, ybatch)

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    set_seed(67)

    BASE_DIR = os.path.expanduser("~/Desktop/DataControl")
    ML_DIR = os.path.join(BASE_DIR, "ML")

    TRAIN_FOLDER = os.path.join(ML_DIR, "Data")
    VAL_FOLDER = os.path.join(ML_DIR, "Validation")

    WEIGHTS_DIR = os.path.join(ML_DIR, "weights")
    FINAL_MODEL_PATH = os.path.join(ML_DIR, "FinalWeights.pt")
    BEST_MODEL_PATH = os.path.join(ML_DIR, "BestWeights.pt")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    train_image_paths = [
        os.path.join(TRAIN_FOLDER, f)
        for f in os.listdir(TRAIN_FOLDER)
        if f.lower().endswith(".png")
    ]

    val_image_paths = [
        os.path.join(VAL_FOLDER, f)
        for f in os.listdir(VAL_FOLDER)
        if f.lower().endswith(".png")
    ]

    train_dataset = TerrainDataset(
        image_paths=train_image_paths,
        train=True,
        crop_size=(2048, 2048),
        blur_probability=0.5,
    )

    val_dataset = TerrainDataset(
        image_paths=val_image_paths,
        train=False,
        crop_size=(2048, 2048),
        blur_probability=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = UNetSmall(
        in_channels=5,
        out_channels=3,
        base_channels=16,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-3)

    num_epochs = 2000
    save_every = 1
    best_val_loss = float(1500)
    start_epoch = 1

    # LOAD OLD WEIGHTS / RESUME TRAINING
    if os.path.exists(BEST_MODEL_PATH):
        start_epoch = load_checkpoint(model, optimizer, BEST_MODEL_PATH, device)
        print(f"Loaded checkpoint from {BEST_MODEL_PATH}")
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found, training from scratch.")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss = {train_loss:.6f} | "
            f"val_loss = {val_loss:.6f}"
        )

        if epoch % save_every == 0:
            checkpoint_path = os.path.join(WEIGHTS_DIR, f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                checkpoint_path,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                BEST_MODEL_PATH,
            )

    save_checkpoint(model, optimizer, num_epochs, FINAL_MODEL_PATH)
    print("Training complete.")


if __name__ == "__main__":
    main()