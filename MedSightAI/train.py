import os
import torch
import numpy as np
from tqdm import tqdm  # Thư viện tạo thanh tiến trình (progress bar)
from torch.utils.data import DataLoader

# Import các module chúng ta đã viết
from src.dataset import TBX11KDataset
from src.model import MedicalConceptModel
from src.loss import MedicalLoss

# --- CẤU HÌNH (CONFIG) ---
CONFIG = {
    "root_dir": "/kaggle/input/tbx-11/TBX11K",  # Đường dẫn Kaggle của bạn
    "batch_size": 16,  # Tùy VRAM, 16 là an toàn cho T4/P100
    "learning_rate": 1e-4,  # LR chuẩn cho các lớp Conv/Linear
    "epochs": 10,  # Số vòng train
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "./checkpoints",  # Nơi lưu model
    "model_name": "google/siglip-base-patch16-384",  # Backbone
}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_seg = 0.0

    # Thanh progress bar
    pbar = tqdm(loader, desc="Training")

    for batch in pbar:
        # 1. Đẩy dữ liệu sang GPU
        images = batch["pixel_values"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        # 2. Forward Pass
        outputs = model(images)

        # 3. Tính Loss
        # Gom mask và label thành 1 dict target để đưa vào hàm loss
        targets = {"mask": masks, "label": labels}
        loss_dict = criterion(outputs, targets)

        loss = loss_dict["total_loss"]

        # 4. Backward & Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. Logging
        running_loss += loss.item()
        running_cls += loss_dict["loss_cls"]
        running_seg += loss_dict["loss_seg"]

        # Cập nhật thông tin lên thanh progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Cls": f"{loss_dict['loss_cls']:.3f}",
                "Seg": f"{loss_dict['loss_seg']:.3f}",
            }
        )

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Không tính gradient khi validate
        for batch in tqdm(loader, desc="Validating"):
            images = batch["pixel_values"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            targets = {"mask": masks, "label": labels}

            # Tính loss
            loss_dict = criterion(outputs, targets)
            running_loss += loss_dict["total_loss"]

            # Tính Accuracy cho phần Classification
            logits = outputs["class_logits"]
            preds = (torch.sigmoid(logits) > 0.5).float().squeeze()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def main():
    # 1. Setup
    os.makedirs(CONFIG["save_path"], exist_ok=True)
    print(f"Training on device: {CONFIG['device']}")

    # 2. Data Loaders
    print("Loading Datasets...")
    train_dataset = TBX11KDataset(
        root_dir=CONFIG["root_dir"], split="train", img_size=384
    )
    val_dataset = TBX11KDataset(root_dir=CONFIG["root_dir"], split="val", img_size=384)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2
    )

    # 3. Model & Loss
    print("Initializing Model...")
    model = MedicalConceptModel(model_name=CONFIG["model_name"])
    model.to(CONFIG["device"])

    criterion = MedicalLoss(seg_weight=5.0, contra_weight=1.0)

    # Chỉ optimize các tham số có requires_grad=True (bỏ qua backbone bị đóng băng)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
    )

    # 4. Training Loop
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, CONFIG["device"]
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG["device"])

        print(f"Epoch {epoch+1} Result:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save Best Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_file = os.path.join(CONFIG["save_path"], "best_model.pth")
            torch.save(model.state_dict(), save_file)
            print(f"✅ Model saved to {save_file}")

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()
