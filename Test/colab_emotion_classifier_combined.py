# ===============================
# 1. Install and Import Dependencies
# ===============================
# In Colab, use these commands in a cell:
# !pip install torch torchvision matplotlib seaborn scikit-learn tqdm pillow

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# ===============================
# 2. Dataset Class
# ===============================
class GenericImageDataset(Dataset):
    def __init__(self, root_dir, split='train', target_size=(96, 96), augment=False):
        self.root_dir = os.path.join(root_dir, split)
        self.classes = sorted([d for d in os.listdir(self.root_dir)
                              if not d.startswith('.') and os.path.isdir(os.path.join(self.root_dir, d))])
        self.samples = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith('.jpg') and not fname.startswith('.'):
                    self.samples.append((os.path.join(class_dir, fname), label))
        self.transform = self._build_transform(target_size, augment)

    def _build_transform(self, target_size, augment):
        from torchvision import transforms
        tfms = [transforms.Resize(target_size)]
        if augment:
            tfms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ]
        tfms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(tfms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

# ===============================
# 3. Model Definition
# ===============================
# --- Swish Activation ---
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# --- DropBlock Regularization ---
class DropBlock2D(nn.Module):
    def __init__(self, block_size, drop_prob):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < gamma).float()
        block_mask = F.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        out = x * (1 - block_mask)
        scale = block_mask.numel() / (block_mask.sum() + 1e-6)
        out = out * scale
        return out

# --- Inception Block ---
class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# --- Residual Dense Block ---
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, in_channels, kernel_size=1)
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        return x + x3

# --- MBConv Block ---
class MBConv(nn.Module):
    def __init__(self, in_channels, activation=None, expansion_factor=6):
        super().__init__()
        if activation is None:
            activation = Swish()
        hidden_dim = in_channels * expansion_factor
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            activation,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            activation,
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )
        self.se = SEBlock(in_channels)
    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return out + x

# --- Ghost Module ---
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, ratio=2):
        super().__init__()
        if activation is None:
            activation = Swish()
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_channels),
            activation
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            activation
        )
    def forward(self, x):
        primary = self.primary_conv(x)
        cheap = self.cheap_operation(primary)
        return torch.cat([primary, cheap], dim=1)

# --- Dual Attention Block (CBAM) ---
class DualAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        ca = self.channel_att(x)
        x = x * ca
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        sa = self.spatial_att(torch.cat([avg, max_], dim=1))
        return x * sa

# --- Selective Kernel Convolution ---
class SKConv(nn.Module):
    def __init__(self, in_channels, activation=None, M=2, G=8, r=16):
        super().__init__()
        if activation is None:
            activation = Swish()
        d = max(in_channels // r, 32)
        self.M = M
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3 + 2 * i, padding=1 + i, groups=G),
                nn.BatchNorm2d(in_channels),
                activation
            ) for i in range(M)
        ])
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, d, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, in_channels * M, 1)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)
        attn = self.fc(torch.sum(feats, dim=1)).view(x.size(0), self.M, -1, 1, 1)
        attn = self.softmax(attn)
        out = torch.sum(feats * attn, dim=1)
        return out

# --- ReZero Residual Block ---
class ReZeroResidualBlock(nn.Module):
    def __init__(self, channels, activation=None):
        super().__init__()
        if activation is None:
            activation = Swish()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            activation,
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x + self.alpha * self.block(x)

# --- CSP-Inception Block ---
class CSPInception(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.split = in_channels // 2
        self.inception_path = nn.Sequential(
            nn.Conv2d(self.split, self.split, 1),
            nn.Conv2d(self.split, self.split, 3, padding=1),
            nn.Conv2d(self.split, self.split, 5, padding=2)
        )
        self.concat_conv = nn.Conv2d(self.split * 2, in_channels, 1)
    def forward(self, x):
        x1, x2 = torch.split(x, self.split, dim=1)
        out = self.inception_path(x1)
        out = torch.cat([x2, out], dim=1)
        out = self.concat_conv(out)
        return out

# --- Global Context Block ---
class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, activation=None):
        super().__init__()
        if activation is None:
            activation = Swish()
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            activation,
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
    def forward(self, x):
        context = self.pool(x)
        context = self.block(context)
        return x + context.expand_as(x)

# --- Multi-Head Self-Attention ---
class MHSA(nn.Module):
    def __init__(self, in_channels, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (C // self.heads) ** 0.5, dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        out = self.proj(out)
        return out + x

# --- Attention Pooling ---
class AttentionPooling(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.attn = nn.Conv2d(in_channels, 1, 1)
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, x):
        w = torch.softmax(self.attn(x).view(x.size(0), -1), dim=1).view(x.size(0), 1, x.size(2), x.size(3))
        x = (x * w).sum(dim=[2, 3])
        return self.fc(x)

# --- ShunyaNet ---
class ShunyaNet(nn.Module):
    def __init__(self, num_classes=8, dropblock_prob=0.1, dropblock_size=5, activation=None):
        super().__init__()
        if activation is None:
            activation = Swish()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation
        )
        self.inception = InceptionBlock(64)
        self.se = SEBlock(128)
        self.res_dense = ResidualDenseBlock(128)
        self.mbconv = MBConv(128, activation=activation)
        self.ghost = GhostModule(128, 128, activation=activation)
        self.sk = SKConv(128, activation=activation)
        self.dual_att = DualAttention(128)
        self.csp_inception = CSPInception(128)
        self.rezero = ReZeroResidualBlock(128, activation=activation)
        self.global_context = GlobalContextBlock(128, activation=activation)
        self.mhsa = MHSA(128)
        self.dropblock = DropBlock2D(dropblock_size, dropblock_prob)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.attn_pool = AttentionPooling(128, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.stem(x)
        x = self.inception(x)
        x = self.se(x)
        x = self.res_dense(x)
        x = self.mbconv(x)
        x = self.ghost(x)
        x = self.sk(x)
        x = self.dual_att(x)
        x = self.csp_inception(x)
        x = self.rezero(x)
        x = self.global_context(x)
        x = self.mhsa(x)
        x = self.dropblock(x)
        out1 = self.classifier(x)
        out2 = self.attn_pool(x)
        return (out1 + out2) / 2

# ===============================
# 4. Training, Validation, Evaluation Functions
# ===============================
# Set device
import platform
if platform.system() == "Linux":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Configuration
class Config:
    data_dir = '/content/drive/MyDrive/Data/EmotionRecognitionSystem/8Emotions/'  # Change for Colab
    target_size = (96, 96)
    num_classes = 8
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    weight_decay = 1e-5
    dropblock_prob = 0.1
    dropblock_size = 5
    checkpoint_dir = './checkpoints'
    results_dir = './results'

os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.results_dir, exist_ok=True)

# Load datasets

def load_data():
    print("Loading datasets...")
    train_dataset = GenericImageDataset(
        Config.data_dir,
        split='train',
        target_size=Config.target_size,
        augment=True
    )
    val_dataset = GenericImageDataset(
        Config.data_dir,
        split='val',
        target_size=Config.target_size,
        augment=False
    )
    test_dataset = GenericImageDataset(
        Config.data_dir,
        split='test',
        target_size=Config.target_size,
        augment=False
    )
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    return train_loader, val_loader, test_loader, class_names

# Training function

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, class_names):
    print("Starting training...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_correct/train_total:.4f}")
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_all = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(predicted.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())
                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{val_correct/val_total:.4f}")
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        scheduler.step(epoch_val_loss)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_path = os.path.join(Config.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names
            }, checkpoint_path)
            print(f"  New best model saved with validation accuracy: {best_val_acc:.4f}")
            cm = confusion_matrix(val_labels_all, val_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Epoch {epoch+1}, Val Acc: {epoch_val_acc:.4f})')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.results_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
            plt.close()
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(Config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_acc': epoch_train_acc,
                'val_acc': epoch_val_acc,
                'class_names': class_names
            }, checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch+1}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.results_dir, 'training_history.png'))
    plt.close()
    return history, best_val_acc

# Evaluation function

def evaluate(model, test_loader, criterion, class_names):
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Test Set Confusion Matrix (Accuracy: {test_acc:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.results_dir, 'test_confusion_matrix.png'))
    plt.close()
    cr = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("Classification Report:")
    print(cr)
    with open(os.path.join(Config.results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cr)
    return test_acc, cm

# ===============================
# 5. Main Execution
# ===============================
def main():
    train_loader, val_loader, test_loader, class_names = load_data()
    print("Initializing ShunyaNet...")
    model = ShunyaNet(
        num_classes=Config.num_classes,
        dropblock_prob=Config.dropblock_prob,
        dropblock_size=Config.dropblock_size
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    history, best_val_acc = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        Config.num_epochs,
        class_names
    )
    checkpoint = torch.load(os.path.join(Config.checkpoint_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model with validation accuracy: {checkpoint['best_val_acc']:.4f}")
    test_acc, confusion_mat = evaluate(model, test_loader, criterion, class_names)
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Results saved in {Config.results_dir}")

if __name__ == "__main__":
    main()

