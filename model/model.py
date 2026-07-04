import os
import warnings
from PIL import Image, ImageFile
from torchvision import transforms, models
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DINOv2_ResNet34_Hybrid(nn.Module):
    def __init__(self, clinical_feature_size=12, num_classes=2, dropout_rate=0.4):
        super().__init__()

        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14'
        )
        self.embed_dim = 384
        resnet = models.resnet34(pretrained=True)

        self.adapter = nn.Conv2d(self.embed_dim, 64, kernel_size=1, bias=False)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.single_img_feature_dim = 896

        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_feature_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.single_img_feature_dim * 2 + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward_image_branch(self, x):
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            patch_tokens = features["x_norm_patchtokens"]  # (B, 256, 384)
            cls_token = features["x_norm_clstoken"]  # (B, 384)

        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)

        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)

        feat_map_upsampled = F.interpolate(
            feat_map,
            size=(56, 56),
            mode='bilinear',
            align_corners=False
        )

        x_cnn = self.adapter(feat_map_upsampled)
        x_cnn = self.layer2(x_cnn)
        x_cnn = self.layer3(x_cnn)
        x_cnn = self.layer4(x_cnn)

        x_cnn = self.avgpool(x_cnn)
        cnn_local_feature = torch.flatten(x_cnn, 1)

        return torch.cat([cnn_local_feature, cls_token], dim=1)

    def forward(self, img1, img2, clinical):
        f1 = self.forward_image_branch(img1)
        f2 = self.forward_image_branch(img2)
        c = self.clinical_processor(clinical)
        combined = torch.cat([f1, f2, c], dim=1)

        return self.fusion_layer(combined)

def prepare_data(csv_path):
    data = pd.read_csv(csv_path, encoding='utf-8')

    continuous_cols = ['age']
    other_basic_cols = ['gender']
    categorical_cols = ['microcalcifications', 'irregular margins', 'markedly hypoechoic', 'solid composition', 'vertical orientation']
    confidence_cols = ['conf_microcalcifications', 'conf_irregular margins', 'conf_markedly hypoechoic', 'conf_solid composition', 'conf_vertical orientation']

    clinical_cols = continuous_cols + other_basic_cols + categorical_cols + confidence_cols

    data[categorical_cols] = data[categorical_cols].fillna(0)
    data[confidence_cols] = data[confidence_cols].fillna(0.5)

    data['age'] = data['age'].fillna(data['age'].median())
    data['gender'] = data['gender'].fillna(data['gender'].mode()[0])

    train_df = data[data['dataset'] == 'train'].copy()
    valid_df = data[data['dataset'] == 'valid'].copy()
    test_df = data[data['dataset'] == 'test'].copy()

    scaler = StandardScaler()
    train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols])
    valid_df[continuous_cols] = scaler.transform(valid_df[continuous_cols])
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

    total_samples = len(data)

    print("training set：\n", train_df['labels'].value_counts(dropna=False).sort_index())
    print("validation set：\n", valid_df['labels'].value_counts(dropna=False).sort_index())
    print("test set：\n", test_df['labels'].value_counts(dropna=False).sort_index())

    return train_df, valid_df, test_df, clinical_cols


class DualImageDataset(Dataset):
    def __init__(self, df, clinical_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.clinical_cols = clinical_cols
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img1 = Image.open(row['image1_path']).convert('RGB')
        img2 = Image.open(row['image2_path']).convert('RGB')
        clinical = torch.tensor(row[self.clinical_cols].values.astype(np.float32))
        label = int(row['labels'])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, clinical, label

class TrainingController:
    def __init__(self, patience=20):
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def plot_training_history(history):
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#e74c3c', marker='o', markersize=3)
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='#3498db', marker='s', markersize=3)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', color='#2ecc71', marker='^', markersize=3)
    plt.plot(epochs_range, history['val_auc'], label='Val AUC', color='#9b59b6', marker='d', markersize=3)
    plt.title('Validation Accuracy and AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("training_metrics_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_file = r"csv/data.csv"
    train_df, valid_df, test_df, clinical_cols = prepare_data(csv_file)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }

    train_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(DualImageDataset(train_df, clinical_cols, train_trans), batch_size=32, shuffle=True)
    valid_loader = DataLoader(DualImageDataset(valid_df, clinical_cols, val_trans), batch_size=32)
    test_loader = DataLoader(DualImageDataset(test_df, clinical_cols, val_trans), batch_size=32)

    model = DINOv2_ResNet34_Hybrid(
        clinical_feature_size=len(clinical_cols),
        num_classes=2
    ).to(device)

    print("Freeze all parameters of the DINOv2 Backbone")
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.AdamW([
        {'params': model.layer2.parameters(), 'lr': 5e-5},
        {'params': model.layer3.parameters(), 'lr': 5e-5},
        {'params': model.layer4.parameters(), 'lr': 5e-5},
        {'params': model.adapter.parameters(), 'lr': 1e-3},
        {'params': model.clinical_processor.parameters(), 'lr': 1e-3},
        {'params': model.fusion_layer.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        threshold=1e-4,
        cooldown=1
    )

    scaler = torch.cuda.amp.GradScaler()
    controller = TrainingController(patience=20)

    epochs = 100

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for img1, img2, clin, labels in train_loader:
            img1, img2, clin, labels = img1.to(device), img2.to(device), clin.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(img1, img2, clin)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item()

        model.eval()
        val_loss, correct = 0, 0
        all_val_labels = []
        all_val_probs = []

        with torch.no_grad():
            for img1, img2, clin, labels in valid_loader:
                img1, img2, clin, labels = img1.to(device), img2.to(device), clin.to(device), labels.to(device)
                outputs = model(img1, img2, clin)

                v_loss = criterion(outputs, labels)
                val_loss += v_loss.item()

                correct += (outputs.argmax(1) == labels).sum().item()

                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        acc = correct / len(valid_df)

        try:
            auc_val = roc_auc_score(all_val_labels, all_val_probs)
        except:
            auc_val = 0.5

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)
        history['val_auc'].append(auc_val)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss={avg_train_loss:.4f} | "
              f"Val Loss={avg_val_loss:.4f} | Val Acc={acc:.4f} | Val AUC={auc_val:.4f}")

        scheduler.step(avg_val_loss)
        if controller.step(avg_val_loss):
            torch.save(model.state_dict(), "best_model.pth")

        if controller.early_stop:
            print("Early stopping triggered. Validation loss hasn't improved for 20 epochs.")
            break

    plot_training_history(history)
