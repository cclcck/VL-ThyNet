import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm


# ============================================================
# 1) 基础设置
# ============================================================
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2) 模型定义 (DINOv2 + ResNet50 增强融合版)
# ============================================================
class DINOv2_ResNet50_Hybrid(nn.Module):
    def __init__(self, clinical_feature_size=9, num_classes=2, dropout_rate=0.4):
        super().__init__()

        # 1) DINOv2 Backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.embed_dim = 384

        # 2) ResNet50 适配器
        resnet = models.resnet50(pretrained=True)

        # --- 核心修复点：将 64 改为 256 ---
        self.adapter = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False),  # 修改此处
            nn.BatchNorm2d(256),  # 修改此处
            nn.ReLU(True),
            nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False)
        )

        # 提取 ResNet50 层
        # 注意：resnet.layer2 的输入期望是 256 个通道
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.img_feature_dim = 2048

        # 3) 临床特征处理器 (保持不变)
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_feature_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(True)
        )

        # 4) 增强型融合分类头
        # 输入维度: 2048*3 + 32 = 6176
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.img_feature_dim * 3 + 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

    def forward_image_branch(self, x):
        features = self.backbone.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)  # [B, 384, 16, 16]

        x = self.adapter(feat_map)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, img1, img2, clinical):
        f1 = self.forward_image_branch(img1)
        f2 = self.forward_image_branch(img2)

        diff = torch.abs(f1 - f2)
        c = self.clinical_processor(clinical)

        combined = torch.cat([f1, f2, diff, c], dim=1)
        return self.fusion_layer(combined)


# ============================================================
# 3) 数据准备
# ============================================================
def prepare_data(csv_path):
    data = pd.read_csv(csv_path)

    basic_cols = ['gender', 'age', 'Nodule_size_length', 'Nodule_size_wide', 'Nodule_size_depth']
    categorical_cols = ['Calcification', 'Margin', 'Echo', 'Internal Structure']
    confidence_cols = ['conf_calcification', 'conf_margin', 'conf_echo', 'conf_internal']

    data[categorical_cols] = data[categorical_cols].fillna(0)
    data[confidence_cols] = data[confidence_cols].fillna(0.5)
    data[basic_cols] = data[basic_cols].fillna(data[basic_cols].median())

    # 极化加权逻辑确认：正确且高效
    weighted_cols = []
    for cat, conf in zip(categorical_cols, confidence_cols):
        col_name = f'polar_weighted_{cat}'
        polarized = data[cat].map({0: -1, 1: 1})  # 0->-1(良), 1->1(恶)
        data[col_name] = polarized * data[conf]
        weighted_cols.append(col_name)

    final_clinical_cols = basic_cols + weighted_cols

    train_val_df, test_df = train_test_split(data, test_size=0.2, stratify=data['labels'], random_state=42)
    train_df, valid_df = train_test_split(train_val_df, test_size=0.25, stratify=train_val_df['labels'],
                                          random_state=42)

    scaler = StandardScaler()
    train_df[final_clinical_cols] = scaler.fit_transform(train_df[final_clinical_cols])
    valid_df[final_clinical_cols] = scaler.transform(valid_df[final_clinical_cols])
    test_df[final_clinical_cols] = scaler.transform(test_df[final_clinical_cols])

    return train_df, valid_df, test_df, final_clinical_cols


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
            img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, clinical, label


# ============================================================
# 4) 辅助工具
# ============================================================
class TrainingController:
    def __init__(self, patience=15):
        self.patience, self.best_loss, self.counter, self.early_stop = patience, np.inf, 0, False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss, self.counter = val_loss, 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
            return False


def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except:
        return 0.5


if __name__ == "__main__":
    # 1. 加载数据
    csv_file = r"D:\文章--甲状腺基因\文本＋图像\用两张图片进行训练.csv"
    train_df, valid_df, test_df, final_cols = prepare_data(csv_file)

    # 2. 数据转换与加载
    train_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(DualImageDataset(train_df, final_cols, train_trans), batch_size=32, shuffle=True)
    valid_loader = DataLoader(DualImageDataset(valid_df, final_cols, val_trans), batch_size=32)

    # 3. 初始化模型 (ResNet50 版本)
    model = DINOv2_ResNet50_Hybrid(clinical_feature_size=len(final_cols)).to(device)

    # 4. 初始优化设置 (冻结 Backbone)
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    controller = TrainingController(patience=20)
    scaler = torch.cuda.amp.GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    is_unfrozen = False

    # 5. 训练循环
    for epoch in range(100):
        # --- 核心修复：解冻逻辑必须在循环内且只执行一次 ---
        if epoch == 18 and not is_unfrozen:
            print("\n解冻 Backbone 微调 (ResNet50)...")
            for param in model.backbone.parameters():
                param.requires_grad = True

            # 针对 ResNet50 微调的专用学习率设置
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': 5e-6},
                {'params': model.adapter.parameters(), 'lr': 1e-5},
                {'params': model.layer2.parameters(), 'lr': 2e-5},
                {'params': model.layer3.parameters(), 'lr': 2e-5},
                {'params': model.layer4.parameters(), 'lr': 2e-5},
                {'params': model.fusion_layer.parameters(), 'lr': 1e-4},
            ], weight_decay=1e-3)
            is_unfrozen = True

        # --- 训练阶段 ---
        model.train()
        train_l = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/100 [Train]")
        for img1, img2, clin, labels in pbar:
            img1, img2, clin, labels = img1.to(device), img2.to(device), clin.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(img1, img2, clin)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_l += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- 验证阶段 ---
        model.eval()
        val_l, correct = 0, 0
        y_true, y_prob = [], []
        with torch.no_grad():
            for img1, img2, clin, labels in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/100 [Valid]"):
                img1, img2, clin, labels = img1.to(device), img2.to(device), clin.to(device), labels.to(device)
                outputs = model(img1, img2, clin)
                val_l += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()
                y_prob.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        # --- 统计与早停 ---
        avg_val_loss = val_l / len(valid_loader)
        auc_val = safe_auc(y_true, y_prob)
        acc_val = correct / len(valid_df)

        history['train_loss'].append(train_l / len(train_loader))
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc_val)
        history['val_auc'].append(auc_val)

        print(f"\nSummary: Val Loss={avg_val_loss:.4f} | Acc={acc_val:.4f} | AUC={auc_val:.4f}")

        scheduler.step(avg_val_loss)
        if controller.step(avg_val_loss):
            torch.save(model.state_dict(), "best_model_resnet50.pth")
            print("最佳模型保存成功")

        if controller.early_stop:
            print("触发早停")
            break

    # 绘制最终曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1);
    plt.plot(history['train_loss'], label='Train');
    plt.plot(history['val_loss'], label='Val');
    plt.title('Loss');
    plt.legend()
    plt.subplot(1, 2, 2);
    plt.plot(history['val_auc'], label='AUC');
    plt.plot(history['val_acc'], label='Acc');
    plt.title('Metrics');
    plt.legend()
    plt.show()