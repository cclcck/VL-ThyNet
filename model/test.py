import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
print('--- Start Final Test ---')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
model.eval()

predictions = []
true_labels = []
all_probs = []

with torch.no_grad():
    correct = 0
    total = 0

    for img1, img2, clinical, labels in test_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        logits = model(img1, img2, clinical)

        probs = torch.softmax(logits, dim=1)

        _, predicted = torch.max(logits, 1)

        predictions.extend(predicted.cpu().numpy().tolist())
        true_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # 保存正类概率

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


conf_matrix = confusion_matrix(true_labels, predictions)
TN, FP, FN, TP = conf_matrix.ravel()

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)
auc_score = roc_auc_score(true_labels, all_probs)

print("\n" + "=" * 30)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test AUC:      {auc_score:.4f}")
print(f"Sensitivity:   {sensitivity:.4f}")
print(f"Specificity:   {specificity:.4f}")
print("=" * 30)

print("\nConfusion Matrix:")
print(conf_matrix)

# 打印详细分类报告
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=['Benign', 'Malignant']))

# 打印部分结果对比
print("\nTrue labels:    ", true_labels[:20], "... (Showing first 20)")
print("Predicted labels:", predictions[:20], "... (Showing first 20)")


def plot_roc_curve(true_labels, all_probs, auc_score):

    fpr, tpr, thresholds = roc_curve(true_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) - Test Set')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig('test_roc_curve.png')
    plt.show()
    print("\nROC curve has been saved as 'test_roc_curve.png'")

plot_roc_curve(true_labels, all_probs, auc_score)