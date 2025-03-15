import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 自定义数据集类
class InstrumentDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.img_labels = []
        if not os.path.exists(txt_file):
            print(f"Error: {txt_file} does not exist.")
            return
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                try:
                    img_name, label = line.strip().split()
                    self.img_labels.append((img_name, int(label)))
                except ValueError:
                    print(f"Invalid line in {txt_file}: {line.strip()}. Skipping...")
        self.root_dir = root_dir
        self.transform = transform
        if len(self.img_labels) == 0:
            print(f"Warning: No valid samples found in {txt_file}.")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        # 只保留文件名部分
        img_name = os.path.basename(img_name)
        img_path = os.path.join(self.root_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Error: {img_path} does not exist.")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 加载训练集
train_dataset = InstrumentDataset(txt_file='dataset/train.txt', root_dir='dataset/images', transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 加载验证集
eval_dataset = InstrumentDataset(txt_file='dataset/eval.txt', root_dir='dataset/images', transform=data_transform)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# 加载预训练模型ResNet50
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 二分类（破损和未破损）

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 在验证集上评估模型
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for data in eval_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')