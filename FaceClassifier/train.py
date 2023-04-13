import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from model import FaceClassifierResNet18
import datetime
import argparse



# 定义和解析命令行参数
parser = argparse.ArgumentParser(description="Train a binary face classifier")
parser.add_argument('--img_size', type=int, default=256, help="Image size (default: 256)")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs (default: 10)")
parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate (default: 0.1)")
parser.add_argument('--data_dir', type=str, default='./data/', help="Data directory (default: './data/')")
args = parser.parse_args()




class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.face_images = datasets.ImageFolder(os.path.join(data_dir, "faces"), transform=transform)
        self.noface_images = datasets.ImageFolder(os.path.join(data_dir, "noface"), transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.face_images) + len(self.noface_images)

    def __getitem__(self, idx):
        if idx < len(self.face_images):
            image, _ = self.face_images[idx]
            label = torch.tensor(1.0)  # 设置人脸图像的标签为1
        else:
            image, _ = self.noface_images[idx - len(self.face_images)]
            label = torch.tensor(0.0)  # 设置非人脸图像的标签为0

        return image, label

# 参数设置
img_size = args.img_size
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
data_dir = args.data_dir

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# 数据集与数据加载器
train_dataset = FaceDataset(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 模型、损失函数和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FaceClassifierResNet18().to(device)
criterion = nn.BCELoss()  # 使用二元交叉熵损失（Binary Cross Entropy Loss）
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        predicted = (outputs > 0.5).float()  # 设置阈值为0.5
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

print("Training completed.")


timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_path = f'./models/face_classifier_{timestamp}.pth'
torch.save(model.state_dict(), model_path)