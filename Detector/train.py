import torch
from torch import nn
import argparse

import os
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import FaceDetectorResNet34
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--face_dir', type=str, default='./data/faces', help='Directory containing face images.')
parser.add_argument('--boxes_path', type=str, default='./data/face-locate/boxes.pkl', help='Pickle file containing face bounding boxes.')
parser.add_argument('--image_size', type=int, default=500, help='Size of the input image.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
args = parser.parse_args()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FaceDetectorResNet34().to(device)


class FaceDataset(Dataset):
    def __init__(self, img_folder, box_file,image_size):
        self.image_size = image_size
        self.img_folder = img_folder
        self.box_data = pickle.load(open(box_file, 'rb'))
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 获取原始图像大小
        self.img_sizes = []
        for i in range(len(self.box_data)):
            img_path = os.path.join(self.img_folder, f"{i+1}.jpg")
            img = Image.open(img_path)
            self.img_sizes.append(img.size)

        # 将边界框归一化
        self.normalized_boxes = []
        for i, box in enumerate(self.box_data):
            width, height = self.img_sizes[i]
            normalized_box = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
            self.normalized_boxes.append(normalized_box)

    def __len__(self):
        return len(self.box_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, f"{idx+1}.jpg")
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.img_transform(img)

        # 将归一化的边界框与缩放后的图像大小相乘
        box = self.normalized_boxes[idx]
        box = [box[0] * self.image_size, box[1] * self.image_size, box[2] * self.image_size, box[3] * self.image_size]

        return img_tensor, torch.tensor(box, dtype=torch.float32)
    

image_size = args.image_size
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
face_dir = args.face_dir    #这个目录里的图片必须用 1.jpg 2.jpg 3.jpg的方式命名

boxes_path = args.boxes_path # 这个文件是一个python list,格式如下：
""""
[
    [x,y,w,h],   # 代表1.jpg的矩形框
    [x,y,w,h]    # 代表2.jpg的矩形框
]
"""



dataset = FaceDataset(face_dir, boxes_path,image_size=image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


model.train() 
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_path = f'./models/face_detection_model_{timestamp}.pth'
torch.save(model.state_dict(), model_path)
