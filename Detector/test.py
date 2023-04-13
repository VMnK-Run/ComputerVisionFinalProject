import torch
import argparse

import os
from PIL import Image
from torchvision import transforms
from model import FaceDetectorResNet34

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None, help='Directory containing the model.')
parser.add_argument('--test_face_dir', type=str, default='./data/test', help='Directory containing test face images.')
parser.add_argument('--image_size', type=int, default=500, help='Size of the input image.')
args = parser.parse_args()

# 如果未指定模型路径，则使用models文件夹中的最新模型
if args.model_path is None:
    model_dir = './models'
    model_files = os.listdir(model_dir)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    args.model_path = os.path.join(model_dir, model_files[0])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FaceDetectorResNet34().to(device).to(device)
model.load_state_dict(torch.load(args.model_path))


test_face_dir = args.test_face_dir
image_size = args.image_size

image_files = os.listdir(test_face_dir)


# 图像预处理
img_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def show_image_with_boxes(ax, image_path, boxes):
    img = Image.open(image_path).convert('RGB')
    
    img = img.resize((image_size, image_size))

    ax.imshow(img)

    for box in boxes:
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

model.eval()  # 将模型设置为评估模式

# 随机选择10张图像
num_images_to_show = 10
random_image_files = np.random.choice(image_files, num_images_to_show, replace=False)

# 创建一个 figure
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, image_file in enumerate(random_image_files):
    image_path = os.path.join(test_face_dir, image_file)
    img = Image.open(image_path).convert('RGB')
    img_tensor = img_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 预测边界框
        box = model(img_tensor)
        box = box.squeeze().tolist()

    # 获取当前子图位置
    row = i // 5
    col = i % 5
    ax = axes[row, col]

    show_image_with_boxes(ax, image_path, [box])

plt.tight_layout()
plt.show()
