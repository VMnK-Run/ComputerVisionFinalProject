"""   encoding: UTF-8
run.py 参数说明:

--model: 选择用于人脸检测和分类的模型。
可选值：
  - "resnet" (默认): 使用自定义的ResNet模型进行人脸检测和分类。
  - "haar_cascade": 使用OpenCV的Haar级联分类器进行人脸检测。

示例:

1. 使用默认的ResNet模型:
   python run.py

2. 使用Haar级联分类器:
   python run.py --model haar_cascade
"""


from skimage import io
from skimage.transform import resize
import os
import torch
import pickle
from torchvision import transforms
from PIL import Image
from model import FaceClassifierResNet18
from model import FaceDetectorResNet34
from model import EmojiResNet18
import numpy as np
from PIL import Image
import cv2
from torch.autograd import Variable
import argparse
USE_GPU = True
device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")



emoji_net = EmojiResNet18().to(device)
checkpoint = torch.load('./models/emoji_resnet_FER2013.t7')
emoji_net.load_state_dict(checkpoint['model'])
emoji_net.eval()


emoji_img_size = 44     # 表情分类模型的输入图片尺寸是44*44
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emoji_transform = transforms.Compose([
    transforms.TenCrop(emoji_img_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# 这个列表是为了消除模型对各别帧的错误判断
# 每次运行get_emojie，列表首元素会被弹出，本次分类的结果会放入列表末尾。
# 返回列表中出现次数最多的表情
# 但是带来了一些延迟
emojis = [0] * 10
# 此方法输入一个BGR人脸图片(opencv默认就按照BGR来处理)，返回一个表情图片（也是BGR）
def get_emoji(face):   #should be BGR image

    # 方便从cv的图片格式转为PIL的格式，先存储成图片再读取
    cv2.imwrite('./temp.png',face)
    raw_img = io.imread(f'./temp.png')          
    os.remove("./temp.png")
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
    # print(gray.shape)
    img = gray[:, :, np.newaxis]
    # print(img.shape)
    img = np.concatenate((img, img, img), axis=2)
    # print(img.shape)
    img = Image.fromarray(img)
    inputs = emoji_transform(img)
    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w).to(device)
    with torch.no_grad():
        inputs = Variable(inputs, volatile=True)
        outputs = emoji_net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    # print(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)
    idx = int(predicted.cpu().numpy())
    emojis.append(idx)
    emojis.pop(0)

    new_idx = max(emojis,key=emojis.count)
    emojis_img = io.imread('images/emojis/%s.png' % str(class_names[new_idx]))
    emojis_img = cv2.cvtColor(emojis_img, cv2.COLOR_RGB2BGR)
    return emojis_img



# 这些是图片二分类resnet模型的参数（用于判断一张图片是否含有人脸的模型）
threshold = 0.5   # 模型输出高于这个值，认为图片中包含人脸
classifier_img_size = 256
classifier_transform = transforms.Compose([
    transforms.Resize((classifier_img_size, classifier_img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 这是人脸检测resnet模型的
detector_img_size = 512  
detector_transform = transforms.Compose([
    transforms.Resize((detector_img_size, detector_img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 二分类模型判断一张图片是否含有人脸
def classifyImage(model,image):
     # 预处理图像
    processed_image = classifier_transform(image)
    processed_image = processed_image.unsqueeze(0).to(device)  # 添加批次维度并将图像转移到GPU
    # 模型分类
    with torch.no_grad():
        output = model(processed_image)
        prediction = (output > threshold).float().item()

    return prediction == 1.0
    

# 人脸检测模型得到人脸矩形框并返回
def getFaceRectByResNet(model, image):
    original_size = image.size
    img_tensor = detector_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 预测边界框
        box = model(img_tensor)
        box = box.squeeze().tolist()
    
    x, y, w, h = box
    
    # 计算缩放因子
    x_scale = original_size[0] / detector_img_size
    y_scale = original_size[1] / detector_img_size
    
    # 使用缩放因子来反向计算原始图像中的边界框
    x1, y1, w1, h1 = x * x_scale, y * y_scale, w * x_scale, h * y_scale

    # 下面这些参数用于微调矩形框，使其更符合表情分类模型
    x_move_retio = 0.025
    y_move_retio = 0.125
    w_expansion_ratio = 1.2
    h_expansion_ratio = w1/h1
    w1_expanded = w1 * w_expansion_ratio
    h1_expanded = h1 * h_expansion_ratio
    x1_expanded = x1 - (w1_expanded - w1) / 2 + x1*x_move_retio
    y1_expanded = y1 - (h1_expanded - h1) / 2 + y1*y_move_retio


    return int(x1_expanded), int(y1_expanded), int(w1_expanded), int(h1_expanded) 




# CV的haar级联器得到图片中的人脸矩形框（可以是多个）
def haar_detect_faces(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


# 向图片中绘制矩形框
def draw_faces_rectangles(img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# 切割图片，得到人脸图片
def get_face(img, faces):
    if len(faces) == 1:
        x, y, w, h = faces[0]
        if(w<emoji_img_size or h<emoji_img_size):
            return None
        face_img = img[y:y+h, x:x+w]
        # gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return face_img
    else:
        return None

# 把原始图片、人脸图片、emoji表情拼接在一起
def concatenate_images(frame, face, emoji, gap):
    height = frame.shape[0]
    width = frame.shape[1]
    
    right_width = width//2
    right_height = (height - gap)//2
    face = cv2.resize(face,(right_width,right_height))
    emoji = cv2.resize(emoji,(right_width,right_height))
    hgap = np.zeros((height,gap,3)).astype(np.uint8)
    vgap = np.zeros((height - 2* right_height,right_width,3)).astype(np.uint8)
    right = cv2.vconcat([face,vgap,emoji])
    img = cv2.hconcat([frame,hgap,right])
    return img
    
# 三种模型
classifier = None
detector = None
cv_detector = None

def init(model_type):
    global classifier, detector, cv_detector

    if model_type != "haar_cascade":
        classifier = FaceClassifierResNet18().to(device)
        # 加载权重
        classifier.load_state_dict(torch.load('./models/face_classifier.pth'))
        classifier.eval()

        detector = FaceDetectorResNet34().to(device)
        # 加载权重
        detector.load_state_dict(torch.load('./models/face_detection_model_weights_80_epochs.pth'))
        detector.eval()
    else:
        # 加载Haar级联分类器
        cv_detector = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")


def main(model_type):
    init(model_type)
    # try:
    run()
    # except Exception as e:
    cv2.destroyAllWindows()

def run():
    cap = cv2.VideoCapture(0)
    while True:
        # 读取摄像头的图像
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture an image.")
            break
        faces = []
        if(cv_detector is not None):   # Haar级联分类器
        # 检测人脸并绘制矩形
            faces = haar_detect_faces(frame, cv_detector)
        else:                          # ResNet
            # 将图像从BGR转换为RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            # 判断是画面中否含有人脸
            has_face = classifyImage(classifier,pil_image)
            if has_face:
                x,y,w,h = getFaceRectByResNet(detector,pil_image)
                faces = [[x,y,w,h]]

        face = get_face(frame,faces) 
        draw_faces_rectangles(frame, faces)
        if(face is not None):
            emoji = get_emoji(face)
            cv2.imshow('Face-Emoji',concatenate_images(frame,face,emoji,10))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
     # 释放摄像头资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select the model for face detection and classification")
    parser.add_argument('--model', type=str, default="resnet", choices=["resnet", "haar_cascade"],
                        help="Select the model for face detection and classification (default: resnet)")
    args = parser.parse_args()
    main(args.model)
