# ExpressionClassifier

表情分类识别模块

基于PyTorch框架，使用 VGG19 和 ResNet18 两种网络结构实现，使用了 FER2013 和 CK+两个数据集

## 模型效果

在 FER2013 数据集上的准确率：

+ VGG19：71.33%
+ ResNet：72.75%

在 CK+ 数据集上的准确率：

+ VGG19：81.82%
+ ResNet：87.88%

## 文件结构

```
│  loadData.py  // 加载数据集
│  model.py		// 自定义模型
│  README.md
│  run.py		// 运行训练、验证、测试
│  utils.py
│  visualize.py	// 可视化展示
│          
├─analyzer
│      plotAcc.py	// 绘制准确率曲线
│      
├─data
│  ├─CK+
│  │  │  CK+_data.h5
│  │          
│  └─FER2013
│          data.h5
│          fer2013.bib
│          fer2013.csv
│          README
│          
├─images	// 测试图片
│  │  1.jpg
│  │  
│  └─emojis
│      
├─models 	// 保存训练好的模型
│      
├─preprocess	// 预处理程序
│      preprocessCK+.py
│      preprocessFER.py
```

## run.py 参数说明

+ --model：指定使用模型名称，"vgg"为VGG19，"resnet"为ResNet18
+ --dataset：指定使用数据集，"FER2013" 为FER2013数据集，"CK+"为CK+数据集
+ --do_train：是否训练
+ --do_eval：是否在验证集上验证
+ --do_test：是否在测试集上测试
+ --train_batch_szie：训练时的batch_size
+ --eval_batch_size：验证测试时的batch_size
+ --epochs：训练轮数
+ --learning_rate：学习率
+ --resume：是否对已有模型从上一批次继续训练
+ --show_confusion：是否展示混淆矩阵

## FER2013数据集运行

+ 需要先获取FER2013原数据集，将其放于`data/FER2013/`，命名为`fer2013.csv`

#### 数据预处理

````bash
cd preprocess
python preprocessFER.py
````

#### 模型训练

+ VGG训练：

    ````bash
    python run.py --model=vgg --dataset=FER2013 --do_train
    ````

+ ResNet训练：

    ````bash
    python run.py --model=resnet --dataset=FER2013 --do_train
    ````

#### 模型测试

+ VGG测试：

    ````bash
    python run.py --model=vgg --dataset=FER2013 --do_test
    ````

+ ResNet测试：

    ````bash
    python run.py --model=resnet --dataset=FER2013 --do_test
    ````

#### 绘制准确率曲线

需要在代码中修改相应的路径（训练过程的log日志，存于log文件夹下）

````bash
cd analyzer
python plotAcc.py
````

#### 绘制混淆矩阵

+ VGG：

    ````bash
    python run.py --model=vgg --dataset=FER2013 --do_test --show_confusion
    ````

+ ResNet测试：

    ````bash
    python run.py --model=resnet --dataset=FER2013 --do_test --show_confusion
    ````

## CK+ 数据集运行

#### 数据预处理

````bash
cd preprocess
python preprocessCK+.py
````

#### 模型训练

+ VGG训练：

    ````bash
    python run.py --model=vgg --dataset=CK+ --do_train
    ````

+ ResNet训练：

    ````bash
    python run.py --model=resnet --dataset=CK+ --do_train
    ````

#### 模型测试

+ VGG测试：

    ````bash
    python run.py --model=vgg --dataset=CK+ --do_test
    ````

+ ResNet测试：

    ````bash
    python run.py --model=resnet --dataset=CK+ --do_test
    ````

#### 绘制准确率曲线

需要在代码中修改相应的路径（训练过程的log日志，存于log文件夹下）

````bash
cd analyzer
python plotAcc.py
````

#### 绘制混淆矩阵

+ VGG：

    ````bash
    python run.py --model=vgg --dataset=CK+ --do_test --show_confusion
    ````

+ ResNet测试：

    ````bash
    python run.py --model=resnet --dataset=CK+ --do_test --show_confusion
    ````

## 模型分类结果可视化分析

需要在visualize.py中修改待分析的图片路径，如果需要测试新的图片，需要将图片放于`images/`路径下

````bash
python visualize.py
````

