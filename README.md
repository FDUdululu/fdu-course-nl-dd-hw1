# 从零开始构建三层神经网络分类器，实现图像分类

作业：从零开始构建三层神经网络分类器，实现图像分类

### 作业要求

任务描述：
手工搭建三层神经网络分类器，在数据集 CIFAR-10 上进行训练以实现图像分类。

基本要求：

- 本次作业要求自主实现反向传播，**不允许使用 pytorch，tensorflow** 等现成的支持自动微分的深度学习框架，**可以使用 numpy**；
- 最终提交的代码中应至少包含模型、训练、测试和参数查找四个部分，鼓励进行模块化设计；
- 其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现 SGD 优化器、学习率下降、交叉熵损失和 L2 正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。

### 项目结构

```
.
├── main.py           # 主程序
├── utils.py          # 辅助工具：包含数据加载器、激活函数集、神经网络模型
├── trainer.py        # 模型训练模块
├── test.py           # 模型评估模块
├── param_search.py   # 超参数搜索模块
├── checkpoints/      # 模型检查点保存目录：包含不同超参数下模型、最佳参数模型
├── data/
    └──cifar-10-batches-py/ # CIFAR-10 数据集目录
├── figs/ 			  # 可视化结果
├── classification_report.txt   # 分类结果报告
└── requirements.txt   # 项目所需 python 包
```

### 项目设计

本项目实现了一个三层神经网络分类器，具有以下特点：

- 可配置的隐藏层大小和激活函数类型（支持ReLU、Sigmoid 和 Tanh）

- 手动实现的前向传播和反向传播算法（不依赖深度学习框架）

- 完整的 SGD 优化器、学习率调整机制、支持早停、交叉熵损失函数和 L2 正则化

- 基于验证集性能的最佳模型选择和保存
- 超参数搜索可调节学习率、隐藏层大小、正则化强度、激活函数等超参数

- 可视化训练过程，训练结果和模型参数

### 数据集

- 数据集：CIFAR-10
- 数据描述：包含 10 个类别的彩色图像，每张图像大小为 32$\times$32，共计 60000 张图像，其中训练集 50000 张，测试集 10000 张。训练过程中将再训练集划分为训练集 45000 张，验证集 5000 张。
- 请从 [CIFAR-10官网](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 Python 版本的 CIFAR-10 数据集，解压后将 `cifar-10-batches-py` 目录放在数据目录下。

### 神经网络模型

三层神经网络分类器：

- **输入层**：接收 CIFAR-10 数据集的图像输入，每张图像大小为 32$\times$32$\times$3，输入层的维度为 3072。
- **隐藏层**：支持自定义大小，默认隐藏层的维度为 512。
- **输出层**：全连接层，使用 softmax 作为激活函数，输出维度为 10，对应 CIFAR-10 的 10 个类别。

### 环境要求

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn (用于绘制混淆矩阵)
- seaborn (用于结果可视化)

安装依赖：

```bash
pip install numpy matplotlib scikit-learn seaborn
```

### 使用说明

#### 1. 超参数搜索

运行超参数搜索以找到最佳的模型配置：

```bash
python main.py --mode param_search --data_dir ./data/cifar-10-batches-py
```

#### 2. 训练模型

使用指定的超参数训练模型：

```bash
python main.py --mode train --data_dir ./data/cifar-10-batches-py --learning_rate 0.1 --hidden_size 512 --reg_lambda 0.0001 --activation relu --epochs 30
```

或者使用默认参数训练：

```bash
python main.py --mode train --data_dir ./data/cifar-10-batches-py
```

#### 3. 测试模型

使用预训练的模型在测试集上进行评估：

```bash
python main.py --mode test --data_dir ./data/cifar-10-batches-py --checkpoint ./checkpoints/best_model.npz
```

### 训练好的模型

训练好的模型权重文件可以从以下链接下载： [模型权重下载链接](https://drive.google.com/file/d/YOUR_MODEL_FILE_ID/view?usp=sharing)

下载后请将模型权重文件放在`checkpoints`目录下。

### 实验结果

#### （一）训练过程分析

- **损失值曲线**：在训练过程中，记录每一轮训练和验证集上的损失值，生成损失值随训练轮数变化的曲线。从下图中可以看出，训练集损失值在开始阶段迅速下降，随着训练的进行，下降速度略微变缓，表明模型在不断学习数据特征。但验证集损失值在前期与训练集损失值趋势相似，而大约第 15 轮之后逐渐收敛，可能是模型开始出现过拟合现象。
- **准确率曲线**：下图记录了每一轮验证集上的准确率，生成准确率随训练轮数变化的曲线。从图中可以看出，验证集准确率在第 15 轮后增长缓慢，略有波动，进一步验证了模型可能存在过拟合问题，但还未完全过拟合。

![training_history](D:/Desktop/大三下/计算机视觉/homework1/figs/training_history.png)

#### （二）模型性能评估

- **总体准确率**：在测试集上，模型的总体准确率达到了 55%，表明模型在整体上能够正确分类大部分样本，但仍有较大的提升空间。

- **各类别精确率、召回率和 F1 值**：

  详细的各类别评估指标如下表所示，从表中可以看出，不同类别之间的性能存在一定差异。例如，airplane/ automobile/ ship 等类别的精确率、召回率和 F1- score 都相对较高，说明模型对这些类别的预测较为准确且能够较好地覆盖真实样本；而 cat 类别对应的性能则相对较低，可能是该类别样本与其他类别具有相似的特征，导致模型容易混淆。

  | 类别         | Precision | Recall | F1 - score | Support |
  | ------------ | --------- | ------ | ---------- | ------- |
  | airplane     | 0.59      | 0.64   | 0.61       | 1000    |
  | automobile   | 0.67      | 0.65   | 0.66       | 1000    |
  | bird         | 0.45      | 0.42   | 0.44       | 1000    |
  | cat          | 0.38      | 0.32   | 0.35       | 1000    |
  | deer         | 0.47      | 0.48   | 0.47       | 1000    |
  | dog          | 0.48      | 0.44   | 0.46       | 1000    |
  | frog         | 0.53      | 0.70   | 0.60       | 1000    |
  | horse        | 0.62      | 0.60   | 0.61       | 1000    |
  | ship         | 0.67      | 0.65   | 0.66       | 1000    |
  | truck        | 0.59      | 0.60   | 0.60       | 1000    |
  | accuracy     | -         | -      | 0.55       | 10000   |
  | macro avg    | 0.55      | 0.55   | 0.55       | 10000   |
  | weighted avg | 0.55      | 0.55   | 0.55       | 10000   |

- **混淆矩阵**：为了更直观地展示模型在各个类别上的分类情况，绘制了混淆矩阵。混淆矩阵的行表示真实类别，列表示预测类别，矩阵中的每个元素表示真实类别为该行对应的类别，而被预测为该列对应的类别的样本数量。

  <img src="D:/Desktop/大三下/计算机视觉/homework1/figs/confusion_matrix.png" alt="confusion_matrix" style="zoom:70%;" />

  从混淆矩阵中可以发现，模型经常将类别 cat 和 dog 混淆。以下是一些模型预测错误的实例：

  <img src="D:/Desktop/大三下/计算机视觉/homework1/figs/misclassifications.png" alt="misclassifications" style="zoom: 58%;" />

#### （三）参数可视化

该部分可视化了输入层权重参数

- **可视化方法**：归一化权重参数，转为 RGB 通道显示

- **可视化结果**：从下图看出，输入层权重参数学习到了边缘或者特征规律等模式。

  <img src="D:/Desktop/大三下/计算机视觉/homework1/figs/weight_visualization.png" alt="weight_visualization" style="zoom:50%;" />

## GitHub仓库

本项目的代码托管在GitHub上：[三层神经网络分类器](https://github.com/FDUdululu/fdu-course-nl-dd-hw1)
