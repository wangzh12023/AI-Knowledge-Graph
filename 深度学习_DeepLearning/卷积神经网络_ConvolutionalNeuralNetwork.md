

## ✨ 卷积神经网络（CNN）的动机与设计目的

### 一、动机：视觉识别的挑战

在计算机视觉任务中，我们的目标是构建一个可以**从图像中识别物体**的神经网络。例如：

- 假设我们希望设计一个可以处理大小为 **200x200** 的 **RGB图像**（每个像素有3个颜色通道）的模型；
    
- 若采用**全连接网络（Fully Connected Network）**，即第一层每个神经元与图像的所有像素连接，那么参数数量将会非常庞大：
    

200×200×3×1000=1.2×108=1.2亿个参数200 \times 200 \times 3 \times 1000 = 1.2 \times 10^8 = 1.2亿个参数

不仅计算量大，而且会导致过拟合、训练困难。

此外：

- **如果图像中的物体稍微移动**（例如向左或向右平移几个像素），则全连接网络可能会完全失效，因为它不具备**平移不变性**或**平移等变性**的能力。
    

---

### 二、目标：构建鲁棒、有效的视觉识别网络

我们希望设计一个神经网络，满足以下几点：

1. 能够处理 **高维输入**（如图像）；
    
2. 能够利用图像像素的 **二维拓扑结构**（即邻近像素之间的相关性）；
    
3. 对一些常见的变化具有 **不变性或等变性**（如图像的平移、形变、光照变化等）。
    

---

### 三、解决方案：卷积神经网络（CNN）

卷积网络通过以下关键设计思想实现上述目标：

#### 1️⃣ 局部连接（Local Connectivity）

- 每个隐藏神经元只与输入图像的一个 **局部小区域（感受野）** 相连；
    
- 感受野通常是一个小的二维窗口（如 3x3、5x5）；
    
- 每个神经元通常会连接该区域的 **所有通道（如 RGB）**；
    
- 这种方式大幅减少了参数数量，并捕捉局部特征（如边缘、角点）。
    

#### 2️⃣ 权重共享（Weight Sharing）

- 同一层中，位于同一个 **特征图（Feature Map）** 的所有神经元都使用**相同的一组卷积核权重**；
    
- 卷积核在图像中滑动，对每个位置应用同样的计算；
    
- 这不仅进一步减少了参数数量，还引入了**平移等变性**：即如果图像中的物体平移了，网络输出会相应平移。
    

#### 3️⃣ 池化操作（Pooling / Subsampling）

- 对局部神经元进行汇聚，如**最大池化（Max Pooling）**或**平均池化（Average Pooling）**；
    
- 这可以**减少空间分辨率**，同时保留关键特征；
    
- 对于小的几何变形（如轻微拉伸、旋转），具有**鲁棒性**，提高了模型的泛化能力。
    

#### 4️⃣ 特征提取与下采样交替进行

- CNN通常将卷积（特征提取）层与池化（降采样）层交替堆叠；
    
- 网络逐层提取更抽象、更具语义的特征；
    
- 这使得网络能够从像素逐步过渡到识别**高层语义类别（如猫、人脸、交通标志）**。
    


## CNN
### convolution layer
convolution filters 卷积滤波器，也称卷积核kernel

![[Pasted image 20250413152725.png|375]]
步长是卷积核移动的长度，一般是1
参数量：$[k\times k \times \text{Chanel} +1(\text{bias})]\times \text{number of kernel}$    


feature maps：（多个）卷积核作用后的输出结果图




spacial convolution：就是正常的卷积

在channel上做卷积，使用$1\times 1$作为卷积核，channel上进行加权：
![[Pasted image 20250413160205.png|325]]

complexity：
![[Pasted image 20250413153717.png|400]]


Receptive Field(感受野)：$k\times k$ 
- def：在神经网络中，一个神经元的**感受野**是指：它所依赖的原始输入图像中的区域大小。

### 卷积的激活函数选择



### Pooling layers & model complexity


作用：

1. **降低特征图的空间尺寸（Width × Height）**
    
2. **减少模型参数和计算量**
    
3. **增强特征的空间不变性（spatial invariance）**
    
4. **避免过拟合**


Reducing the spatial size of the feature maps


- max pooling：无需学习参数
- down sampling：convolution with stride of $| k|$ ，卷积核参数需要学习 
	- 许多相关的宏观特征往往会跨越图像的大部分区域，因此在卷积时采用步幅通常不会遗漏太多信息。

complexity：
![[Pasted image 20250413154524.png|475]]

F：池化窗口大小


 **卷积神经网络（CNN）的数学属性（Math Properties）**：


我们可以把 CNN 中的表示看作一种**抽象函数**：
$$
\phi(x) = \text{CNN表示}(x)
$$
其中 $x$ 是输入图像，$\phi(x)$ 是该图像在 CNN 某一层（或最终）的表示。


| 数学属性 | 含义 | 理想情况 |
|----------|------|----------|
| **不变性（invariance）** | 表示对变换保持不变 | $\phi(T(x)) = \phi(x)$ |
| **等变性（equivariance）** | 表示随变换同步变化，但有规律 | $\phi(T(x)) = T'(\phi(x))$，例如卷积对平移等变 |

| 变换类型 | 示例 | 描述 |
|----------|------|------|
| **平移（Translation）** | 向左或向右移动图像 | CNN 中的卷积操作对小平移等变 |
| **缩放（Scale）** | 放大或缩小图像 | 池化可以在一定程度上提供尺度不变性 |
| **旋转（Rotation）** | 顺时针或逆时针旋转图像 | 标准 CNN 不擅长对旋转不变，需要额外设计 |
| **局部形变（Local Deformation）** | 图像某部分被拉伸或扭曲 | 深层网络可以一定程度建模非线性形变 |
| **亮度/颜色变化** | 改变光照或颜色 | BatchNorm、数据增强等有助于处理此类变化 |

为什么这些性质重要？

- **泛化能力强**：模型对不同位置、大小、姿态的同类物体识别准确率更高；
- **鲁棒性强**：在真实世界中图像总是有噪声和变化；
- **设计指导**：帮助我们选择卷积结构、添加注意力机制或变换不变性模块（如 Spatial Transformer Networks、Group Equivariant CNNs 等）。

### examples

LeNet-5
![[Pasted image 20250413155119.png|550]]


 **Flatten（展平）操作** 是卷积神经网络（CNN）中非常常见的一步，它起到了连接 **卷积层/池化层** 和 **全连接层（Fully Connected Layer）** 的桥梁作用。

什么是 Flatten？

在卷积神经网络中，前面一系列卷积和池化层处理的输出是一个 **三维张量**（通常是：高 × 宽 × 通道数，即 feature map）。而全连接层（FC）接收的是一个 **一维向量**。

> 所以 **Flatten 就是将多维特征图“摊平”成一维向量**，使其能够作为全连接层的输入。

---

### 📦 图中 Flatten 过程解释：

图中最后一个池化层输出大小是：
$$
4 \times 4 \times n_2
$$
表示：4×4 的空间维度、n₂ 个通道。

经过 Flatten 操作后，输出变成：
$$
(4 \times 4 \times n_2) = 16 \times n_2
$$

Flatten 的作用总结：

| 功能 | 说明 |
|------|------|
| **维度转换** | 把卷积层输出的 3D 张量转换为 1D 向量 |
| **连接结构** | 将 CNN 特征提取部分（conv+pool）与分类器部分（FC）衔接 |
| **无参数操作** | Flatten 只是数据重排，没有可学习参数 |


```python
# PyTorch
x = torch.randn(1, 4, 4, 32)  # 假设是 4x4x32 的输出
x = x.view(x.size(0), -1)     # Flatten 操作

# TensorFlow / Keras
x = tf.keras.layers.Flatten()(input_tensor)
```



AlexNet

![[Pasted image 20250413155307.png]]

这里dense的input是flattern之后的一维向量吗
dense实际就是一个全连接层


### CNN Architecture


