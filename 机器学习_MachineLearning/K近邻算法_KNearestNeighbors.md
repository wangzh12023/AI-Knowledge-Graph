
## K最近邻算法（K-Nearest Neighbors, KNN）

K最近邻算法是一种**监督学习算法**，既可以用于分类问题，也可以用于回归问题。其基本思想是：对于一个待预测样本，找出训练集中与其最接近的 $K$ 个样本，通过“投票”或“加权平均”等方式进行预测。

---

## 一、KNN 的基本原理

给定一个训练数据集：
$$
\mathcal{D} = \{(\boldsymbol{x}^{(i)}, y^{(i)})\}_{i=1}^{m}
$$

对于一个新的样本点 $\boldsymbol{x}$，其预测步骤为：

1. 计算 $\boldsymbol{x}$ 与训练集中每个点的距离；
2. 选出距离最近的 $K$ 个点；
3. **分类任务**：根据 $K$ 个邻居中最多的类别进行预测；
4. **回归任务**：根据 $K$ 个邻居的平均值（或加权平均）进行预测。

---

## 二、距离度量方式

最常用的是**欧几里得距离**：

$$
d(\boldsymbol{x}, \boldsymbol{x}^{(i)}) = \sqrt{ \sum_{j=1}^{n} (x_j - x^{(i)}_j)^2 }
$$

其他常见距离：

- 曼哈顿距离（L1 距离）：
  $$
  d(\boldsymbol{x}, \boldsymbol{x}^{(i)}) = \sum_{j=1}^{n} |x_j - x^{(i)}_j|
  $$

- 闵可夫斯基距离：
  $$
  d(\boldsymbol{x}, \boldsymbol{x}^{(i)}) = \left( \sum_{j=1}^{n} |x_j - x^{(i)}_j|^p \right)^{1/p}
  $$

- 余弦相似度（常用于文本）：
  $$
  \cos(\boldsymbol{x}, \boldsymbol{x}^{(i)}) = \frac{\boldsymbol{x} \cdot \boldsymbol{x}^{(i)}}{ \|\boldsymbol{x}\| \cdot \|\boldsymbol{x}^{(i)}\| }
  $$

---

## 三、KNN 的分类预测方式

对于分类问题，KNN 通常使用**多数投票法**：

- 假设 $K=5$，其中 3 个邻居是类别 A，2 个邻居是类别 B，则预测类别为 A。

如果使用加权投票（距离越近权重越大）：

$$
\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_K} \mathbf{1}(y^{(i)} = c) \cdot \frac{1}{d(\boldsymbol{x}, \boldsymbol{x}^{(i)})}
$$

其中 $\mathcal{N}_K$ 表示 $K$ 个最近邻的下标集合。

---

## 四、KNN 的回归预测方式

对于回归问题，预测值为邻居的平均值：

$$
\hat{y} = \frac{1}{K} \sum_{i \in \mathcal{N}_K} y^{(i)}
$$

或使用距离加权平均：

$$
\hat{y} = \frac{ \sum_{i \in \mathcal{N}_K} \frac{1}{d(\boldsymbol{x}, \boldsymbol{x}^{(i)})} \cdot y^{(i)} }{ \sum_{i \in \mathcal{N}_K} \frac{1}{d(\boldsymbol{x}, \boldsymbol{x}^{(i)})} }
$$

---

## 五、K 值选择

- **K 值过小**：对噪声敏感，容易过拟合；
- **K 值过大**：邻居包含过多异类，容易欠拟合；
- 通常使用交叉验证来选择合适的 $K$ 值。

---

## 六、KNN 的优缺点

### ✅ 优点：

- 理解简单，易于实现；
- 不需要训练过程（懒惰学习）；
- 对多分类问题天然支持。

### ❌ 缺点：

- 预测开销大，计算复杂度高（每次都要遍历训练集）；
- 对维度敏感（“维度灾难”）；
- 对异常值敏感；
- 特征尺度不一致会影响效果（需要归一化）。

---

## 七、改进方法

1. **特征归一化（Standardization / MinMax）**
2. **使用 KD-Tree / Ball-Tree 加速邻居搜索**
3. **使用局部加权、核函数等改进预测策略**

---

## 八、KNN 示例（使用 scikit-learn）

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 数据准备
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2. 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 模型训练与预测
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# 4. 评估准确率
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)
