

## 决策树回归（Decision Tree Regression）

决策树回归是一种非参数的监督学习算法，用于预测**连续值**。它通过将输入特征空间分割成若干个区域，并在每个区域上预测一个固定值来实现回归建模。

---

## 一、基本思想

与分类树类似，决策树回归也使用“**特征划分 + 递归构造子树**”的方法，但不同的是：

- 分类树输出的是类别标签；
- 回归树输出的是每个叶子节点对应的**实数值**。

### 模型结构：

- 每个非叶子节点表示一个特征划分条件；
- 每个叶子节点表示一个实数值（例如训练样本的均值）；
- 模型预测时根据输入特征沿树路径遍历，最终到达叶子节点并输出其值。

---

## 二、构建回归树的流程

给定训练数据集：

$$
\mathcal{D} = \{(\boldsymbol{x}^{(i)}, y^{(i)})\}_{i=1}^m
$$

每次划分的目标：

选定某个特征 $j$ 和一个划分点 $s$，将数据集划分为两部分：

$$
\begin{aligned}
\mathcal{D}_\text{left} &= \{(\boldsymbol{x}^{(i)}, y^{(i)}) \mid x_j^{(i)} \leq s \} \\
\mathcal{D}_\text{right} &= \{(\boldsymbol{x}^{(i)}, y^{(i)}) \mid x_j^{(i)} > s \}
\end{aligned}
$$

**最小化两边的总误差：**

$$
\text{Loss} = \sum_{i \in \mathcal{D}_\text{left}} (y^{(i)} - \bar{y}_\text{left})^2 + \sum_{i \in \mathcal{D}_\text{right}} (y^{(i)} - \bar{y}_\text{right})^2
$$

其中，$\bar{y}_\text{left}$ 和 $\bar{y}_\text{right}$ 分别是左右子节点中样本的平均值。



这个公式是用来衡量**在“当前节点”下进行某一次划分之后的“总残差平方和”**（Residual Sum of Squares, RSS）。

也就是说，它**只和当前这个节点的一次划分有关**，用于判断**要不要在这个地方划分、怎么划分**。

这个 Loss 是为了找到“当前这个节点”上最优的划分方式（feature 和划分点）而定义的目标函数。

#### 在当前节点 $D$：

- 我们尝试遍历所有可能的划分特征 $j$ 和划分点 $s$；
- 每一个候选划分 $(j, s)$ 会将当前样本集 $D$ 分为左右两部分 $\mathcal{D}_\text{left}, \mathcal{D}_\text{right}$；
- 然后计算对应的 Loss：

$$
\text{Loss}_{(j, s)} = \text{RSS}_\text{left} + \text{RSS}_\text{right}
$$

- 最终选择那个使 Loss 最小的划分方式。

这个 Loss **不是整个树的总损失**，也不是从根节点开始的累加，而是**构造某个具体节点的局部最优划分**时使用的。

整个树的训练过程是一个**贪心递归构建**过程，每次只考虑当前节点的最优划分，而不是全局最优。


---

## 三、停止条件与剪枝

- 当样本数量小于某个阈值；
- 或者不能找到有效划分时停止划分。

可进行**剪枝（Pruning）**操作，减少模型复杂度，防止过拟合：
- 预剪枝（限制最大深度、最小样本数）
- 后剪枝（先构建完全树再进行剪枝）

---

## 四、预测过程

给定一个新的样本 $\boldsymbol{x}$，从根节点开始，依次根据条件划分进入左子树或右子树，直到到达一个叶子节点，返回该叶子的预测值 $\hat{y}$：

$$
\hat{y} = \text{mean} \left\{ y^{(i)} \mid (\boldsymbol{x}^{(i)}, y^{(i)}) \in \text{叶子节点} \right\}
$$
i.e.
对于训练好的回归树 $T$，我们将输入 $\boldsymbol{x}$ 映射到某个叶子节点 $L$：

$$
\boldsymbol{x} \in \text{Region}_L
$$

预测值为：

$$
\hat{y} = \frac{1}{|\mathcal{D}_L|} \sum_{(\boldsymbol{x}^{(i)}, y^{(i)}) \in \mathcal{D}_L} y^{(i)}
$$

其中：

- $\mathcal{D}_L$ 表示落入叶子节点 $L$ 的训练样本集合；
- $|\mathcal{D}_L|$ 是该叶子节点中的样本数量。

即：**预测值是该区域内所有训练样本目标值的均值**。


---

## 五、决策树回归优缺点

### ✅ 优点：

- 非线性建模能力强；
- 可解释性好（结构直观）；
- 对特征预处理要求低；
- 可以处理混合类型特征（数值和类别）。

### ❌ 缺点：

- 容易过拟合（尤其是深树）；
- 对训练集中的小变化敏感；
- 不能很好地预测连续变量的边界外区域（无法外推）。

---

## 六、Python 示例（使用 scikit-learn）

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 生成数据
X, y = make_regression(n_samples=100, n_features=1, noise=15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. 模型训练
model = DecisionTreeRegressor(max_depth=3)
model.fit(X_train, y_train)

# 3. 预测
y_pred = model.predict(X_test)

# 4. 可视化预测效果
plt.scatter(X_test, y_test, color='blue', label='True')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.title("Decision Tree Regression")
plt.show()
```

