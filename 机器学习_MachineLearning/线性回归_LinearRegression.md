

## 线性回归（Linear Regression）

线性回归是一种用于建模因变量（目标变量）与一个或多个自变量（特征）之间线性关系的监督学习算法。其基本思想是找到一组最优的线性系数，使得模型预测值与真实值之间的误差最小。

---

## 一、模型形式

对于给定的输入特征向量 $$\boldsymbol{x} = (x_1, x_2, \dots, x_n)^T$$，线性回归的预测模型为：

$$
\hat{y} = \boldsymbol{w}^T \boldsymbol{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

其中：

- $\boldsymbol{w} \in \mathbb{R}^n$ 是权重向量
- $b \in \mathbb{R}$ 是偏置项（截距）
- $\hat{y}$ 是预测值

---

## 二、损失函数：最小二乘法（MSE）

为了度量模型预测值与真实值之间的误差，线性回归常用**均方误差（MSE）**作为损失函数：

给定数据集 $$\mathcal{D} = \{(\boldsymbol{x}^{(i)}, y^{(i)})\}_{i=1}^m$$，损失函数为：

$$
J(\boldsymbol{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 = \frac{1}{2m} \sum_{i=1}^{m} \left( \boldsymbol{w}^T \boldsymbol{x}^{(i)} + b - y^{(i)} \right)^2
$$

引入 $\frac{1}{2}$ 的目的是在后续求导时简化表达。

---

## 三、参数求解

### 1. 梯度下降法

[梯度下降](梯度下降算法_GradientDescentAlgorithms)是一种数值优化方法，通过不断更新参数来最小化损失函数。

**参数更新公式为**：

$$
\boldsymbol{w} := \boldsymbol{w} - \eta \cdot \nabla_{\boldsymbol{w}} J(\boldsymbol{w}, b)
$$

$$
b := b - \eta \cdot \nabla_b J(\boldsymbol{w}, b)
$$

其中 $\eta$ 是学习率。

计算偏导：

$$
\frac{\partial J}{\partial \boldsymbol{w}} = \frac{1}{m} \sum_{i=1}^{m} \left( \boldsymbol{w}^T \boldsymbol{x}^{(i)} + b - y^{(i)} \right) \boldsymbol{x}^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \boldsymbol{w}^T \boldsymbol{x}^{(i)} + b - y^{(i)} \right)
$$

过程：

![[Pasted image 20250415090954.png|725]]

---

### 2. 解析解（正规方程）

如果数据规模不大，可以直接使用最小二乘法的解析解：

$$
\boldsymbol{w} = \left( X^T X \right)^{-1} X^T \boldsymbol{y}
$$

其中：

- $X \in \mathbb{R}^{m \times n}$ 是输入数据矩阵
- $\boldsymbol{y} \in \mathbb{R}^m$ 是目标向量

偏置项 $b$ 通常可以通过将一个常数列向量 $\boldsymbol{1}$ 加入 $X$ 中来合并到 $\boldsymbol{w}$ 中处理。

---

## 四、线性回归的几何解释

线性回归实质上是**在特征空间中拟合一个超平面**，使得所有样本点到该超平面的垂直距离的平方和最小。

---

## 五、评估指标

- 均方误差（MSE）：
  
  $$
  \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
  $$

- 决定系数（R² Score）：
  
  $$
  R^2 = 1 - \frac{\sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^{m} (y^{(i)} - \bar{y})^2}
  $$

  其中 $\bar{y}$ 是所有 $y^{(i)}$ 的平均值。

---

## 六、正则化扩展

为防止过拟合，可引入正则项，进行[正则化]()：

### 岭回归（L2 正则化）：

$$
J(\boldsymbol{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2} \|\boldsymbol{w}\|^2
$$

### Lasso 回归（L1 正则化）：

$$
J(\boldsymbol{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \lambda \|\boldsymbol{w}\|_1
$$

---

## 七、优缺点总结

**优点：**

- 模型简单、可解释性强
- 计算代价小
- 适合线性关系建模

**缺点：**

- 无法建模非线性关系
- 对异常值敏感
- 对特征之间的多重共线性敏感

---

## 八、代码示例（使用 scikit-learn）

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print("MSE:", mse)

```

