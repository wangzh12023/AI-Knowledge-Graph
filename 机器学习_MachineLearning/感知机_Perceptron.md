

>[!note] 
>内容由ChatGPT生成



## 感知机（Perceptron）总结

感知机是最早的二分类线性模型之一，是机器学习中用于二分类问题的基础算法。它由 Frank Rosenblatt 于1957年提出，是神经网络的雏形。

---

## 1. 模型定义

感知机试图学习一个线性判别函数，对输入样本进行二分类。

设输入特征向量为：
$$
\boldsymbol{x} = [x_1, x_2, \dots, x_n]^T \in \mathbb{R}^n
$$

权重向量为：
$$
\boldsymbol{w} = [w_1, w_2, \dots, w_n]^T \in \mathbb{R}^n, \quad b \in \mathbb{R}
$$

模型的输出为：
$$
f(\boldsymbol{x}) = \text{sign}(\boldsymbol{w}^T \boldsymbol{x} + b)
$$

其中 $\text{sign}(\cdot)$ 为符号函数：
$$
\text{sign}(z) = 
\begin{cases}
+1, & z > 0 \\
-1, & z \leq 0
\end{cases}
$$

---

## 2. 假设空间与几何解释

- 感知机的决策边界是一个超平面：
$$
\boldsymbol{w}^T \boldsymbol{x} + b = 0
$$

- 正例位于超平面一侧，负例位于另一侧。

---

## 3. 学习策略

### 3.1 数据集表示

设训练集为：
$$
\mathcal{D} = \{ (\boldsymbol{x}_1, y_1), (\boldsymbol{x}_2, y_2), \dots, (\boldsymbol{x}_m, y_m) \}
$$

其中：
- $\boldsymbol{x}_i \in \mathbb{R}^n$
- $y_i \in \{-1, +1\}$ 表示类别标签

---

### 3.2 损失函数

感知机采用的是 **误分类样本的损失函数**：

$$
L(\boldsymbol{w}, b) = -\sum_{i \in \mathcal{M}} y_i(\boldsymbol{w}^T \boldsymbol{x}_i + b)
$$

其中 $\mathcal{M}$ 是当前被误分类的样本集合。

---

### 3.3 更新规则（梯度下降法）

对每一个被误分类的样本 $(\boldsymbol{x}_i, y_i)$，按以下规则更新参数：

- 权重更新：
$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + \eta y_i \boldsymbol{x}_i
$$

- 偏置更新：
$$
b \leftarrow b + \eta y_i
$$

其中 $\eta > 0$ 是学习率。

>[!note] 
>这里可以证明新参数一定比原参数更优



---

## 4. 感知机学习算法（原始形式）


输入：训练集 $\mathcal{D}$，学习率 $\eta$
初始化：$\boldsymbol{w} = 0, b = 0$
重复：
    对每个样本 $(\boldsymbol{x}_i, y_i)$：
        若 $y_i(\boldsymbol{w}^T \boldsymbol{x}_i + b) \leq 0$：
            $\boldsymbol{w} \leftarrow \boldsymbol{w} + \eta y_i \boldsymbol{x}_i$
            $b \leftarrow b + \eta y_i$
直到没有误分类样本


---

## 5. 感知机对偶形式

定义：
- 拉格朗日乘子 $\alpha_i \geq 0$
- 样本之间的内积：$K(\boldsymbol{x}_i, \boldsymbol{x}_j) = \boldsymbol{x}_i^T \boldsymbol{x}_j$

权重向量表示为：
$$
\boldsymbol{w} = \sum_{i=1}^m \alpha_i y_i \boldsymbol{x}_i
$$

预测函数为：
$$
f(\boldsymbol{x}) = \text{sign}\left( \sum_{i=1}^m \alpha_i y_i \boldsymbol{x}_i^T \boldsymbol{x} + b \right)
$$

更新规则变为更新 $\alpha_i$ 而不是直接更新 $\boldsymbol{w}$。

---

## 6. 感知机收敛性定理

假设训练数据线性可分，存在 $\boldsymbol{w}^*$ 和 $b^*$，使得：
$$
y_i(\boldsymbol{w}^{*T} \boldsymbol{x}_i + b^*) > 0, \quad \forall i
$$

则感知机算法在有限步内必然停止，即算法收敛。

---

## 7. 优缺点分析

### 优点
- 实现简单
- 易于理解
- 在线性可分数据上可保证收敛

### 缺点
- 只能处理线性可分问题
- 对噪声敏感
- 无法输出概率，无法处理非线性问题

---

## 8. 感知机与支持向量机（SVM）比较

| 项目 | 感知机 | 支持向量机 |
|------|--------|-------------|
| 决策边界 | 任意线性分割面 | 最大间隔分割面 |
| 损失函数 | 误分类损失 | Hinge 损失 |
| 对噪声的鲁棒性 | 差 | 强 |
| 是否支持软间隔 | 否 | 是 |
| 优化方法 | 在线学习 | 凸优化 |

---

## 9. 感知机的扩展

- 多类别感知机（One-vs-All）
- 内积核方法（可扩展为核感知机）
- 与神经网络中激活函数 ReLU 的历史关联
