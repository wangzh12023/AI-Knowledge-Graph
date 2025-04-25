


>[!note] 
>内容由ChatGPT生成


## 支持向量机（Support Vector Machines, SVM）总结

支持向量机（SVM）是一种强大的监督学习方法，广泛用于分类和回归任务。其核心思想是在特征空间中寻找一个最优超平面，以最大化分类间隔。

---

## 1. 基本思想

SVM 的目标是找到一个能够**最大间隔**地分隔正负类的超平面：

设训练集为：
$$
\mathcal{D} = \{ (\boldsymbol{x}_1, y_1), (\boldsymbol{x}_2, y_2), \dots, (\boldsymbol{x}_m, y_m) \}
$$

其中 $\boldsymbol{x}_i \in \mathbb{R}^n$, $y_i \in \{-1, +1\}$。

超平面定义为：
$$
\boldsymbol{w}^T \boldsymbol{x} + b = 0
$$

分类函数为：
$$
f(\boldsymbol{x}) = \text{sign}(\boldsymbol{w}^T \boldsymbol{x} + b)
$$

---

## 2. 间隔与支持向量

### 几何间隔（Geometric Margin）：
给定样本 $(\boldsymbol{x}_i, y_i)$，其到超平面的几何间隔为：（通过点到平面距离或者向量的投影长推导）
$$
\gamma_i = \frac{y_i(\boldsymbol{w}^T \boldsymbol{x}_i + b)}{\|\boldsymbol{w}\|}
$$

SVM 目标：最大化所有样本的最小几何间隔，即：
$$
\max_{\boldsymbol{w}, b} \min_i \gamma_i
$$

通过约束最小几何间隔为 1，可转化为如下优化问题。
几何间隔关于 $(\mathbf{w}, b)$ 是**缩放不变的**：

$$
\frac{y_i(\lambda \mathbf{w}^\top \mathbf{x}_i + \lambda b)}{\|\lambda \mathbf{w}\|} = \frac{\lambda y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\lambda \|\mathbf{w}\|} = \frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|}
$$

因此我们可以不失一般性地**固定 margin 的下界为 1**，即约定：

$$
y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \quad \forall i
$$


这一步称为**函数间隔归一化**。

而只有最近的点才能取到1，所以$\min_i \gamma_{i}$ 对应的$i$ 能够满足$y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$
所以可以完成代换转化成
$$
\begin{aligned}
& \max_{\boldsymbol{w}, b} \quad  \frac{1}{\|\boldsymbol{w}\|^2} \\
& \text{subject to} \quad y_i(\boldsymbol{w}^T \boldsymbol{x}_i + b) \geq 1, \quad i=1,\dots,m
\end{aligned}
$$
但是为了后续求导方便和为了让函数变为凸函数，我们通常会引入一些常数并且进行形式优化：

---

## 3. 硬间隔 SVM（线性可分情形）

### 原始问题（Primal Form）：

变成了一个优化问题
$$
\begin{aligned}
& \min_{\boldsymbol{w}, b} \quad \frac{1}{2} \|\boldsymbol{w}\|^2 \\
& \text{subject to} \quad y_i(\boldsymbol{w}^T \boldsymbol{x}_i + b) \geq 1, \quad i=1,\dots,m
\end{aligned}
$$

---

## 4. 拉格朗日对偶形式（Dual Form）

通过拉格朗日乘子法，引入 $\alpha_i \geq 0$，构造拉格朗日函数：
$$
L(\boldsymbol{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \|\boldsymbol{w}\|^2 - \sum_{i=1}^m \alpha_i [y_i(\boldsymbol{w}^T \boldsymbol{x}_i + b) - 1]
$$

对偶问题为：
$$
\begin{aligned}
& \max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^T \boldsymbol{x}_j \\
& \text{subject to} \quad \alpha_i \geq 0, \quad \sum_{i=1}^m \alpha_i y_i = 0
\end{aligned}
$$

---

## 5. 支持向量与决策函数

- $\alpha_i > 0$ 的训练样本称为**支持向量（Support Vectors）**
- 最终分类决策函数：
$$
f(\boldsymbol{x}) = \text{sign}\left( \sum_{i \in SV} \alpha_i y_i \boldsymbol{x}_i^T \boldsymbol{x} + b \right)
$$

---

## 6. 软间隔 SVM（线性不可分情形）

引入松弛变量 $\xi_i \geq 0$ 允许一定程度的误分类，并通过正则化项进行惩罚控制。目标函数变为：

### 原始问题：
$$
\begin{aligned}
& \min_{\boldsymbol{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2} \|\boldsymbol{w}\|^2 + C \sum_{i=1}^m \xi_i \\
& \text{subject to} \quad y_i(\boldsymbol{w}^T \boldsymbol{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
\end{aligned}
$$

其中 $C > 0$ 是正则化参数，用于控制间隔最大化和误差惩罚之间的权衡。

各变量含义

| 符号 | 含义 |
|------|------|
| $\boldsymbol{w}$ | 超平面法向量 |
| $b$ | 偏置项 |
| $\xi_i$ | 松弛变量，表示样本违反 margin 的程度 |
| $C$ | 正则化超参数，权衡 margin 与违约惩罚 |


目标函数由两部分组成：

- $\frac{1}{2} \|\boldsymbol{w}\|^2$：最大化 margin，即使分界更宽；
- $C \sum_{i=1}^m \xi_i$：惩罚违反 margin 的样本，$\xi_i$ 越大，惩罚越大。

参数 $C$ 控制两者之间的权衡：
- $C$ 越大：更关注正确分类（更少错误），但可能过拟合；
- $C$ 越小：更关注 margin 的宽度，泛化能力更强。

---


主约束条件：

$$
y_i(\boldsymbol{w}^\top \boldsymbol{x}_i + b) \geq 1 - \xi_i
$$

几种情形：

- $\xi_i = 0$：样本满足硬间隔约束，分类正确且在 margin 外；
- $0 < \xi_i < 1$：样本落在 margin 内部但仍被正确分类；
- $\xi_i \geq 1$：样本被错误分类。

---

## 7. 核函数与非线性 SVM

当样本非线性不可分时，使用[核方法](核方法_KernelMethod)将数据映射到高维空间，使其线性可分。

设映射函数 $\phi(\boldsymbol{x})$，核函数定义为：
$$
K(\boldsymbol{x}_i, \boldsymbol{x}_j) = \phi(\boldsymbol{x}_i)^T \phi(\boldsymbol{x}_j)
$$

常用核函数：

- 线性核：
  $$
  K(\boldsymbol{x}, \boldsymbol{z}) = \boldsymbol{x}^T \boldsymbol{z}
  $$

- 多项式核：
  $$
  K(\boldsymbol{x}, \boldsymbol{z}) = (\boldsymbol{x}^T \boldsymbol{z} + c)^d
  $$

- 高斯核（RBF）：
  $$
  K(\boldsymbol{x}, \boldsymbol{z}) = \exp\left(-\frac{\|\boldsymbol{x} - \boldsymbol{z}\|^2}{2\sigma^2}\right)
  $$

---

## 8. 支持向量机的优化算法

- **SMO（Sequential Minimal Optimization）**：经典对偶问题的高效解法
- **内点法、梯度下降法**：用于原始问题的求解（尤其在大规模数据上）

---

## 9. SVM 用于回归（SVR）

支持向量回归（Support Vector Regression）使用 $\varepsilon$-不敏感损失函数：

$$
\begin{aligned}
& \min_{\boldsymbol{w}, b, \xi_i, \xi_i^*} \quad \frac{1}{2} \|\boldsymbol{w}\|^2 + C \sum_{i=1}^m (\xi_i + \xi_i^*) \\
& \text{subject to} \\
& y_i - (\boldsymbol{w}^T \boldsymbol{x}_i + b) \leq \varepsilon + \xi_i \\
& (\boldsymbol{w}^T \boldsymbol{x}_i + b) - y_i \leq \varepsilon + \xi_i^* \\
& \xi_i, \xi_i^* \geq 0
\end{aligned}
$$

---

## 10. 优缺点总结

### 优点
- 理论基础坚实（统计学习理论）
- 全局最优解
- 适用于高维小样本数据
- 可以通过核方法处理非线性问题

### 缺点
- 对参数（$C$, $\sigma$, 核函数）敏感
- 对大规模数据训练较慢
- 不直接给出概率输出（可借助 Platt scaling）

