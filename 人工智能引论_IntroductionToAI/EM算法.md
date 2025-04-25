
## EM算法（Expectation-Maximization Algorithm）

EM算法是一种用于含有**隐变量或缺失数据**的概率模型中的参数估计方法，尤其常用于极大似然估计（MLE）无法直接求解时的场景。

---

## 一、基本动机

当模型中包含隐变量（latent variable）$Z$，而我们只能观测到部分变量 $X$ 时，直接利用[MLE](../数学基础_MathBasics/概率论与数理统计_ProbabilityAndStatistics/数理统计推断_StatisticalInference#2.2极大似然估计（MLE）)最大化似然函数：

$$
\log p(X \mid \theta) = \log \sum_{Z} p(X, Z \mid \theta)
$$

在大多数情况下，由于对数和不能交换，**该式难以直接求导优化**。

---

## 二、EM算法的思想

EM算法通过引入**隐变量的后验分布**，将原本难以优化的目标函数转换为迭代优化两个子问题的形式：

> - **E步（Expectation）：** 计算期望  
> - **M步（Maximization）：** 最大化参数

通过构造下界并反复优化，EM算法保证每步都不会降低原始的对数似然函数。

---

## 三、EM算法的基本流程

设 $X$ 为观测数据，$Z$ 为隐变量，$\theta$ 为模型参数。


### Step 1：初始化参数 $\theta^{(0)}$

在开始 EM 迭代之前，需要为参数 $\theta$ 设定一个初始值。这个初值会影响最终收敛的结果，因为 EM 可能陷入局部最优。

---

### Step 2：E 步（Expectation Step） — 计算期望

目标：在当前参数 $\theta^{(t)}$ 下，估计隐变量 $Z$ 的**后验分布**，并计算联合对数似然的期望。

定义如下：

$$
Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z \sim p(Z \mid X, \theta^{(t)})} \left[ \log p(X, Z \mid \theta) \right]
$$

更具体地写为：

$$
Q(\theta, \theta^{(t)}) = \sum_Z p(Z \mid X, \theta^{(t)}) \cdot \log p(X, Z \mid \theta)
$$

#### 💡 含义解释：

- $p(Z \mid X, \theta^{(t)})$：在旧参数下，估计“隐变量”的后验分布；
- $p(X, Z \mid \theta)$：联合概率的对数；
- $Q$ 是一个关于新参数 $\theta$ 的目标函数，它是**对完整数据对数似然的期望**。

---

### Step 3：M 步（Maximization Step） — 最大化期望

目标：找到新的参数 $\theta^{(t+1)}$ 来最大化刚刚构造的 $Q$ 函数：

$$
\theta^{(t+1)} = \arg\max_{\theta} Q(\theta, \theta^{(t)})
$$

#### 💡 含义解释：

- 把 E 步中计算的“软标签”作为参考，重新估计参数；
- 相当于“在带权样本下重新最大化似然”。

---

### Step 4：迭代直到收敛

通常的收敛标准：

- 对数似然增益足够小：
  $$
  \log p(X \mid \theta^{(t+1)}) - \log p(X \mid \theta^{(t)}) < \varepsilon
  $$
- 或者参数变化幅度足够小。

---

## 四、EM算法的数学解释：变分下界

我们要最大化的目标是：

$$
\log p(X \mid \theta) = \log \sum_{Z} p(X, Z \mid \theta)
$$

通过引入任意分布 $q(Z)$：

$$
\log p(X \mid \theta) = \mathcal{L}(q, \theta) + D_{KL}(q(Z) \| p(Z \mid X, \theta))
$$

其中：

- $\mathcal{L}(q, \theta) = \sum_Z q(Z) \log \frac{p(X, Z \mid \theta)}{q(Z)}$
- $D_{KL}$ 是[ KL 散度](../数学基础_MathBasics/信息论_InformationTheory#相对熵（KL散度）)，非负

由于 KL 散度非负，$\mathcal{L}(q, \theta)$ 是 $\log p(X \mid \theta)$ 的下界。

> - **E步**：令 $q(Z) = p(Z \mid X, \theta^{(t)})$，使 KL 散度最小 ，让下界
> - **M步**：固定 $q(Z)$，最大化下界 $\mathcal{L}$，等价于最大化 $Q(\theta, \theta^{(t)})$

---

在含隐变量的情形中，我们的优化目标是：

$$
\log p(X \mid \theta) = \log \sum_Z p(X, Z \mid \theta)
$$

这个目标函数通常 **难以直接最大化**。于是我们引入了一个任意分布 $q(Z)$，利用 Jensen 不等式构造了下界（变分下界）：

$$
\log p(X \mid \theta) \geq \mathcal{L}(q, \theta)
$$

其中：

$$
\mathcal{L}(q, \theta) = \mathbb{E}_{q(Z)} \left[ \log \frac{p(X, Z \mid \theta)}{q(Z)} \right] = \sum_Z q(Z) \log \frac{p(X, Z \mid \theta)}{q(Z)}
$$

>[!tip] 
>这是 EM 的数学基础：**最大化下界来间接最大化原目标**。


### 关键观察：$\mathcal{L}(q, \theta)$ 可以分解为：

$$
\mathcal{L}(q, \theta) = \underbrace{\sum_Z q(Z) \log p(X, Z \mid \theta)}_{\text{我们称之为 } Q(\theta, \theta^{(t)})} - \underbrace{\sum_Z q(Z) \log q(Z)}_{\text{熵项，与 } \theta \text{ 无关}}
$$

当我们**固定 $q(Z)$** 时，第二项为常数。因此：

> **最大化 $\mathcal{L}(q, \theta)$ 等价于最大化第一项，也就是 EM 中的 Q 函数！**

也就是说：

$$
\boxed{
\theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q, \theta) = \arg\max_\theta Q(\theta, \theta^{(t)})
}
$$

---

### ✳️ 三、直观理解

- E步：我们令 $q(Z) = p(Z \mid X, \theta^{(t)})$，这是让下界最紧；
- M步：我们在这个固定的 $q(Z)$ 下最大化 $\mathcal{L}(q, \theta)$；
- 因为 $\mathcal{L}$ 中与 $\theta$ 相关的部分就是 $Q$，所以最大化 $\mathcal{L}$ 就是最大化 $Q$！

---






## 五、EM算法的性质

✅ 优点：

- 每一步都保证使对数似然非下降；
- 非参数化，不需要梯度信息；
- 易于实现，尤其适合含隐变量的模型。

❌ 缺点：

- 易陷入局部最优；
- 依赖初始参数选择；
- E步可能需要近似或采样计算。

---

## 六、经典应用：高斯混合模型（GMM）

设数据来自 $K$ 个高斯分布的混合：

$$
p(x) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

其中：

- $\pi_k$：第 $k$ 个高斯分量的权重
- $z_i \in \{1, ..., K\}$：表示 $x_i$ 属于哪个分量（为隐变量）

### GMM 中的 EM算法步骤

#### E步：

计算后验概率：

$$
\gamma_{ik} = P(z_i = k \mid x_i; \theta^{(t)}) = \frac{\pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j \cdot \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
$$

#### M步：

更新参数：

- 混合系数：

  $$
  \pi_k = \frac{1}{n} \sum_{i=1}^n \gamma_{ik}
  $$

- 均值：

  $$
  \mu_k = \frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}}
  $$

- 协方差：

  $$
  \Sigma_k = \frac{\sum_{i=1}^n \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^n \gamma_{ik}}
  $$

---

## 七、其他应用场景

- 隐马尔可夫模型（HMM）的 Baum-Welch 算法；
- 潜在狄利克雷分配（LDA）的近似推断；
- 软聚类与混合密度建模；
- 带缺失数据的最大似然估计。

---

## 八、总结

| 步骤 | 内容 |
|------|------|
| E步 | 计算隐变量的后验分布，构造期望下界 |
| M步 | 最大化该期望，更新参数 |
| 优势 | 对含隐变量的模型非常有效 |
| 局限 | 易陷入局部最优，初值敏感 |
