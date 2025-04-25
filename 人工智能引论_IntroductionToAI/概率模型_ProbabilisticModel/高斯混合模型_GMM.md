

## 高斯混合模型（Gaussian Mixture Model, GMM）

高斯混合模型是一种经典的**生成式聚类模型**，假设数据由多个不同的高斯分布组成。

---

## 一、模型定义

高斯混合模型认为观测数据 $x$ 来自 $K$ 个高斯分布的加权组合：

$$
p(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

其中：

- $\pi_k$ 是第 $k$ 个高斯分量的**混合权重**，满足：
  $$
  \sum_{k=1}^K \pi_k = 1,\quad 0 \leq \pi_k \leq 1
  $$

- $\mathcal{N}(x \mid \mu_k, \Sigma_k)$ 是均值为 $\mu_k$、协方差为 $\Sigma_k$ 的**高斯分布密度函数**：

  $$
  \mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left( -\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu) \right)
  $$

---

## 二、图像直觉

- 每个高斯分布对应数据中的一个“簇”；
- 混合权重 $\pi_k$ 表示该簇的重要性；
- 多个高斯分布组合可以表示复杂的、非球形的数据分布。

---

## 三、引入隐变量

我们引入一个**隐变量** $z_i$ 表示第 $i$ 个样本属于哪一个高斯分量：

- $z_i \in \{1, 2, ..., K\}$
- 其先验分布为 $\pi_k$
- 条件分布为 $x_i \sim \mathcal{N}(\mu_{z_i}, \Sigma_{z_i})$

于是联合分布可写为：

$$
p(x_i, z_i = k) = \pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \Sigma_k)
$$

边缘化 $z_i$ 后得到混合分布 $p(x_i)$。

---



## 四、参数估计：EM算法

GMM 中的参数 $\theta = \{\pi_k, \mu_k, \Sigma_k\}$ 可用 **EM算法** 估计。
### 直观理解：最大化联合概率分布


- 第一步：写出联合概率

根据生成过程（隐变量决定分量 + 条件高斯分布生成数据）：

$$
P(y_1, ..., y_n, \gamma \mid \theta) = \prod_{j=1}^{n} P(y_j, \gamma_j \mid \theta)
$$

每个 $y_j$ 的生成可以写为：

$$
P(y_j, \gamma_j \mid \theta) = \prod_{k=1}^K \left[ \pi_k \cdot \mathcal{N}(y_j \mid \mu_k, \sigma_k) \right]^{\gamma_{jk}}
$$

这利用了 one-hot 编码的隐变量：只有 $\gamma_{jk}=1$ 的项会保留，其余变为 $1$。

---

- 第二步：合并整合写法

整体联合概率为：

$$
P(y_1, ..., y_n, \gamma \mid \theta) 
= \prod_{j=1}^{n} \prod_{k=1}^{K} \left[ \pi_k \cdot \mathcal{N}(y_j \mid \mu_k, \sigma_k) \right]^{\gamma_{jk}}
$$

可进一步拆分为两部分：

$$
= \prod_{j=1}^{n} \prod_{k=1}^{K} \pi_k^{\gamma_{jk}} \cdot \mathcal{N}(y_j \mid \mu_k, \sigma_k)^{\gamma_{jk}}
$$

---
- 第三步：整理乘积顺序（交换两个乘积）

交换乘积顺序，先按 $k$ 聚合：

$$
= \prod_{k=1}^{K} \prod_{j=1}^{n} \left[ \pi_k^{\gamma_{jk}} \cdot \mathcal{N}(y_j \mid \mu_k, \sigma_k)^{\gamma_{jk}} \right]
$$

合并为：

$$
= \prod_{k=1}^{K} \pi_k^{n_k} \prod_{j=1}^{n} \mathcal{N}(y_j \mid \mu_k, \sigma_k)^{\gamma_{jk}}
$$

其中：

$$
n_k = \sum_{j=1}^{n} \gamma_{jk}
$$

表示来自第 $k$ 个分量的样本数量。

---

- 第四步：使用 $\phi(y_j \mid \theta_k)$ 记号

令：

$$
\phi(y_j \mid \theta_k) = \mathcal{N}(y_j \mid \mu_k, \sigma_k)=\frac{1}{\sqrt{2\pi} \sigma_k} \exp\left( -\frac{(y_j - \mu_k)^2}{2\sigma_k^2} \right)
$$


所以最终联合概率可写为：

$$
P(y_1, ..., y_n, \gamma \mid \theta) 
= \prod_{k=1}^{K} \pi_k^{n_k} \cdot \prod_{j=1}^{n} \phi(y_j \mid \theta_k)^{\gamma_{jk}}
$$
然后取对数似然
$$
\log P(y_1, ..., y_n, \gamma \mid \theta) = 
\sum_{k=1}^{K} n_k \log \pi_k + \sum_{j=1}^{n} \sum_{k=1}^{K} \gamma_{jk} \log \phi(y_j \mid \theta_k)
$$




### E 步（Expectation）

在 EM 算法的 E 步中，我们的目标是：  
计算每个样本 $x_i$ 属于每个第 $k$ 个高斯分量的**后验概率**（“软标签”）：

$$
\gamma_{ik} = P(z_i = k \mid x_i, \theta)
$$

计算每个样本属于每个高斯分量的“后验概率” $\gamma_{ik}$：

$$
\gamma_{ik} = P(z_i = k \mid x_{i},\theta) = \frac{\pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \cdot \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
$$
[更本质的解释](#E步软标签本质解释)

如果带上EM算法的迭代标志：

$$
\gamma_{ik} = P(z_i = k \mid x_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \cdot \mathcal{N}(x_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \cdot \mathcal{N}(x_i \mid \mu_j^{(t)}, \Sigma_j^{(t)})}
$$

对于一个高维的高斯分布：
$d$ 为样本维度，$\mu_k$ 是第 $k$ 个高斯分布的均值，$\Sigma_k$ 是协方差矩阵。

$$
\mathcal{N}(x_i \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2}(x_i - \mu_k)^T \Sigma_k^{-1} (x_i - \mu_k) \right)
$$


将高斯密度函数带入上面的 E 步公式，得到：

$$
\gamma_{ik} = 
\frac{
\pi_k \cdot \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} 
\exp\left( -\frac{1}{2}(x_i - \mu_k)^T \Sigma_k^{-1} (x_i - \mu_k) \right)
}{
\sum_{j=1}^{K} 
\pi_j \cdot \frac{1}{(2\pi)^{d/2} |\Sigma_j|^{1/2}} 
\exp\left( -\frac{1}{2}(x_i - \mu_j)^T \Sigma_j^{-1} (x_i - \mu_j) \right)
}
$$

✅ 连乘形式（全数据集上的“似然贡献”）:

对于 $n$ 个样本：

$$
L(\theta) = \prod_{i=1}^{n} \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \Sigma_k)
$$

注意：这是**观测数据的边际似然函数**，E 步是为其分解“软归属”。

---

>[!note] 
>GMM 中的 E 步与 EM 通用公式的对应关系如下：

在 EM 算法的通用框架中，E 步的任务是：

 **计算隐变量 $Z$ 的后验分布：**  
 $$
q(Z) = p(Z \mid X, \theta^{(t)})
 $$
 对应GMM :
 
 $$
\gamma_{ik} = P(z_i = k \mid x_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \cdot \mathcal{N}(x_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \cdot \mathcal{N}(x_i \mid \mu_j^{(t)}, \Sigma_j^{(t)})}
$$

并用于构造期望下界：

$$
Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z \sim q(Z)} \left[ \log p(X, Z \mid \theta) \right]
$$

对应GMM ：

$$Q = \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik} \log \left[ \pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right]$$


✅ 在 GMM 中，对应的隐变量是什么？

- 隐变量：$Z = \{z_i\}_{i=1}^n$
- 每个 $z_i$ 表示样本 $x_i$ 属于哪个高斯分量（$z_i \in \{1, ..., K\}$）
- $z_i$ 是一个**离散的 latent class indicator**


| EM 通用形式                          | GMM 中的具体对应                                                                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 观测变量 $X$                         | 观测数据 $\{x_i\}_{i=1}^n$                                                                                                                               |
| 隐变量 $Z$                          | 样本属于哪个高斯分量：$z_i \in \{1, ..., K\}$                                                                                                                   |
| 当前参数 $\theta^{(t)}$              | $\theta^{(t)} = \{\pi_k^{(t)}, \mu_k^{(t)}, \Sigma_k^{(t)}\}$                                                                                        |
| 后验分布 $p(Z \mid X, \theta^{(t)})$ | $\gamma_{ik} = P(z_i = k \mid x_i, \theta^{(t)})$                                                                                                    |
| E 步输出 $q(Z)$                     | $\gamma_{ik}$，即“软分配”权重矩阵                                                                                                                             |
| $Q(\theta, \theta^{(t)})$        | 基于 $\gamma_{ik}$ 加权的完整数据似然期望：<br>$$Q = \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik} \log \left[ \pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right]$$ |


- 在 EM 的 E 步中，我们需要估计“缺失”的隐变量的分布；
- 在 GMM 中，这些隐变量是每个样本属于哪个高斯分量；
- E 步计算的 $\gamma_{ik}$，实际上就是每个样本的“软标签”（概率分配）；
- 这一步不更新参数，仅构造 M 步要使用的 $Q$ 函数。


- 每个数据点被软性划分到 $K$ 个高斯簇中；
- $\gamma_{ik}$ 表示 $x_i$ 属于第 $k$ 个簇的可能性；
- 所有 $\gamma_{ik}$ 构成一个 $n \times K$ 的责任矩阵。


- 目标：构造联合概率分布 $P(y_1, ..., y_n, \gamma \mid \theta)$

我们希望写出观测变量 $y_1, ..., y_n$ 与隐变量 $\gamma$（每个样本属于哪个分量）的联合概率。

其中：

- $n$：样本个数
- $K$：高斯分量个数
- $\gamma_{jk} \in \{0, 1\}$：样本 $y_j$ 是否属于第 $k$ 个分量
- $\theta_k = (\mu_k, \sigma_k)$：第 $k$ 个高斯分布的参数
- $\pi_k$：第 $k$ 个分量的混合系数



---

### M 步（Maximization）

更新模型参数：

- 混合系数：

  $$
  \pi_k = \frac{1}{n} \sum_{i=1}^n \gamma_{ik}
  $$

- 均值：

  $$
  \mu_k = \frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}}
  $$

- 协方差矩阵：

  $$
  \Sigma_k = \frac{\sum_{i=1}^n \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^n \gamma_{ik}}
  $$

重复 E/M 步直到对数似然收敛。


#### 优化过程

$$
Q(\theta, \theta^{(t)}) 
= \sum_{j=1}^{n} \sum_{k=1}^{K} \gamma_{jk}^{(t)} \left[
\log \pi_k + \log \mathcal{N}(x_j \mid \mu_k, \Sigma_k)
\right]
$$

我们将其拆成三部分分别优化：

---

- 优化混合系数 $\pi_k$

目标：

$$
\max_{\pi_k} \sum_{j=1}^n \sum_{k=1}^K \gamma_{jk}^{(t)} \log \pi_k
$$

约束：

$$
\sum_{k=1}^K \pi_k = 1,\quad \pi_k \geq 0
$$

定义拉格朗日函数：

$$
\mathcal{L}(\pi, \lambda) = \sum_{k=1}^K n_k \log \pi_k + \lambda \left( \sum_{k=1}^K \pi_k - 1 \right)
$$

其中：

$$
n_k = \sum_{j=1}^n \gamma_{jk}^{(t)}
$$

对 $\pi_k$ 求偏导并令为 0：

$$
\frac{\partial \mathcal{L}}{\partial \pi_k} = \frac{n_k}{\pi_k} + \lambda = 0
\quad\Rightarrow\quad \pi_k = -\frac{n_k}{\lambda}
$$

代入约束 $\sum_k \pi_k = 1$：

$$
\sum_k \pi_k = \sum_k \left( -\frac{n_k}{\lambda} \right) = -\frac{n}{\lambda} = 1
\quad\Rightarrow\quad \lambda = -n
$$

最终得到：

$$
\boxed{
\pi_k^{(t+1)} = \frac{n_k}{n}
}
$$

---

- 优化均值参数 $\mu_k$

我们最大化：

$$
\sum_{j=1}^n \gamma_{jk}^{(t)} \log \mathcal{N}(x_j \mid \mu_k, \Sigma_k)
$$
>[!note] 
>为什么在优化 $\mu_k$ 时可以省略 $\sum_k$ 项？

$$
Q(\theta, \theta^{(t)}) = \sum_{j=1}^n \sum_{k=1}^K \gamma_{jk}^{(t)} \left[
\log \pi_k + \log \mathcal{N}(x_j \mid \mu_k, \Sigma_k)
\right]
$$

当我们优化某个分量的参数（如 $\mu_k$）时：

- 所有 $k' \ne k$ 的项都**不依赖于** $\mu_k$；
- 在求导、优化时是常数项，可以忽略；
- 只保留和 $\mu_k$ 有关的部分：

$$
\sum_{j=1}^n \gamma_{jk}^{(t)} \log \mathcal{N}(x_j \mid \mu_k, \Sigma_k)
$$



只看与 $\mu_k$ 相关部分，忽略常数：

$$
\log \mathcal{N}(x_j \mid \mu_k, \Sigma_k) 
\propto -\frac{1}{2} (x_j - \mu_k)^T \Sigma_k^{-1} (x_j - \mu_k)
$$

因此我们需要最小化加权平方损失：

$$
J(\mu_k) = \sum_{j=1}^n \gamma_{jk}^{(t)} (x_j - \mu_k)^T \Sigma_k^{-1} (x_j - \mu_k)
$$

对 $\mu_k$ 求导：

$$
\nabla_{\mu_k} J = -2 \Sigma_k^{-1} \sum_{j=1}^n \gamma_{jk}^{(t)} (x_j - \mu_k)
$$

令导数为 0，得：

$$
\sum_{j=1}^n \gamma_{jk}^{(t)} x_j = \mu_k \sum_{j=1}^n \gamma_{jk}^{(t)}
\quad\Rightarrow\quad 
\boxed{
\mu_k^{(t+1)} = \frac{1}{n_k} \sum_{j=1}^n \gamma_{jk}^{(t)} x_j
}
$$

---

- 优化协方差矩阵 $\Sigma_k$

最大化：

$$
\sum_{j=1}^n \gamma_{jk}^{(t)} \log \mathcal{N}(x_j \mid \mu_k, \Sigma_k)
$$

等价于最小化：

$$
J(\Sigma_k) = \sum_{j=1}^n \gamma_{jk}^{(t)} \left[
\log |\Sigma_k| + (x_j - \mu_k)^T \Sigma_k^{-1} (x_j - \mu_k)
\right]
$$

对 $\Sigma_k$ 求导并令其为 0（略去高阶推导）可得最终更新公式：

$$
\boxed{
\Sigma_k^{(t+1)} = \frac{1}{n_k} \sum_{j=1}^n \gamma_{jk}^{(t)} (x_j - \mu_k)(x_j - \mu_k)^T
}
$$


## 五、GMM 与 K-Means 的对比

| 项目 | GMM | K-Means |
|------|-----|---------|
| 聚类方式 | 软聚类（概率分配） | 硬聚类（分簇） |
| 聚类形状 | 任意椭球形（通过 $\Sigma_k$） | 球形 |
| 模型类型 | 生成式 | 判别式 |
| 算法复杂度 | 高 | 低 |
| 可扩展性 | 强 | 一般 |

---

## 六、GMM 的优缺点

### ✅ 优点：

- 能拟合复杂数据分布（非球形）；
- 软聚类具有更丰富的信息；
- 可用于密度估计；
- 支持协方差矩阵建模维度间相关性。

### ❌ 缺点：

- 对初值敏感，可能陷入局部最优；
- 对异常值敏感；
- 模型选择（如 $K$ 的选取）较困难。

---

## 七、应用场景

- 图像分割（颜色聚类）
- 语音建模（MFCC 向量聚类）
- 数据挖掘与异常检测
- 半监督学习中的生成模型初始化




## Appendix

#### E步软标签本质解释
我们计算在当前参数 $\theta^{(t)}$ 下，每个样本属于第 $k$ 个分量的**后验概率**：

$$
\hat{\gamma}_{jk} = \mathbb{E}[\gamma_{jk} \mid y_j, \theta^{(t)}]
$$

由于 $\gamma_{jk} \in \{0, 1\}$ 是 one-hot 编码，且：

$$
\mathbb{E}[\gamma_{jk}] = 1 \cdot P(\gamma_{jk} = 1 \mid y_j, \theta^{(t)}) + 0 \cdot P(\gamma_{jk} = 0 \mid \cdot) = P(\gamma_{jk} = 1 \mid y_j, \theta^{(t)})
$$

所以：

$$
\hat{\gamma}_{jk} = P(\gamma_{jk} = 1 \mid y_j, \theta^{(t)})
$$



根据贝叶斯公式：

$$
P(\gamma_{jk} = 1 \mid y_j, \theta^{(t)}) = 
\frac{P(y_j \mid \gamma_{jk} = 1, \theta^{(t)}) \cdot P(\gamma_{jk} = 1 \mid \theta^{(t)})}
{\sum_{r=1}^{K} P(y_j \mid \gamma_{jr} = 1, \theta^{(t)}) \cdot P(\gamma_{jr} = 1 \mid \theta^{(t)})}
$$

其中：

- $P(y_j \mid \gamma_{jk} = 1, \theta^{(t)}) = \phi(y_j \mid \theta_k^{(t)})$  
- $P(\gamma_{jk} = 1 \mid \theta^{(t)}) = \pi_k^{(t)}$

最终表达为：

$$
\hat{\gamma}_{jk} = \frac{
\pi_k^{(t)} \cdot \phi(y_j \mid \theta_k^{(t)})
}{
\sum_{r=1}^{K} \pi_r^{(t)} \cdot \phi(y_j \mid \theta_r^{(t)})
}
$$
