---
~
---

## 1. 概述（Overview）

- **统计推断**：利用样本信息推断总体分布及其未知特征的理论与方法集合。
- **两大分支**：
  1. **频率派（Frequentist）** —— 参数视为固定未知量，推断基于样本频率。
  2. **贝叶斯派（Bayesian）** —— 参数视为随机变量，引入先验分布并更新为后验分布。
- **核心任务**：
  - **参数估计（点/区间）**
  - **假设检验**
  - **预测与决策**

---

## 2. 参数估计（Parameter Estimation）

### 2.1 点估计 vs 区间估计

| 类型   | 输出                   | 典型指标       | 优点     | 局限        |
| ---- | -------------------- | ---------- | ------ | --------- |
| 点估计  | 单个数值  $\hat{\theta}$ | 无偏性、效率、一致性 | 直观     | 不体现不确定性   |
| 区间估计 | 区间  $[L,U]$          | 置信水平       | 量化不确定性 | 结果区间而非具体值 |

---

### 2.2极大似然估计（MLE）
Maximum Likelihood Estimation

其核心思想是：**选择一组参数，使得在这些参数下，训练数据的“出现概率”最大。**

>[!note] 
>$\theta ^*$ : "我曾经见过很多$\theta$, 但他们都叫我super idol"

- **定义**： $\hat{\theta}_{\text{MLE}}=\arg\max_{\theta} \; L(\theta)=P(D\mid\theta)$。
- **性质**：一致性、渐近正态性、渐近效率；对先验无需求。
- **常见算法**：解析求导（$L'(\theta)=0$ ）、Newton-Raphson、EM（含隐变量）。
#### 基本思想

设我们有一个样本集：

$$
\mathcal{D} = \{x^{(1)}, x^{(2)}, \dots, x^{(n)}\}
$$
我们的目标是寻找参数$\theta ^*$ 满足：
$$
\theta^* = \arg\max_\theta p(\mathcal{D} \mid \theta)
$$

假设数据是从一个参数为 $\theta$ 的概率模型 $p(x \mid \theta)$ 中**独立同分布**采样的（这和Naive Beyes假设不同，这个是样本之间是iid的，而不是同一样本的不同性质之间是独立的），则联合概率为：

$$
L(\theta) = \prod_{i=1}^{n} p(x^{(i)} \mid \theta)
$$

这个函数 $L(\theta)$ 称为**似然函数（likelihood function）**。

MLE 的目标是选择使 $L(\theta)$ 最大的参数：

$$
\theta^* = \arg\max_\theta L(\theta)
$$
一般通过对$L(\theta)$ 求导，然后令导数等于0来求得极大值点

#### 对数似然函数（Log-Likelihood）

因为连乘运算不便于求导，我们通常取对数，得到对数似然函数：

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log p(x^{(i)} \mid \theta)
$$

最大化似然等价于最大化对数似然：

$$
\theta^* = \arg\max_\theta \ell(\theta)
$$

#### MLE 在机器学习中的应用

| 模型 | 假设 | 损失函数来源 |
|------|------|---------------|
| 逻辑回归 | $y \sim \text{Bernoulli}(\hat{y})$ | 对数似然推导出交叉熵 |
| 高斯判别分析 | 类条件高斯分布 | 对数似然推导出参数闭式解 |
| 朴素贝叶斯 | 条件独立的多项分布 | MLE 用于估计条件概率 |
| 生成模型（如 HMM） | 联合概率建模 | MLE 用于估计状态转移与发射概率 |


##### ✅ 优点：

- 理论严谨，广泛适用；
- 在样本数趋于无穷时，一致性好（收敛于真实值）；
- 在正则条件下是渐近无偏且有效的。

>[!note] ###### 什么是“在正则条件下是渐近无偏且有效的”？
>
> **正则条件**是一些技术性假设，用来保证 MLE 的理论性质成立。常见的正则条件包括：  
> 1. 参数空间是开集；  
> 2. 似然函数可微多次；  
> 3. Fisher 信息矩阵存在且非零；  
> 4. 真实分布包含在模型中；  
> 5. 某些极限交换条件成立（如微分与积分交换）。  
> 👉 **直观理解**：正则条件确保我们构造的模型“数学上足够良好”。
>
> 当样本量 $n \to \infty$ 时，最大似然估计 $\hat{\theta}_{\text{MLE}}$ 的期望趋近于真实参数 $\theta_0$：
>
> $$
> \lim_{n \to \infty} \mathbb{E}[\hat{\theta}_{\text{MLE}}] = \theta_0
> $$
>
> 也就是说，在足够多数据的情况下，MLE 会“平均上”估计出真实值，是**渐近无偏（Asymptotically Unbiased）** 的。
>
> **“渐近有效”（Asymptotically Efficient）** 指的是在所有渐近无偏估计量中，MLE 的方差是最小的。  
> 也就是说，当 $n$ 很大时，MLE 达到 **Cramér-Rao 下界**：
>
> $$
> \text{Var}(\hat{\theta}_{\text{MLE}}) \to \frac{1}{n \cdot \mathcal{I}(\theta_0)}
> $$
>
> 其中 $\mathcal{I}(\theta_0)$ 是 Fisher 信息量：
>
> $$
> \mathcal{I}(\theta_0) = -\mathbb{E}\left[ \frac{\partial^2 \log p(x \mid \theta)}{\partial \theta^2} \bigg|_{\theta = \theta_0} \right]
> $$
>
> **直观理解**：MLE 是“最精确的”渐近估计器。
>
> ✅ **结论**：在满足正则条件时，MLE 是一种**非常优秀的估计方法**：当样本足够大时，它既不会偏离真值（无偏），又在所有“合理估计方法”中有最小的方差（有效）。


##### ❌ 缺点：

- 对噪声敏感（无正则项）；
- 在样本较少时可能产生过拟合；
- 对数似然函数可能有多个局部极值。

---

#### 总结

> **MLE 本质上是从频率学派的角度寻找“最有可能产生数据的参数”**。它是构建很多机器学习模型（如逻辑回归、朴素贝叶斯、HMM 等）的基础算法。


---

### 2.3最大后验估计（MAP）

- **定义**： $\hat{\theta}_{\text{MAP}}=\arg\max_{\theta} P(\theta\mid D)=\arg\max_{\theta} P(D\mid\theta)P(\theta)$。
- **与 MLE 关系**：先验  $P(\theta)$ 为常数时退化为 MLE；可视为在对数似然上加入正则项。
- **示例**：高斯先验对应岭回归；拉普拉斯先验对应 Lasso。

---

### 2.4 贝叶斯估计（Bayesian Estimation）

1. **后验分布**： $P(\theta\mid D)=\dfrac{P(D\mid\theta)P(\theta)}{P(D)}$。
2. **决策视角**：给定损失函数  $L(\theta,a)$，最优决策  $a^*$ 使  $\mathbb E_{\theta\mid D}[L]$ 最小。
3. **常用点估计**：
   - **MMSE**：平方损失 → 取后验均值。
   - **MAP**：0-1 损失 → 取后验众数。
4. **求解方法**：解析共轭、MCMC、变分推断。

---

### 2.5 最小均方误差估计（MMSE）

- **目标**： $\hat{X}_{\text{MMSE}}=\mathbb E[X\mid Y=y]$。
- **性质**：平方损失下全局最优；需要完整后验分布。
- **例子**：加性高斯噪声模型下  $\hat{X}=\frac{\sigma_X^2}{\sigma_X^2+\sigma_N^2}y$。

---

### 2.6 线性最小均方误差估计（LLSE）

- **形式**： $\hat{X}_{\text{LLSE}}=aY+b$。
- **最优系数**： $a=\dfrac{\text{Cov}(X,Y)}{\text{Var}(Y)},\; b=\mu_X-a\mu_Y$。
- **特点**：仅需一二阶矩；高斯情形下与 MMSE 等价。

---

## 3. 假设检验（Hypothesis Testing）

### 3.1 基本框架

1. **原假设 H₀ / 备择假设 H₁**
2. **检验统计量 T(D)**
3. **拒绝域 / p 值**
4. **错误类型**：I 型 (α)，II 型 (β)，功效 (1-β)

### 3.2 常用检验

| 检验   | 统计量                                     | 适用场景             |
| ---- | --------------------------------------- | ---------------- |
| z 检验 | $\dfrac{\bar X-\mu_0}{\sigma/\sqrt n}$ | $\sigma$ 已知，大样本 |
| t 检验 | $\dfrac{\bar X-\mu_0}{S/\sqrt n}$      | $\sigma$ 未知，小样本 |
| 卡方检验 | $\sum\dfrac{(O-E)^2}{E}$               | 拟合优度、独立性         |
| F 检验 | $\dfrac{S_1^2}{S_2^2}$                 | 方差齐性             |

---

## 4. 置信区间（Confidence Intervals）

- **定义**：随机区间  $[L(D),U(D)]$ 满足  $P(\theta\in[L,U])\ge 1-\alpha$。
- **常见构造**：正态 / t / 卡方 / F 分布。
- **与检验对偶**：区间包含  $\theta_0$ ⇔ 无法拒绝  $H_0: \theta=\theta_0$。

---

## 5. 回归分析（Regression Analysis）

### 5.1 线性回归

- **模型**： $Y=X\beta+\varepsilon,\; \varepsilon\sim\mathcal N(0,\sigma^2I)$。
- **最小二乘 (OLS)**： $\hat\beta=(X^TX)^{-1}X^Ty$。
- **统计推断**： $t$ 检验系数显著性、 $F$ 检验整体显著性、 $R^2$ 拟合优度。

### 5.2 贝叶斯回归

- 先验  $\beta\sim\mathcal N(\beta_0,\Sigma_0)$，后验仍为高斯。
- 可以输出后验区间与预测分布。

---

## 6. 统计推断方法总结（Methodology Summary）

| 方法 | 归属 | 依赖先验 | 输出 | 优点 | 局限 |
|------|------|----------|------|------|------|
| MLE | 频率派 | 否 | 点估计 | 渐近最优，易实现 | 小样本不稳，忽略先验 |
| MAP | 贝叶斯 | 是 | 点估计 | 融合先验，抗过拟合 | 结果随先验敏感 |
| 贝叶斯估计 | 贝叶斯 | 是 | 后验/期望 | 全概率推断 | 计算成本高 |
| MMSE | 贝叶斯 | 是 | 点估计 | MSE 最优 | 需完整后验 |
| LLSE | 频率/贝叶斯 | 可选 | 线性点估计 | 仅需一二阶矩 | 线性限制 |
| 置信区间 | 频率派 | 否 | 区间 | 量化不确定性 | 解释依赖频率观 |
| 假设检验 | 频率派 | 否 | 决策 | 控制错误率 | p 值易被误用 |

---

## 7. 常见误区与实践建议

1. **p 值 ≠ 置信度**：p 值小并不表示备择假设为真。
2. **协方差为 0 不代表独立**：仅对线性相关性为 0。
3. **先验选择影响 MAP/MMSE**：应做敏感性分析。
4. **多重检验问题**：控制 FDR/Bonferroni。
5. **模型假设检验**：残差分析是回归有效性的关键。

---

