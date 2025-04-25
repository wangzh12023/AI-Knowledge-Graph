当然可以！下面是为你整理的 `[[常见分布模型_CommonDistributions]]` 的完整 Markdown 内容笔记，包含尽量多的**离散型和连续型常见概率分布模型**，每一个都包括：

- 分布定义
- PDF/PMF 或 CDF 公式
- 参数解释
- 期望与方差
- 应用场景

你可以直接粘贴到 Obsidian 等支持 Markdown 的笔记系统中使用。

---

## `常见分布模型（Common Distributions）`

> 概率论中常见的离散与连续型分布模型及其性质汇总

---

## 🔷 一、离散型分布（Discrete Distributions）

---

### 📌 1. 伯努利分布（Bernoulli Distribution）

- 定义：仅有两个可能结果（通常为0和1）
- 参数：$p \in [0,1]$
- PMF：
  $$
  P(X = x) = p^x (1 - p)^{1 - x}, \quad x \in \{0, 1\}
  $$
- 期望：$\mathbb{E}[X] = p$
- 方差：$\mathrm{Var}(X) = p(1 - p)$
- 应用：投硬币、一次试验成功与失败

---

### 📌 2. 二项分布（Binomial Distribution）

- 定义：进行$n$ 次独立伯努利试验，统计成功次数
- 参数：$n \in \mathbb{N},\ p \in [0,1]$
- PMF：
  $$
  P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k},\quad k = 0,1,...,n
  $$
- 期望：$\mathbb{E}[X] = np$
- 方差：$\mathrm{Var}(X) = np(1 - p)$
- 应用：掷硬币、产品合格数

---

### 📌 3. 几何分布（Geometric Distribution）

- 定义：独立重复伯努利试验，直到第一次成功所需的试验次数
- 参数：$p \in (0,1)$
- PMF：
  $$
  P(X = k) = (1 - p)^{k - 1}p,\quad k = 1, 2, ...
  $$
- 期望：$\mathbb{E}[X] = \frac{1}{p}$
- 方差：$\mathrm{Var}(X) = \frac{1 - p}{p^2}$
- 应用：等待第一次成功的模型，如网络重传

#### 3.1 FirstSuccess(FS)
- 定义：独立重复伯努利试验，直到第一次成功所失败的次数

---

### 📌 4. 泊松分布（Poisson Distribution）

- 定义：单位时间/空间内事件发生的次数，满足稀疏、独立条件
- 参数：$\lambda > 0$
- PMF：
  $$
  P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!},\quad k = 0, 1, 2, ...
  $$
- 期望：$\mathbb{E}[X] = \lambda$
- 方差：$\mathrm{Var}(X) = \lambda$
- 应用：排队论、电话呼叫数、自然灾害次数

---

## 🔶 二、连续型分布（Continuous Distributions）

---

### 📌 1. 均匀分布（Uniform Distribution）

- 定义：在区间$[a, b]$ 上所有值等可能出现
- 参数：$a < b$
- PDF：
  $$
  f(x) = \begin{cases}
    \frac{1}{b - a}, & a \le x \le b \\
    0, & \text{otherwise}
  \end{cases}
  $$
- 期望：$\mathbb{E}[X] = \frac{a + b}{2}$
- 方差：$\mathrm{Var}(X) = \frac{(b - a)^2}{12}$
- 应用：模拟器采样、随机数生成

---

### 📌 2. 指数分布（Exponential Distribution）


#### 🧠 定义：
指数分布描述的是：

> “**某事件首次发生所需的等待时间**”，比如：
> - 第一次顾客到达商店的时间
> - 第一次元件失效的时间

---

#### ⚙️ 参数：
- $\lambda > 0$：**事件发生速率**（rate）
- 单个参数即可描述分布

#### 📈 PDF（概率密度函数）：

$$
f(x) = \lambda e^{-\lambda x},\quad x \ge 0
$$

#### 📉 CDF（累积分布函数）：

$$
F(x) = P(X \le x) = 1 - e^{-\lambda x}
$$

#### 📊 性质：

| 性质 | 表达式 |
|------|--------|
| 支撑集 | $x \in [0, \infty)$ |
| 期望 | $\mathbb{E}[X] = \frac{1}{\lambda}$ |
| 方差 | $\mathrm{Var}(X) = \frac{1}{\lambda^2}$ |
| 无记忆性 | $P(X > s + t \mid X > s) = P(X > t)$ |


#### 📌 应用：
- 模拟随机事件的等待时间
- 马尔可夫过程的状态转移时间间隔
- 元器件寿命建模


#### 🔄 与伽马分布的关系：
> **伽马分布是多个独立指数分布的总和。**  
若 $X_i \sim \text{Exp}(\lambda)$，则：
$$
\sum_{i=1}^{k} X_i \sim \text{Gamma}(k, \lambda)
$$


---

### 📌 3. 正态分布（Normal Distribution）

- 定义：自然界中最常见的分布，又称高斯分布
- 参数：$\mu \in \mathbb{R},\ \sigma > 0$
- PDF：
  $$
  f(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
  $$
- 期望：$\mathbb{E}[X] = \mu$
- 方差：$\mathrm{Var}(X) = \sigma^2$
- 应用：噪声模型、人类身高误差、金融市场等

---

### 📌 4. 对数正态分布（Log-Normal Distribution）

- 定义：若 $\ln X \sim \mathcal{N}(\mu, \sigma^2)$，则$X$ 服从对数正态分布
- PDF：
  $$
  f(x) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left( -\frac{(\ln x - \mu)^2}{2\sigma^2} \right), \quad x > 0
  $$
- 应用：金融资产价格、乘法噪声模型

---

### 📌 5. 卡方分布（Chi-Square Distribution）

#### 📚 定义（Definition）

卡方分布是指：  
> 若 $Z_1, Z_2, ..., Z_k$ 为 **相互独立的标准正态分布变量**，即 $Z_i \sim \mathcal{N}(0, 1)$，那么它们平方和的随机变量：
$$
X = \sum_{i=1}^{k} Z_i^2
$$
服从自由度为 $k$ 的 **卡方分布**，记作：
$$
X \sim \chi^2_k
$$

---

#### ⚙️ 参数（Parameter）

- 自由度 $k$：必须为正整数，代表参与平方求和的独立标准正态变量个数。

#### 📈 PDF（概率密度函数）

卡方分布的 PDF 为：

$$
f(x) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2 - 1} e^{-x/2}, \quad x \ge 0
$$

其中：
- $\Gamma(\cdot)$ 是伽马函数，满足 $\Gamma(n) = (n - 1)!$ 对于整数 $n$。

#### 📉 CDF（累积分布函数）

- 没有初等函数的封闭表达式；
- 通常使用数值积分或查表/软件（如 Python `scipy.stats.chi2.cdf`）计算；
- 可表示为正则化伽马函数：
  $$
  F(x; k) = \frac{\gamma(k/2, x/2)}{\Gamma(k/2)}
  $$

#### 🧮 统计特性（期望、方差、矩）

| 特性 | 表达式 |
|------|--------|
| 期望（Mean） | $\mathbb{E}[X] = k$ |
| 方差（Variance） | $\mathrm{Var}(X) = 2k$ |
| 偏度（Skewness） | $\sqrt{8/k}$ |
| 峰度（Kurtosis） | $12/k$ |

#### 🔁 与其他分布的关系

- 若 $X \sim \chi^2_k$，则 $X$ 是伽马分布 $\Gamma(k/2, 2)$ 的特例。
- 如果 $X_i \sim \chi^2_{k_i}$ 且彼此独立，则：
  $$
  \sum X_i \sim \chi^2_{\sum k_i}
  $$
- 构成 t 分布：
  $$
  T = \frac{Z}{\sqrt{X/k}} \sim t_k,\quad Z \sim \mathcal{N}(0,1),\ X \sim \chi^2_k
  $$
- 构成 F 分布：
  $$
  F = \frac{(X_1/k_1)}{(X_2/k_2)} \sim F_{k_1, k_2},\quad X_i \sim \chi^2_{k_i}
  $$

#### 🧪 应用场景（Applications）

- **假设检验**：
  - 方差是否等于某值（单总体方差检验）
  - 两总体方差比较（通过 F 分布）
- **拟合优度检验（卡方检验）**：
  - 比较观测频数与理论频数之间是否有显著差异（例如人口调查、掷骰子实验）
- **置信区间**：
  - 利用 $\chi^2$ 分布构造方差的置信区间：
    $$
    P\left( \frac{(n-1)S^2}{\chi^2_{1-\alpha/2}} \le \sigma^2 \le \frac{(n-1)S^2}{\chi^2_{\alpha/2}} \right) = 1 - \alpha
    $$

#### 🔍 分布形状随自由度变化

| 自由度 $k$ | 分布形状描述 |
|--------------|---------------|
| $k = 1$ | 极度偏斜，单峰在 $x \to 0^+$ |
| $k = 2$ | 指数分布形式（记得它是指数分布特例） |
| $k > 2$ | 曲线右移、逐渐趋近对称，类似正态分布 |
| $k \to \infty$ | 收敛于 $\mathcal{N}(k, 2k)$（中心极限定理） |

#### 🐍 Python 示例（可视化）

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

x = np.linspace(0, 20, 500)
for k in [1, 2, 5, 10]:
    plt.plot(x, chi2.pdf(x, df=k), label=f"k={k}")

plt.title("Chi-Square Distribution PDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
```


---


### 📌 6. t 分布（Student's t Distribution）

- **定义**：当总体服从正态分布，但**样本方差未知**，使用样本均值和样本标准差构造的统计量服从 t 分布，广泛用于小样本推断。
  
  若 $X \sim \mathcal{N}(0,1)$，$Z \sim \chi^2_k$ 且独立，则：
  $$
  T = \frac{X}{\sqrt{Z/k}} \sim t_k
  $$

- **参数**：自由度 $k > 0$

- **PDF**（概率密度函数）：
  $$
  f(x) = \frac{\Gamma\left(\frac{k+1}{2}\right)}{\sqrt{k\pi}\ \Gamma\left(\frac{k}{2}\right)} \left(1 + \frac{x^2}{k}\right)^{-\frac{k+1}{2}},\quad x \in \mathbb{R}
  $$

  - 其中 $\Gamma(\cdot)$ 是伽马函数，满足 $\Gamma(n) = (n-1)!$（当 $n$ 为正整数时）

- **期望**：
  $$
  \mathbb{E}[T] = 0 \quad (\text{当 } k > 1)
  $$

- **方差**：
  $$
  \mathrm{Var}(T) = \frac{k}{k - 2}, \quad (\text{当 } k > 2)
  $$

- **应用**：
  - 小样本均值检验（t 检验）
  - 回归参数显著性检验
  - 构造置信区间时使用


---

### 📌 8. 贝塔分布（Beta Distribution）

#### 🧠 定义：
贝塔分布用于建模：
> **概率值（0 到 1 之间）上的随机变量分布**，非常适合表示“成功概率的不确定性”。

是最常见的**共轭先验分布**（用于贝叶斯推断）。

#### ⚙️ 参数：
- $\alpha > 0$：成功次数的“拟合”
- $\beta > 0$：失败次数的“拟合”

#### 📈 PDF：

$$
f(x) = \frac{x^{\alpha - 1}(1 - x)^{\beta - 1}}{B(\alpha, \beta)},\quad x \in (0,1)
$$

其中 $B(\alpha, \beta)$ 是贝塔函数，与伽马函数的关系为：

$$
B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}
$$

#### 📊 性质：

| 性质 | 表达式 |
|------|--------|
| 支撑集 | $x \in (0, 1)$ |
| 期望 | $\mathbb{E}[X] = \frac{\alpha}{\alpha + \beta}$ |
| 方差 | $\mathrm{Var}(X) = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}$ |

#### 📌 应用：

- 贝叶斯推断中建模概率 $p$：$p \sim \text{Beta}(\alpha, \beta)$
- Beta-Bernoulli 模型：后验仍为 Beta
- 表示命中率、转化率等“概率不确定性”

#### 🎨 不同形状示意：

| 参数组合 | 分布形状 |
|----------|-----------|
| $\alpha = \beta = 1$ | 均匀分布 |
| $\alpha > 1,\ \beta > 1$ | 单峰 |
| $\alpha < 1,\ \beta < 1$ | U 形 |
| $\alpha > \beta$ | 向右偏 |
| $\alpha < \beta$ | 向左偏 |


---

### 📌 9. 伽马分布（Gamma Distribution）

#### 🧠 定义：
伽马分布是描述：
> **事件“累计发生 k 次”所需时间的分布**

是指数分布的推广，允许“多次等待”。

---

#### ⚙️ 参数：
- $k > 0$：**形状参数（shape）**
- $\lambda > 0$：**速率参数（rate）** 或 $\theta = 1/\lambda$：**尺度参数（scale）**

---

#### 📈 PDF：

$$
f(x) = \frac{\lambda^k}{\Gamma(k)} x^{k - 1} e^{-\lambda x},\quad x \ge 0
$$

或换成尺度参数：

$$
f(x) = \frac{1}{\Gamma(k)\theta^k} x^{k - 1} e^{-x/\theta}
$$

对于任意 $x > 0$，伽马函数定义为不定积分：

$$
\Gamma(x) = \int_0^{\infty} t^{x - 1} e^{-t} \, dt
$$
并且存在递推公式：
$$
\Gamma(x + 1) = x \cdot \Gamma(x)
$$

这就是很多分布（如 Gamma、Beta、t、卡方分布）中的 $\Gamma(k)$。

| 情况            | $\Gamma(k)$ 的处理方式                                         |
| ------------- | --------------------------------------------------------- |
| $k$ 为正整数      | $\Gamma(k) = (k - 1)!$                                    |
| $k = n + 0.5$ | 用递推公式和 $\Gamma(1/2) = \sqrt{\pi}$ 推导                      |
| $k$ 任意正实数     | 用积分或数值近似（如 `scipy.special.gamma`）计算                       |
| $k$ 很大        | 用斯特林近似公式 $\Gamma(k) \approx \sqrt{2\pi}k^{k - 1/2}e^{-k}$ |

#### 📊 性质：

| 性质 | 表达式 |
|------|--------|
| 支撑集 | $x \in [0, \infty)$ |
| 期望 | $\mathbb{E}[X] = \frac{k}{\lambda} = k\theta$ |
| 方差 | $\mathrm{Var}(X) = \frac{k}{\lambda^2} = k\theta^2$ |

#### 📌 特殊情况：

- 当 $k = 1$：$\text{Gamma}(1, \lambda) = \text{Exponential}(\lambda)$
- 当 $k = n/2,\ \theta = 2$：伽马分布就是自由度为 $n$ 的卡方分布：  
  $$
  \chi^2_n \sim \text{Gamma}(n/2, 2)
  $$

#### 📌 应用：

- 系统寿命（多个元件需全部失效）
- 等待多个事件发生的时间
- 贝叶斯先验建模（如 Gamma-Poisson 模型）

