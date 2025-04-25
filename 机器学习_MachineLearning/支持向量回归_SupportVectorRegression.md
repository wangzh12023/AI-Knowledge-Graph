
## 支持向量回归（Support Vector Regression, SVR）

支持向量回归（Support Vector Regression, SVR）是支持向量机（SVM）的回归版本。它试图学习一个函数，使得预测值和真实值之间的偏差尽量小，并且模型本身尽可能“平滑”。

---

## 一、基本思想

SVR 的目标是：

> 找到一个函数 $f(x) = \langle \boldsymbol{w}, x \rangle + b$，使得对所有训练数据点，预测值与真实值的偏差在 $\varepsilon$ 范围之内，并尽量使模型的复杂度（即 $\|\boldsymbol{w}\|^2$）最小。

---

## 二、线性 SVR 的原始优化问题

给定训练集：

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^m
$$

目标是找到线性函数：

$$
f(x) = \boldsymbol{w}^T x + b
$$

使其对所有样本满足误差在 $\varepsilon$ 范围内，同时使 $\|\boldsymbol{w}\|^2$ 最小。

### 如果允许一定误差（软间隔），引入松弛变量 $\xi_i, \xi_i^*$，则原始问题变为：

$$
\begin{aligned}
\min_{\boldsymbol{w}, b, \xi_i, \xi_i^*} \quad & \frac{1}{2} \|\boldsymbol{w}\|^2 + C \sum_{i=1}^m (\xi_i + \xi_i^*) \\
\text{s.t.} \quad & y^{(i)} - \boldsymbol{w}^T x^{(i)} - b \leq \varepsilon + \xi_i \\
& \boldsymbol{w}^T x^{(i)} + b - y^{(i)} \leq \varepsilon + \xi_i^* \\
& \xi_i, \xi_i^* \geq 0,\quad i = 1, \dots, m
\end{aligned}
$$

对于样本 $(x^{(i)}, y^{(i)})$：

- 如果预测值比真实值**小**很多（下穿 $\varepsilon$ 带）：
  - 用 $\xi_i$ 表示超出部分。
- 如果预测值比真实值**大**很多（上穿 $\varepsilon$ 带）：
  - 用 $\xi_i^*$ 表示超出部分。


| 情况                                     | 解释                 | $\xi_i$ | $\xi_i^*$ |
| -------------------------------------- | ------------------ | ------- | --------- |
| $f(x^{(i)}) - y^{(i)}\leq \varepsilon$ | 在“容忍带”内，无误差        | 0       | 0         |
| $f(x^{(i)}) < y^{(i)} - \varepsilon$   | 下穿 $\varepsilon$ 带 | >0      | 0         |
| $f(x^{(i)}) > y^{(i)} + \varepsilon$   | 上穿 $\varepsilon$ 带 | 0       | >0        |

- $\frac{1}{2} \|\boldsymbol{w}\|^2$：控制函数的平滑性（即小的斜率）
- $C \sum (\xi_i + \xi_i^*)$：控制违反 $\varepsilon$ 精度范围的惩罚

---

## 四、对偶问题（Dual Problem）推导

通过拉格朗日函数引入对偶变量 $\alpha_i, \alpha_i^*$：

构建拉格朗日函数：

$$
\begin{aligned}
L &= \frac{1}{2} \|\boldsymbol{w}\|^2 + C \sum_{i=1}^m (\xi_i + \xi_i^*) \\
&- \sum_{i=1}^m \alpha_i (\varepsilon + \xi_i - y^{(i)} + \boldsymbol{w}^T x^{(i)} + b) \\
&- \sum_{i=1}^m \alpha_i^* (\varepsilon + \xi_i^* + y^{(i)} - \boldsymbol{w}^T x^{(i)} - b) \\
&- \sum_{i=1}^m \mu_i \xi_i - \sum_{i=1}^m \mu_i^* \xi_i^*
\end{aligned}
$$

对 $\boldsymbol{w}, b, \xi_i, \xi_i^*$ 求偏导并令其为 0，得到 KKT 条件：

- $\boldsymbol{w} = \sum_{i=1}^m (\alpha_i - \alpha_i^*) x^{(i)}$
- $\sum_{i=1}^m (\alpha_i - \alpha_i^*) = 0$
- $\alpha_i, \alpha_i^* \in [0, C]$

代入原始函数，得到对偶问题：

$$
\begin{aligned}
\max_{\alpha_i, \alpha_i^*} \quad & -\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*) \langle x^{(i)}, x^{(j)} \rangle \\
& - \varepsilon \sum_{i=1}^m (\alpha_i + \alpha_i^*) + \sum_{i=1}^m y^{(i)} (\alpha_i - \alpha_i^*) \\
\text{s.t.} \quad & \sum_{i=1}^m (\alpha_i - \alpha_i^*) = 0 \\
& 0 \leq \alpha_i, \alpha_i^* \leq C
\end{aligned}
$$

---

## 五、决策函数

解出 $\alpha_i, \alpha_i^*$ 后，原始函数为：

$$
f(x) = \sum_{i=1}^m (\alpha_i - \alpha_i^*) \langle x^{(i)}, x \rangle + b
$$

---

## 六、核方法扩展（非线性 SVR）

通过核函数 $K(x^{(i)}, x)$ 将 SVR 拓展到高维特征空间，得到核回归函数：

$$
f(x) = \sum_{i=1}^m (\alpha_i - \alpha_i^*) K(x^{(i)}, x) + b
$$

常见核函数：

- 线性核：$K(x, x') = x^T x'$
- 多项式核：$K(x, x') = (x^T x' + r)^d$
- 高斯核（RBF）：$K(x, x') = \exp\left( -\frac{\|x - x'\|^2}{2\sigma^2} \right)$

---

## 七、SVR 的 $\varepsilon$-敏感损失函数

SVR 采用 $\varepsilon$-insensitive 损失：

$$
L_\varepsilon(f(x), y) =
\begin{cases}
0, & \text{if } |f(x) - y| \leq \varepsilon \\
|f(x) - y| - \varepsilon, & \text{otherwise}
\end{cases}
$$

这种损失允许“误差带”内的预测不被惩罚，提高模型鲁棒性。


