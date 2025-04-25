

- [[贝叶斯网络_BayesianNetworks]]
- 参数贝叶斯模型
	- 贝叶斯线性回归_BayesianLinearRegression
	- 贝叶斯逻辑回归_BayesianLogisticRegression
	- 层次贝叶斯模型_HierarchicalBayesianModel
- 非参数贝叶斯模型
	- [[高斯过程_GaussianProcesses]]



> [!note] 定义
> **贝叶斯模型（Bayesian Models）** 是一类利用贝叶斯公式，将**先验知识**与**观测数据**结合，进行**不确定性建模与推断**的模型族。

贝叶斯公式核心：
$$
P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}
$$
- $\theta$：模型参数（或隐变量）
- $D$：观测数据
- $P(\theta)$：**先验**分布
- $P(D \mid \theta)$：**似然函数**
- $P(\theta \mid D)$：**后验**分布（推断目标）


>[!note] 参数贝叶斯模型（Parametric）
>模型参数个数是**固定的**（不随数据量变化），只是我们对这些参数建了概率分布。
例如：$\theta \sim \mathcal{N}(0, I)$

#### 1. [[贝叶斯线性回归_BayesianLinearRegression]]

- 在经典线性回归的基础上：
	- 为权重 $w$ 引入高斯先验
	- 得到后验分布 $p(w|X, y)$
- 预测时不输出一个点，而是输出一个高斯分布
- 可以自然给出**不确定性估计**

#### 2. [[贝叶斯逻辑回归_BayesianLogisticRegression]]

- 在逻辑回归的基础上为 $w$ 加先验
- 后验没有解析解，需要近似推断（如 MCMC 或 Laplace）
- 适用于小数据集或不确定性高的分类任务

#### 3. [[层次贝叶斯模型_HierarchicalBayesianModel]]

- 在参数的上层再放一个先验，也就是“先验的先验”
- 常用于建模“组之间的差异”，比如：
  > 学生→班级→学校，每一层有参数，每层都用贝叶斯建模
- 表达力更强、共享信息能力更好


---


>[!note] 参数贝叶斯模型（Parametric）
>模型的参数维度**不固定**，可以随数据增长而扩展，理论上可以使用**无限维空间**来建模。


#### 1. [[高斯过程_GaussianProcesses]]

- 建模函数空间中的分布：$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$
- 每个点的预测是高斯分布，整体是一个函数的分布
- 用核函数 $k$ 控制函数的“平滑性”“变化范围”

