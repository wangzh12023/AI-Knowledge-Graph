

- [[参数模型_ParametricModels]]
	- 线性回归_LinearRegression
	- 逻辑回归_LogisticRegression
	- 朴素贝叶斯_NaiveBayes
	- 感知机_Perceptron
	- 支持向量机_SupportVectorMachines
	- 神经网络_NeuralNetworks
	- 最大熵模型_MaximumEntropyModel
	- 高斯判别分析_GaussianDiscriminantAnalysis
	- 隐马尔可夫模型_HMM
	- 贝叶斯线性回归_BayesianLinearRegression
	- 贝叶斯逻辑回归_BayesianLogisticRegression

- [[非参数模型_NonParametricModels]]
	- K近邻算法_KNearestNeighbors
	- 决策树_DecisionTrees
	- 随机森林_RandomForests
	- 核密度估计_KernelDensityEstimation
	- 高斯过程_GaussianProcesses
	- 核回归_KernelRegression




## 定义

- **参数模型（Parametric Models）**：  
  假设模型具有固定结构，使用**有限个参数**来表示数据的分布。参数数量与数据量无关。

- **非参数模型（Non-Parametric Models）**：  
  不对数据分布做固定结构假设，模型的复杂度可随着数据量增长而增加，参数数量**非固定**。

| 特征        | 参数模型 Parametric | 非参数模型 Non-Parametric |
| --------- | --------------- | -------------------- |
| 参数数量      | 固定              | 随数据量变化               |
| 分布假设      | 明确假设（如高斯分布）     | 少或无分布假设              |
| 灵活性       | 较低              | 高                    |
| 对数据量要求    | 少量数据也可训练        | 通常需要更多数据             |
| 学习速度/计算开销 | 快               | 慢                    |
| 容易过拟合？    | 不容易             | 容易（需正则化）             |

