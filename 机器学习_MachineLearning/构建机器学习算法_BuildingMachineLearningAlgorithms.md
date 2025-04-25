

- 数据预处理_DataPreprocessing
	
	- 标准化_Normalization
		
	- 归一化_Standardization
		
	- 特征工程_FeatureEngineering
		
- 模型评估_ModelEvaluation
	
	- 交叉验证_CrossValidation
		
	- 混淆矩阵_ConfusionMatrix
		
	- 准确率、精确率、召回率_AccuracyPrecisionRecall
		
	- F1分数_F1Score
		
	- ROC曲线与AUC_ROCCurveAndAUC
		
- 过拟合与欠拟合_OverfittingAndUnderfitting
	
	- 正则化_Regularization
		
	- 早停法_EarlyStopping
		
- 模型调优_ModelTuning
	
	- 超参数调优_HyperparameterTuning
		
	- 网格搜索_GridSearch
		
	- 随机搜索_RandomSearch


# 机器学习训练流程基础知识

本笔记围绕机器学习模型训练过程中的重要环节进行整理，包括数据预处理、模型评估、模型调优以及如何应对过拟合与欠拟合等问题。

---

## 数据预处理 DataPreprocessing

数据预处理是机器学习的关键步骤之一，目标是提升数据质量、消除偏差，为模型训练做好准备。

### 标准化 Normalization

将特征转换为**均值为 0、标准差为 1** 的分布，常用于要求特征分布对称或算法依赖欧几里得距离（如SVM、线性回归）时。

公式：

$$
x' = \frac{x - \mu}{\sigma}
$$

### 归一化 Standardization

将特征缩放到固定区间（如 $[0, 1]$），适用于特征取值范围差异较大的情况。

公式：

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

### 特征工程 FeatureEngineering

包括特征构造、选择、变换等操作，旨在提升模型对数据的表达能力。例如：

- One-hot 编码
- 多项式扩展
- PCA 降维
- 文本向量化

| 步骤 | 内容说明 |
|------|----------|
| 特征提取 | 从原始数据中生成初始特征，例如从文本、图像、时间戳中提取信息 |
| 特征转换 | 对已有特征进行变换，如缩放、离散化、编码等 |
| 特征选择 | 筛选出对任务最有用的特征，去除冗余和无关特征 |
| 特征构造 | 根据已有特征构造新特征，如交叉特征、多项式特征、统计特征 |

---

## 模型评估 ModelEvaluation

评估模型在训练集和验证集上的表现，用于判断模型好坏与泛化能力。

### 交叉验证 CrossValidation

将数据划分为若干份，轮流作为验证集，综合多个验证结果进行评估，常见方法有 K 折交叉验证（K-Fold）。

### 混淆矩阵 ConfusionMatrix

用于分类任务，显示模型在不同类别上的预测情况。

|          | 预测正类 | 预测负类 |
|----------|----------|----------|
| 实际正类 | TP       | FN       |
| 实际负类 | FP       | TN       |

### 准确率、精确率、召回率 Accuracy, Precision, Recall

- TP：True Positive（真正）
    
- TN：True Negative（真负）
    
- FP：False Positive（假正，即预测为正但是实际是负）
    
- FN：False Negative（假负，即预测为负但是实际是正）

- **准确率（accuracy）**：预测正确的样本占比  
  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  $$

- **精确率（precision）**：预测为正的样本中实际为正的比例  
  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$

- **召回率（recall）**：所有正样本中被正确预测的比例  
  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$

### F1分数_F1Score

精确率与召回率的调和平均，适用于不平衡数据集：

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

 **Micro-F1**

$$
\text{Micro Precision} = \frac{\sum_i TP_i}{\sum_i (TP_i + FP_i)}
$$

$$
\text{Micro Recall} = \frac{\sum_i TP_i}{\sum_i (TP_i + FN_i)}
$$

$$
\text{Micro F1} = \frac{2 \cdot \text{Micro Precision} \cdot \text{Micro Recall}}{\text{Micro Precision} + \text{Micro Recall}}
$$

✅ **特点**：更关注样本数量多的类别，适合不平衡类别情况。



**Macro-F1**

$$
\text{Macro F1} = \frac{1}{K} \sum_{i=1}^{K} F1_i
$$

其中 $K$ 是类别数，$F1_i$ 是第 $i$ 类的 F1 分数。

✅ **特点**：对每个类别赋予**相同权重**，更适合**关注每个类别表现均衡性**的场景。


### ROC 曲线与 AUC ROCCurveAndAUC

- ROC 曲线：绘制 TPR 与 FPR 的关系曲线
- AUC（曲线下面积）衡量模型整体区分能力，越接近 1 越好。

---

## 过拟合与欠拟合 OverfittingAndUnderfitting

- **过拟合**：模型在训练集表现很好，但在测试集表现差；
- **欠拟合**：模型在训练集都无法很好学习数据结构。
![[Pasted image 20250423170601.png|350]]
### 正则化 Regularization
[正则化](正则化_Regularization)
在损失函数中添加惩罚项，限制模型参数大小。

- L1 正则化：产生稀疏解，适合特征选择；
- L2 正则化：惩罚大权重，提升泛化能力。

### 早停法 EarlyStopping

在验证集误差上升时提前停止训练，防止过拟合，常用于深度学习中。

### Laplace Smoothing 

![[Pasted image 20250423170650.png|400]]

### Linear Interpolation

![[Pasted image 20250423170828.png|350]]

## 模型调优 ModelTuning

模型调优是选择最佳模型结构和参数组合以提升泛化性能的过程。

### 超参数调优 HyperparameterTuning

调整如学习率、正则化系数、树的深度、核函数等模型结构参数。

### 网格搜索 GridSearch

暴力搜索所有可能的超参数组合，适合参数较少的情况。

### 随机搜索 RandomSearch

随机采样部分超参数组合，比网格搜索高效，适合高维超参数空间。


## 总结

这部分知识构成了机器学习“非模型结构本身”之外的核心操作，理解并合理使用这些工具，是实现高质量模型训练的关键。

建议在实际工程中：

- **数据预处理**：标准化/归一化应配合算法；
- **模型评估**：交叉验证比单次划分更稳健；
- **防止过拟合**：正则化 + 早停法是常用组合；
- **调参技巧**：先用随机搜索粗调，再用网格搜索精调。

