
## 什么是序列标注？

> 给定一个输入序列 $\boldsymbol{x} = [x_1, x_2, ..., x_n]$，预测对应的标签序列 $\boldsymbol{y} = [y_1, y_2, ..., y_n]$。

每个输入元素 $x_i$ 都对应一个输出标签 $y_i$。


## Some sequence labeling tasks

### 2.1 PartOfSpeech Tagging
Assign syntactic labels (e.g., Noun, Verb, etc.) to each word in a sentence.

🧠 Motivation:
- Improves parsing, MachineTranslation, sentiment analysis, and TTS(text-to-speech).

| 词性     | 英文缩写    | 词性英文                                  | 示例                        |
| ------ | ------- | ------------------------------------- | ------------------------- |
| 名词     | NN      | Noun                                  | dog, idea                 |
| 动词     | VB      | Verb                                  | run, is                   |
| 形容词    | JJ      | Adjective                             | red, happy                |
| 副词     | RB      | Adverb                                | quickly, very             |
| 介词     | IN      | Preposition                           | in, on, under             |
| 冠词/限定词 | DT      | Determiner                            | the, a, some              |
| 代词     | PRP     | Pronoun                               | he, she, it               |
| 连词     | CC      | Coordinating Conjunction              | and, but                  |
|        | **VB**  | Verb, base form                       | run, eat, go              |
|        | **VBD** | Verb, past tense                      | ran, ate                  |
|        | **VBG** | Verb, gerund or present participle    | running, eating           |
|        | **VBN** | Verb, past participle                 | run, eaten                |
|        | **VBP** | Verb, non-3rd person singular present | run, eat (用于 I/you/they)  |
|        | **VBZ** | Verb, 3rd person singular present     | runs, eats (用于 he/she/it) |

### 2.2 Chinese Word Segmentation
Label characters with B/I/E/S tags indicating word boundaries.

| 标签  | 含义            |
| --- | ------------- |
| B   | 单词的开始（Begin）  |
| I   | 单词的中间（Inside） |
| E   | 单词的结束（End）    |
| S   | 单字成词（Single）  |

![[Pasted image 20250422094722.png|375]]
### 2.3 Named Entity Recognition (NER)
Label tokens as Person (PER), Location (LOC), Organization (ORG), etc., using BIOES or BIO tagging schemes.

| 标签    | 说明                             |
| ----- | ------------------------------ |
| B-XXX | 实体起始位置，XXX 是类别（如 B-PER 表示人名开始） |
| I-XXX | 实体中间                           |
| E-XXX | 实体结束（BIOES 专有）                 |
| S-XXX | 单字实体（BIOES 专有）                 |
| O     | 非实体位置（Outside）                 |

![[Pasted image 20250422094631.png|350]]
### 2.4 Semantic Role Labeling (SRL)
Assign predicate-argument structure roles: ARG0 (agent), ARG1 (patient), PRED (predicate), etc.

| 标签    | 含义（依 PropBank） |
| ----- | -------------- |
| ARG0  | 动作的施事（Agent）   |
| ARG1  | 动作的受事（Patient） |
| PRED  | 谓词（Predicate）  |
| ARG2+ | 其他参与者（受益者、地点等） |
![[Pasted image 20250422094709.png|325]]

## ModelAndMethod

### HMM

- 假设标签序列 $y_1, ..., y_n$ 构成马尔可夫链
- 学习转移概率 + 发射概率
- generative model

![[Pasted image 20250422095632.png|450]]
#### HMM 的建模假设

HMM 将输入序列看作是**由隐状态序列生成观测序列的过程**。

#### 模型组成五元组：

$$
\text{HMM} = (S, O, A, B, \pi)
$$

| 符号                        | 含义                 |
| ------------------------- | ------------------ |
| $Y = \{s_1, ..., s_K\}$   | 状态集合（如词性标签）        |
| $X = \{o_1, ..., o_V\}$   | 观测集合（如单词、字）        |
| $Q = P(y_t \mid y_{t-1})$ | 状态转移概率矩阵transition |
| $E = P(x_t \mid y_t)$     | 发射概率矩阵emission     |
| $\pi = P(y_1)$            | 初始状态分布             |

---
#### Decoding（Viterbi decoding 算法）
##### 任务目标

给定观测序列 $\boldsymbol{x} = x_1, x_2, ..., x_n$，**寻找最可能的标签序列** ${\boldsymbol{y}} = y_1,y_2,...,y_n$：

$$
\hat{\boldsymbol{y}} = \arg\max_{\boldsymbol{y}} P(\boldsymbol{y} \mid \boldsymbol{x})
$$

根据贝叶斯定理：

$$
\arg\max_{\boldsymbol{y}} P(\boldsymbol{y} \mid \boldsymbol{x}) 
= \arg\max_{\boldsymbol{y}} \frac{P(\boldsymbol{x}, \boldsymbol{y})}{P(\boldsymbol{x})}
= \arg\max_{\boldsymbol{y}} P(\boldsymbol{x}, \boldsymbol{y})
$$

---

##### 联合概率展开公式


$$
P(\boldsymbol{x}, \boldsymbol{y}) = P(y_1) P(x_1 \mid y_1) \prod_{t=2}^{n} P(y_t \mid y_{t-1}) P(x_t \mid y_t)
$$

---


##### 目标定义

$$
\boldsymbol{y}^* = \arg\max_{y_1 \cdots y_n} P(x_1 \cdots x_n, y_1 \cdots y_n)
$$

定义动态规划子问题函数：

> $\pi(i, y_i)$ 表示前 $i$ 个词，以状态 $y_i$ 结尾的所有状态路径中最大概率值。

即：

$$
\pi(i, y_i) = \max_{y_1 \cdots y_{i-1}} P(x_1 \cdots x_i, y_1 \cdots y_i)
$$
##### 递推公式推导

从联合概率的定义出发：

$$
\begin{aligned}
\pi(i, y_i)
&= \max_{y_1 \cdots y_{i-1}} P(x_1 \cdots x_i, y_1 \cdots y_i) \\
&= \max_{y_1 \cdots y_{i-1}} P(x_i \mid y_i) \cdot P(y_i \mid y_{i-1}) \cdot P(x_1 \cdots x_{i-1}, y_1 \cdots y_{i-1}) \\
&= e(x_i \mid y_i) \cdot \max_{y_{i-1}} \left[ q(y_i \mid y_{i-1}) \cdot \pi(i-1, y_{i-1}) \right]
\end{aligned}
$$

其中：

- $e(x_i \mid y_i)$ 是发射概率；是在隐状态 $y_i$ 给定的条件下，生成观测值 $x_i$ 的概率。也就是：**“状态 $y_i$ 发射出观测 $x_i$ 的概率”**。
- $q(y_i \mid y_{i-1})$ 是状态转移概率；
- $\pi(i-1, y_{i-1})$ 是前一状态路径的最优值。
- $e$,$q$ 的值和$i$ 以及对应的$y_i$ 取值都有关

##### ✅ 初始条件：

设定起始状态 START：

$$
\pi(0, \text{START}) = 1
$$

##### ✅ 终止条件：

加入 STOP 状态，表示结束：

$$
\begin{aligned}
P(y^*) 
&= \max_{y_1 \cdots y_n} P(x_1 \cdots x_n, y_1 \cdots y_n, \text{STOP}) \\
&= \max_{y_n} q(\text{STOP} \mid y_n) \cdot \pi(n, y_n) \\
&= \pi(n+1, \text{STOP})
\end{aligned}
$$

---

##### 复杂度分析

- 每个时间步计算 $|Y|$ 个状态；
- 每个状态要从 $|Y|$ 个前驱状态中取最大值；
- 总体复杂度：$\mathcal{O}(n \cdot |Y|^2)$，其中 $n$ 是序列长度，$|Y|$ 是标签数。

---

##### 总结

Viterbi 算法通过动态规划，**避免穷举所有标签路径**，高效地找到最大概率路径（最优标签序列），是 HMM 和 CRF 中解码问题的标准解法。


#### 状态路径图（State Trellis）

![[Pasted image 20250422101932.png|500]]

每列代表时间步，每个节点是一个可能状态，红色箭头表示最优路径。  
路径得分为：

$$
\text{路径得分} = \prod_{i} e(x_i \mid y_i) \cdot q(y_i \mid y_{i-1})
$$

- **“State Trellis: Viterbi”** 表示用于“找最优路径”的网格，**最大化联合概率**；
    
- **“State Trellis: Marginal”** 表示用于“边缘推断”的网格，**对所有路径求和**；
---

#### 边缘推断：前向算法（Forward）
- Dynamic algorithm
目标：

$$
P(x_1, ..., x_n) = \sum_{y_1, ..., y_n} P(x_1, ..., x_n, y_1, ..., y_{n)}= \sum_{y_1, ..., y_n}P(y_1) P(x_1 \mid y_1) \prod_{t=2}^{n} P(y_t \mid y_{t-1}) P(x_t \mid y_t)
$$

引入前向变量：

$$
\alpha(i, y_i) = P(x_1, ..., x_i, y_i)
$$

递推公式：

$$
\alpha(i, y_i) = e(x_i \mid y_i) \cdot \sum_{y_{i-1}} \alpha(i-1, y_{i-1}) \cdot q(y_i \mid y_{i-1})
$$
![[Pasted image 20250416091140.png|400]]


初始：

$$
\alpha(0, y_0) = 
\begin{cases}
1 & \text{if } y_0 = \text{START} \\
0 & \text{otherwise}
\end{cases}
$$

终止：

$$
P(x_1 \cdots x_n) = \sum_{y_n} \alpha(n, y_n) \cdot q(\text{STOP} \mid y_n)
$$
![[Pasted image 20250416091246.png|375]]


复杂度：$O(n \cdot |Y|^2)$ 

---

#### HMM监督训练（Supervised Learning）

已知训练数据 $\{(x_1, y_1), ..., (x_n, y_n)\}$，使用[最大似然估计](./../数学基础_MathBasics/概率论与数理统计_ProbabilityAndStatistics/数理统计推断_StatisticalInference#2.2极大似然估计（MLE）)学习参数。

联合概率写作：

$$
P(x, y) = \prod_{i=1}^{n+1} e(x_i \mid y_i) \cdot q(y_i \mid y_{i-1})
$$

MLE 估计公式：

- 发射概率：

  $$
  e(x \mid y) = \frac{c(y, x)}{\sum_{x'} c(y, x')}
  $$

- 转移概率：

  $$
  q(y_i \mid y_{i-1}) = \frac{c(y_{i-1}, y_i)}{\sum_{y'} c(y_{i-1}, y')}
  $$

其中 $c(\cdot)$ 为计数（共现次数）。


#### HMM Unsupervised Learning

给定的数据形式：

$$
\{x_1, x_2, ..., x_n\}
$$

我们不知道每个 $x_i$ 对应的隐藏状态（比如词性、实体），只能观测到输入序列。

##### ✅ 目标函数：最大化边缘似然（Marginal Likelihood）

我们希望找到一组参数，使得**观测数据出现的概率最大**：

$$
\max_{\theta} \log P(x_1 \cdots x_n)
$$
$\theta$ ：
1. **初始状态概率分布** $P(y_1)$
2. **状态转移概率分布** $P(y_t \mid y_{t-1})$
3. **发射概率分布** $P(x_t \mid y_t)$

由于隐藏变量 $y_1 \cdots y_n$ 不可见，需要对所有可能路径求和：

$$
P(x_1 \cdots x_n) = \sum_{y_1 \cdots y_n} P(x_1 \cdots x_n, y_1 \cdots y_n)
$$

这个目标称为：**边缘似然（Marginal Likelihood）**

#####  EM 算法 for HMM $\to$ Baum-Welch Algorithm
- Can reach a local optimum but not necessarily a global optimum
[EM算法](../人工智能引论_IntroductionToAI/EM算法)

- **E 步**：计算当前参数下 $P(y_t = y \mid x)$ 等后验概率（用 forward-backward）
- **M 步**：利用期望统计重新估计参数（更新 $q(y_t \mid y_{t-1})$，$e(x_t \mid y_t)$）

##### Forward-Backward
- Forward 前向变量

定义：

$$
\alpha(i, y_i) = P(x_1, \dots, x_i,\ y_i)
$$
递推：

$$
\alpha(i, y_i) = \left( \sum_{y_{i-1} \in \mathcal{Y}} \alpha(i - 1, y_{i-1}) \cdot q(y_i \mid y_{i-1}) \right) \cdot e(x_i \mid y_i)
$$

-  Backward 后向变量

定义：

$$
\beta(i, y_i) = P(x_{i+1}, \dots, x_n \mid y_i)
$$
递推：

$$
\beta(i, y_i) = \sum_{y_{i+1} \in \mathcal{Y}} q(y_{i+1} \mid y_i) \cdot e(x_{i+1} \mid y_{i+1}) \cdot \beta(i+1, y_{i+1})
$$
- 单时刻边缘概率（标签后验）：

$$
\gamma(i, y_i) = P(y_i \mid x) = \frac{\alpha(i, y_i) \cdot \beta(i, y_i)}{P(x)}
$$

其中观测序列的总概率：

$$
P(x) = \sum_{y_n \in \mathcal{Y}} \alpha(n, y_n)
$$

- 相邻标签边缘概率（双标签后验）：

$$
\xi(i, y_i, y_{i+1}) = P(y_i, y_{i+1} \mid x) = \frac{ \alpha(i, y_i) \cdot q(y_{i+1} \mid y_i) \cdot e(x_{i+1} \mid y_{i+1}) \cdot \beta(i+1, y_{i+1}) }{P(x)}
$$
proof:
$$
P(y_i, y_{i+1} \mid x) = \frac{P(y_i, y_{i+1}, x)}{P(x)}
$$

把整个观测序列 $x = x_1, \dots, x_n$ 分成三段：
- 前缀 $x_1, \dots, x_i$
- 当前观测 $x_{i+1}$
- 后缀 $x_{i+2}, \dots, x_n$


$$
P(y_i, y_{i+1}, x) = P(x_1^i, y_i) \cdot P(y_{i+1} \mid y_i) \cdot P(x_{i+1} \mid y_{i+1}) \cdot P(x_{i+2}^n \mid y_{i+1})
$$
- $P(x_1^i, y_i) = \alpha(i, y_i)$
- $P(y_{i+1} \mid y_i) = q(y_{i+1} \mid y_i)$
- $P(x_{i+1} \mid y_{i+1}) = e(x_{i+1} \mid y_{i+1})$
- $P(x_{i+2}^n \mid y_{i+1}) = \beta(i+1, y_{i+1})$




🔁 迭代过程

**E Step**(Forward-Backward)：
1. 单标签边缘概率：
$$
\gamma_t(i) = P(y_t = i \mid x_{1:T})
$$
2. 相邻标签边缘概率：
$$
\xi_t(i,j) = P(y_t = i, y_{t+1} = j \mid x_{1:T})
$$

**M Step**：
初始概率：
$$
\pi(i) = \gamma_1(i)
$$
转移概率：
$$
q_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
$$
发射概率：
$$
e(j,x) = \frac{\sum_{t: x_t = x} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}
$$



### MEMM
![[Pasted image 20250425214152.png|325]]


$$
P(y_{1:n} \mid x_{1:n}) = \prod_{t=1}^n P(y_t \mid y_{t-1}, x_t)
$$
$$
P(y_t \mid y_{t-1}, x_t) = \frac{\exp(s(y_{t-1}, y_t, x_t))}{Z(y_{t-1}, x_t)}
$$
(其他定义和下面的相似，只是把$x_t$ 扩展到了$x_{1:n}$ ,所以只给出下面更general的定义。)
其中：

- $s(y_{t-1}, y_t, x_t) = \boldsymbol{w}^\top f(y_{t-1}, y_t, x_t)$ 是线性得分函数；
- $Z(y_{t-1}, x_t)$ 是局部归一化因子


为了让其参考上下文token($x$) ，模型变为：


![[Pasted image 20250425220116.png|350]]

$$
P(y_{1:n} \mid x_{1:n}) = \prod_{t=1}^n P(y_t \mid y_{t-1}, x_{1:n})
$$

$$
P(y_t \mid y_{t-1}, x_{1:n}) = \frac{\exp(s(y_{t-1}, y_t, x_t))}{Z(y_{t-1}, x_{1:n})}
$$
score function:

$$
s(y_{t-1}, y_t, x_{1:n}) = \mathbf{w}^\top f(y_{t-1}, y_t, x_{1:n})
$$

为了满足归一化条件：

$$
\sum_{y_t \in \mathcal{Y}} P(y_t \mid y_{t-1}, x_{1:n}) = 1
$$

必须有：（i.e. def of $Z(\dots)$） 

$$
Z(y_{t-1}, x_{1:n}) = \sum_{y_t \in \mathcal{Y}} \exp(s(y_{t-1}, y_t, x_{1:n}))
$$
含义：
$Z(y_{t-1}, x_{1:n})$ 是所有可能的 $y_t$ 下，打分指数和的归一化因子。

这是一个 **局部 softmax 归一化**，依赖于：

- 当前输入 $x_{1:n}$；
- 当前标签 $y_t$；
- 前一个标签 $y_{t-1}$。

但是他存在一个问题——label bias 
>[!note] label bias
> 假设我们一共有$Y$个label，那么对于两个不同的label $i$, $j$ 如果$q_{k_1,i},q_{k_2,i}$, but $q_{k_3,j}\dots q_{k_6,j}$ 

所以考虑全局归一化函数$\to$ CRF

- MEMM 使用的是局部归一化；
- CRF 使用的是**全局归一化**：

$$
Z(x_{1:n}) = \sum_{\hat{y}_{1:n}} \exp\left( \sum_t s(\hat{y}_{t-1}, \hat{y}_t, x_{1:n}) \right)
$$

### CRF

- 判别式模型，直接建模 $P(\boldsymbol{y} \mid \boldsymbol{x})$
- 允许使用丰富的特征函数
- 通常使用 Viterbi 算法解码最优标签序列

![[Pasted image 20250425222600.png|400]]




$$
P(y_{1:n} \mid x_{1:n}) = \frac{1}{Z(x_{1:n})} \prod_t \exp(s(y_{t-1}, y_t, x_{1:n}))
$$


$$
Z(x_{1:n}) = \sum_{y'_{1:n}} \prod_t \exp(s(y'_{t-1}, y'_t, x_{1:n}))
$$
#### Learning
##### Objective Function (Supervised Learning)

Given a set of labeled sequences $\{(x^{(i)}_{1:n}, y^{(i)}_{1:n})\}$, the CRF defines the conditional probability:


$$
P(y_{1:n} \mid x_{1:n}) = \frac{1}{Z(x_{1:n})} \prod_t \exp(s(y_{t-1}, y_t, x_{1:n}))
$$


where:
- $s(y_{t-1}, y_t, x_{1:n})$ is the **score function**, typically linear in feature weights.
- $Z(x_{1:n})$ is the **partition function**:


$$
Z(x_{1:n}) = \sum_{y'_{1:n}} \prod_t \exp(s(y'_{t-1}, y'_t, x_{1:n}))
$$

**Training Goal**

Maximize the log-likelihood over training data:

$$
\log P(y_{1:n} \mid x_{1:n}) = \sum_t s(y_{t-1}, y_t, x_{1:n}) - \log Z(x_{1:n})
$$

**Gradient Computation**

The gradient of the log-likelihood involves:

1. **Empirical counts** of features from the ground truth sequence.
2. **Expected counts** of features under the model distribution (requires computing marginal probabilities).

These expected counts are computed using the **Forward-Backward algorithm**, similar to HMMs.

---
**Optimization Methods**

- **Gradient Descent / L-BFGS**: Often used with auto-differentiation frameworks.
- **Structured SVM variant**: Uses a **margin-based loss**:


$$
L_{\text{SSVM}} = \max_{y \neq y^*} [s(y) + \Delta(y, y^*)] - s(y^*)
$$


where $\Delta$ is a task-specific cost (e.g., number of misclassified labels).

**Unsupervised Learning**

CRFs cannot model $P(x)$, making **unsupervised learning** tricky. A workaround is the **CRF Autoencoder (CRF-AE)**:
- CRF acts as the **encoder** to produce latent labels.
- A simple decoder tries to **reconstruct** the input sequence from the labels.
- The training objective maximizes the **reconstruction likelihood**.
![[Pasted image 20250425223511.png|200]]




### NeuralNetworks

#### RNN 
利用了context of each word，但是没有利用relations between neighboring labels

#### Bidirectional RNN
利用了context of each word，但是没有利用relations between neighboring labels
#### BiLSTM
- 使用双向 LSTM 处理上下文信息
- 输出每个位置的表示用于分类
利用了context of each word，但是没有利用relations between neighboring labels
#### Transformer / BERT
利用了context of each word，但是没有利用relations between neighboring labels

#### BiLSTM/Transformer + CRF
同时利用了context of each word，relations between neighboring labels
- Neural Network 计算CRF中的score，CRF 去预测标签
- 是目前主流方法之一

### EvaluationOfModel

| 模型类型                               | 上下文感知 😢（1）                              | 标签间建模 🙂（2）             |
| ---------------------------------- | ---------------------------------------- | ----------------------- |
| **HMM** (Hidden Markov Model)      | ❌ 不考虑上下文特征（仅基于当前位置的观测）                   | ✅ 显式建模 $P(y_t∣y_{t−1})$ |
| **CRF** (Conditional Random Field) | ❌（线性 CRF 仅使用当前位置特征）  <br>（但可通过手工特征加入上下文） | ✅ 显式建模整个标签序列的条件概率       |
| **独立神经网络**（如 BiLSTM + Softmax）     | ✅ 网络可捕捉上下文（如 BiLSTM/Transformer）         | ❌ 每个位置独立预测，不建模标签依赖      |
| **神经网络 + CRF**（如 BiLSTM + CRF）     | ✅ 上下文由网络建模（如 BiLSTM）                     | ✅ 标签依赖由 CRF 层建模         |







---



## LossFunctionsAndTraining

- 对于分类模型（如 BiLSTM），常用：
  $$
  \mathcal{L} = - \sum_{i=1}^{n} \log P(y_i \mid h_i)
  $$

- 对于 CRF，则使用：
  $$
  \mathcal{L} = -\log P(\boldsymbol{y} \mid \boldsymbol{x}) = -\left( s(\boldsymbol{x}, \boldsymbol{y}) - \log Z(\boldsymbol{x}) \right)
  $$

其中 $s(\cdot)$ 是打分函数，$Z(\cdot)$ 是配分函数。



