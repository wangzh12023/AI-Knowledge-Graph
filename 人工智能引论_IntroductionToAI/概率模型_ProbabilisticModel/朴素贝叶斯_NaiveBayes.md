

>[!note] 
>本文章参考自[blog](https://www.cnblogs.com/xfuture/p/17838109.html)


## 贝叶斯定理

**贝叶斯定理（Bayes' Theorem）** 就是我们熟悉的贝叶斯公式：
$$P(A \mid B)=\frac{P(B \mid A) \times P(A)}{P(B)}$$
## 贝叶斯定理基础

![[Pasted image 20250410114521.png|350]]


>[!caution] 以下内容源自ChatGPT


- **先验概率（Prior Probability）**  
    👉 在观察到数据之前，你对事件的原始估计。
    
- **可能性函数（似然函数）（Likelihood Function）**  
    👉 观察到某个数据/证据，在假设某个参数/事件为真的情况下，这个数据出现的可能性。
    
- **贝叶斯定理（Bayes' Theorem）**  
    👉 将先验概率和似然结合起来，计算更新后的信念（后验概率）。
    
- **后验概率（Posterior Probability）**  
    👉 给定观察数据后，你对事件的更新估计（推理结果）。
    


### 🧮 数学公式表示：

贝叶斯定理如下：

$$
P(H \mid D) = \frac{P(D \mid H) \cdot P(H)}{P(D)}
$$

- $H$：某个假设，例如“你感冒了”
- $D$：数据或观察结果，例如“你正在咳嗽”
- $P(H)$：先验概率（你感冒的概率）
- $P(D \mid H)$：可能性函数（你感冒时咳嗽的概率）
- $P(H \mid D)$：后验概率（你咳嗽时感冒的概率）

---

### 🌰 举个例子：你看到自己咳嗽了，是否可能感冒？

> [!note] 
> 假设
>- 总体中 5% 的人感冒（先验概率）：$P(\text{感冒}) = 0.05$
>- 如果你感冒了，有 80% 的概率会咳嗽：$P(\text{咳嗽} \mid \text{感冒}) = 0.8$
>- 不感冒时，也有 10% 的人咳嗽：$P(\text{咳嗽} \mid \text{没感冒}) = 0.1$

你现在咳嗽了，问：你感冒的概率是多少？（也就是后验概率）

---

#### 解：

根据贝叶斯公式：

$$
P(\text{感冒} \mid \text{咳嗽}) = \frac{P(\text{咳嗽} \mid \text{感冒}) \cdot P(\text{感冒})}{P(\text{咳嗽})}
$$

其中：
$$
P(\text{咳嗽}) = P(\text{咳嗽} \mid \text{感冒}) \cdot P(\text{感冒}) + P(\text{咳嗽} \mid \text{没感冒}) \cdot P(\text{没感冒}) \\
= 0.8 \cdot 0.05 + 0.1 \cdot 0.95 = 0.04 + 0.095 = 0.135
$$

带入：

$$
P(\text{感冒} \mid \text{咳嗽}) = \frac{0.8 \cdot 0.05}{0.135} \approx \frac{0.04}{0.135} \approx 0.296
$$

---

✅ **结论：虽然你咳嗽了，但你感冒的概率只有 29.6%，并不是很高**。  
这体现了先验概率的“克制”作用：不能因为看到了咳嗽就武断断定感冒，还要结合感冒的基率。

---

### 📌 总结：

| 概念                     | 解释          |
| ---------------------- | ----------- |
| **先验概率 (Prior)**       | 你原来的信念      |
| **可能性函数 (Likelihood)** | 数据在某假设下的可能性 |
| **后验概率 (Posterior)**   | 你更新后的信念     |


## 朴素贝叶斯算法

$$
P\left(C_k \mid x_1, x_2, \ldots, x_n\right)=\frac{P\left(C_k\right) \prod_{i=1}^n P\left(x_i \mid C_k\right)}{\sum_j P\left(C_j\right) \prod_{i=1}^n P\left(x_i \mid C_j\right)}
$$

其中：
- $P\left(C_k \mid x_1, x_2, \ldots, x_n\right)$ 是给定特征向量的情况下，样本属于类别 $C_k$ 的后验概率。
- $P\left(C_k\right)$ 是类别 $C_k$ 的先验概率。
- $P\left(x_i \mid C_k\right)$ 是给定样本属于类别 $C_k$ 的情况下，特征 $x_i$ 出现的条件概率。



>[!caution] 以下内容源自ChatGPT

### 🧠 什么是朴素贝叶斯（Naive Bayes）？

朴素贝叶斯是一种基于**贝叶斯定理（Bayes Theorem）**的**监督学习算法**，主要用于**分类任务**（Classification）。

> 🎯 它的核心思想是：
> > 给定一个待分类样本，利用贝叶斯定理计算它属于某个类别的后验概率，然后选择后验概率最大的那个类别作为预测结果。

---

### 💡 数学原理

假设我们要预测样本 $X = (x_1, x_2, ..., x_n)$ 属于某个类别 $y$，  
根据贝叶斯定理：

$$
P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}
$$

我们只关心谁最大，可以忽略分母：

$$
P(y \mid X) \propto P(X \mid y) \cdot P(y)
$$

#### 🔍 朴素假设（Naive Assumption）：
朴素贝叶斯的“朴素”就在于：**假设特征之间是条件独立的**，即：

$$
P(X \mid y) = \prod_{i=1}^n P(x_i \mid y)
$$

所以：

$$
P(y \mid X) \propto P(y) \cdot \prod_{i=1}^n P(x_i \mid y)
$$

---

### 📚 举个例子：邮件垃圾分类（Spam vs. Ham）

假设你有以下训练数据：

| 邮件内容 | 类别 |
|----------|------|
| "Win money now" | Spam |
| "Limited time offer" | Spam |
| "Meeting at noon" | Ham |
| "Lunch tomorrow?" | Ham |

现在给你一个新邮件："Win lunch offer"，你要判断它是 **Spam** 还是 **Ham**。

#### 步骤如下：

1. **先验概率**（从训练集中统计）：
   - $P(\text{Spam}) = \frac{2}{4} = 0.5$
   - $P(\text{Ham}) = \frac{2}{4} = 0.5$

2. **词频统计**（假设所有单词是独立的）：
   - 在Spam中：
     - "win" 出现1次，"offer" 1次，"lunch" 0次
   - 在Ham中：
     - "win" 0次，"offer" 0次，"lunch" 1次
   - 可能加入平滑项（Laplace Smoothing）防止0概率。

3. **计算后验概率**（只比较相对大小）：
   $$
   P(\text{Spam} \mid \text{"win lunch offer"}) \propto P(\text{Spam}) \cdot P(\text{"win"} \mid \text{Spam}) \cdot P(\text{"lunch"} \mid \text{Spam}) \cdot P(\text{"offer"} \mid \text{Spam})
   $$
   $$
   P(\text{Ham} \mid \text{"win lunch offer"}) \propto P(\text{Ham}) \cdot P(\text{"win"} \mid \text{Ham}) \cdot P(\text{"lunch"} \mid \text{Ham}) \cdot P(\text{"offer"} \mid \text{Ham})
   $$

最后选取较大的那一个作为预测分类。

---

### 🛠 应用场景

- 📩 垃圾邮件分类
- 💬 文本情感分析（positive/negative）
- 🌐 文档主题识别
- 📋 医疗诊断（给定症状，预测疾病）

---

### ✅ 优点

- 简单、快速，训练和预测效率都很高
- 对小数据量也比较鲁棒
- 对高维数据（如文本）尤其有效

### ⚠️ 缺点

- “朴素”的独立性假设不太现实
- 特征相关性强时性能下降


## 朴素贝叶斯的种类变体



