

## What is a Dependency Parse?
- A **directed tree** with words as nodes and syntactic dependencies as edges.
- ROOT is a special node pointing to the root of the sentence.
- Edges may be **typed/labeled** (e.g., `nsubj`, `obj`).
### Goal 
Given a sentence and return the best parse tree
### Advantages:
- Better for free word-order languages (e.g., Czech, Turkish).
- More cross-lingually consistent than constituency.
- Often better for downstream tasks.
- Faster than CFG-based parsing.

### Disadvantages:
- Less consensus on definition of dependencies.
- Harder to map to formal semantics.
### Properties
1. **树结构性质**
- 是一棵 **有向树（Directed Tree）**
- 设句子长度为 $n$，则：
  - 有 $n + 1$ 个节点（含一个虚拟 `ROOT`）
  - 有 $n$ 条依存边
- 所有节点（除 `ROOT`）的 **入度恰好为 1**（只有一个 head）
- `ROOT` 节点的入度为 0，且只能有一个子节点（句子的 syntactic head）

2. **单头性（Single-head Constraint）**
每个词只能有一个父节点（head）：

$$
\forall w_i \in \text{sentence},\quad \exists! \; h_i \text{ such that } h_i \rightarrow w_i
$$

否则会导致冲突，无法形成树结构。

 3. **无环性（Acyclic）**
依存关系不允许形成闭环：

$$
\text{不存在} \; w_i \rightarrow \dots \rightarrow w_j \rightarrow w_i
$$

即：依存树是一个无环图（DAG），且还是一棵树。

4. **连通性（Connectedness）**
整棵依存树是连通的：从 `ROOT` 出发，能遍历到所有词。


## Projectivity
- A parse is **projective** if no dependency arcs cross when drawn above the sentence.
- Most natural language parses are projective, but **non-projective** structures exist.

一个依存树是投射性的，当且仅当：

> **若存在一条从单词 $w_i$ 指向 $w_j$ 的依存边（arc），则在 $w_i$ 和 $w_j$ 之间的所有单词 $w_k$ 都是 $w_i$ 的后代（即 $w_k$ 是 $w_i$ 的后裔，或者 $w_i$ 的后裔的后裔）。**

更直观的说法：

> **将句子中的单词按线性顺序从左到右排列，若将所有依存边画在这些词的上方，则不存在任何两条边相交（crossing arcs），则该依存树是投射性的。**

>方法 1：边交叉检查

1. 对每一对依存弧 $i\to j$ 和 $k\to l$，其中 $i < j$ 且 $k < l$，检查以下是否成立：
   - $i < k < j < l$ 或 $k < i < l < j$

1. 如果存在任意一对满足上述条件，则这两条边交叉 $\to$ 非投射性。

>方法 2：检查路径中断

对于每一条依存边 $i \rightarrow j$，取区间 $[i+1, j-1]$（假设 $i < j$），然后：

- 检查这个区间内的所有单词是否都在 $i$ 的子树中。
- 如果存在不在其子树中的词 → 非投射性。

## Transformations

### Dependency to Constituency
- A projective dependency tree can be converted to a constituency tree.
- The subtree rooted at each word corresponds to a constituent.
- If the dependency tree is non-projectivity, it can be converted to a discontinuous-consistency tree
### Constituency to Dependency
- Identify heads in the constituency tree.
- Constituent tree with heads,lexicalized
- Remove label and merge
- The final dependency tree is definitely  projectivity

---


## Dependency Parsing Algorithms

### Graph-Based Parsing

#### First-order Parsing
- **Arc-scoring model**: Score each arc individually.
- Parse tree score = sum of arc scores.

##### Methods:
- **MST (Chu-Liu/Edmonds)**
- **Eisner's algorithm**: For projective parsing, $O(n^3)$ time.
- **CYK**: Can model dependency with CFG; $O(n^5)$ time.

#### Parsing Strategy
- Predict all arcs independently.
- Construct tree via MST or head-selection.
- Label arcs after tree is built.

### # Evaluation
- **UAS**: Unlabeled attachment score.
- **LAS**: Labeled attachment score.

---

### Second-order Parsing
- Score **arc pairs** (e.g., siblings, grandparents).
- Higher accuracy.
- More expensive: $O(n^4)$ for projective, NP-hard for non-projective.

---

## Evaluation

![[Pasted image 20250416141506.png|400]]

i.e. 任意parsing tree都能写成一个三元组

- UAS:unlabeled attachment score
忽略预测标签，只看父节点是否正确

- LAS:labeled attachment score
只有父节点和依存标签都对才算正确












## Graph-based Dependency Parsing 

### First-order graph-based
- score each arc and sum them up(Assumes that each arc is scored independently of others.)
- standard: features of the two word: neighbors, POS tags, contextual word

#### Parsing Strategies

##### 1. Head-Selection
- For each word, choose the head with the highest scoring arc.
- **Problem**: May produce **invalid trees** such as:
  - Cycles
  - Multiple roots
  - Disconnected graphs
##### 2. Maximum Spanning Tree (MST)
- Construct a complete directed graph with arc scores.
- Use algorithms to find the **maximum scoring directed spanning tree**.

##### CYK

![[Pasted image 20250423100541.png|350]]
Convert to CFG and use CYK
- Time complexity: $n^{3}|G|= n^5$ 
- 任何可能得依存边都必须构建一条规则

#### Algorithms:
- **Chu-Liu/Edmonds algorithm** (for non-projective trees)
- **Eisner's algorithm** (for projective trees)

#### Time Complexities:
- Chu-Liu/Edmonds: $O(n^2 + n\log n)$ (fast implementation)
- Eisner’s: $O(n^3)$
 
---

## Labeling Dependencies

After finding the structure (i.e., which word points to which), we label each arc:
- Predict a label (e.g., `nsubj`, `obj`, `det`) using a classifier.
- Usually done **separately** from structure prediction.

---

## Advantages of First-Order Models

- **Simplicity**: Easier to implement and understand.
- **Speed**: Efficient decoding algorithms available.
- **Competitive performance**: Strong baselines, especially when combined with neural representations.

---

## Limitations

- Ignores interactions between arcs (e.g., does not consider whether two dependents share the same head).
- May miss structural constraints or preferences captured by higher-order models.

---

## Summary

| Aspect | First-Order Parsing |
|--------|---------------------|
| Score Granularity | Individual arcs only |
| Decoding | Eisner (projective), Chu-Liu/Edmonds (non-projective) |
| Features | Head + dependent only |
| Complexity | $O(n^3)$ (Eisner), $O(n^2 + n\log n)$ (CLE) |
| Pros | Fast, simple, strong baseline |
| Cons | No modeling of arc interactions |

### Second-order graph-based
Parse tree scoring
- Each connected pair of arcs has a score. 
- The tree score is the sum of arc-pair scores.

![[Pasted image 20250423103309.png|250]]
![[Pasted image 20250423103404.png|250]]
## Learning

### Supervised
- Train on labeled treebanks.
- Objective: Maximize $P(\text{tree} \mid \text{sentence})$
- Optimization: SGD
- Use partition functions (sum over all trees):
  - **Projective**: Use inside algorithm on Eisner.
  - **Non-projective**: Use Kirchhoff's Matrix Tree Theorem.

![[Pasted image 20250423105107.png|375]]
### Unsupervised
- No treebank needed.
#### Generative model
- Use PCFG
	- eg. Dependency Model with Valence(DMV)
		- Run EM or SGD to max(likelihood P(sentence))
#### Discriminative model
- CRF-Autoencoder
	- Encoder: a graph-based dependency parser
	- Decoder: predict each word from its head
- Maximize reconstruction probability using SGD


## Transition-Based Dependency Parsing

### Core Idea
- Represent parsing as a **sequence of transitions**.

### Parser Configuration
- **Buffer $B$**: Input words not yet processed.
- **Stack $S$**: Partial tree under construction.

![[Pasted image 20250423105338.png|350]]
### Transitions
![[Pasted image 20250423105647.png|375]]
### Properties:
- No backtracking.
- Linear time complexity.（$2n$）

---

# Summary
- Dependency parsing supports both **graph-based** and **transition-based** methods.
- Graph-based methods find global optimum but are slower.
- Transition-based methods use greedy or beam search for fast parsing.
- Parsing can be cast as a sequence labeling task for speed.
- Supervised and unsupervised learning both exist.
- Projectivity is key to choosing the right algorithm (Eisner vs. MST).


















































投射性和非投射性：
Projectivity vs. Non-projectivity

根据依存关系弧是否有交叉，如果没有就是Projectivity



#### Dependency$\to$constituency
![[Pasted image 20250416140434.png|425]]
把子树展开即可


#### Non-projectivity $\to$ Discontinuous Constituent

![[Pasted image 20250416140405.png|350]]


#### Constituency $\to$ dependency

- 先选head
- 再词化（lexicalized）
- 然后直保留单词，去掉标签
- 最后合并重复词，得到dependency tree


### Algorithms 
- Graph-based parsing: MST, Eisner
assume independence between different parts of a parse tree
find the global optimum
- Transition-based parsing: arc-standard, arc-eager, arc hybrid
no independence assumption
local optimum, fast

### Evaluating dependency parsing



### Graph-Based Dependency Parsing

