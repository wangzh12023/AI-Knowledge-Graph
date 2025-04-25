
## Table of Contents

- [1. Introduction to Semantics](#1-introduction-to-semantics)
  - [Two Types of Meaning Representations](#two-types-of-meaning-representations)
- [2. Lexical Semantics](#2-lexical-semantics)
  - [Word Senses](#word-senses)
  - [Semantic Relations](#semantic-relations)
  - [WordNet](#wordnet)
  - [Word Sense Disambiguation (WSD)](#word-sense-disambiguation-wsd)
- [3. Sentence Semantics](#3-sentence-semantics)
  - [Formal Meaning Representation](#formal-meaning-representation)
  - [Representations](#representations)
- [4. Semantic Graphs](#4-semantic-graphs)
- [5. Semantic Parsing](#5-semantic-parsing)
  - [Methods](#methods)
- [6. Semantic Role Labeling (SRL)](#6-semantic-role-labeling-srl)
  - [Goal](#goal)
  - [Role Resources](#role-resources)
  - [Methods](#methods-1)
- [7. Information Extraction (IE)](#7-information-extraction-ie)
  - [Tasks](#tasks)
  - [NER Techniques](#ner-techniques)
  - [Relation Extraction](#relation-extraction)
- [8. Summary](#8-summary)

## 1. Introduction to Semantics
Semantics is the study of meaning, connecting language to the real world.

Two Types of Meaning Representations:
- **Implicit (Vector semantics)**: e.g., word embeddings
- **Explicit (Symbolic representations)**: formal logic, semantic graphs

## Lexical Semantics

### Word Senses
- A lemma（条目） is a dictionary headword  form.（词条在词典中的“原型”形式）
- word senses 指多义单词的具体一条语义
	- **Lemma**：词典中的“原型形式”，例如：
		  - `run` 是 `ran`, `running` 的 lemma
		  - `mouse` 是 `mice` 的 lemma
	- **Word Sense**：一个 lemma 的具体含义（可以有多个）  
		  - `mouse.n.01`：小动物  
		  - `mouse.n.02`：计算机输入设备

- Words can have multiple meanings （一个 lemma 可以拥有多个“义项”（polysemy））
- Techniques to distinguish senses:
	- Independent truth conditions
	- Different syntactic behaviors
	- Antagonistic meanings
	- Independent sense relations


| 判据 | 核心思想 | 操作方法 | 经典示例 | 适用场景 & 局限 |
|------|----------|----------|-----------|------------------|
| **独立的真值条件**<br>(independent truth conditions) | 在可区分的情境中，两种用法的真值条件互不重叠 ⇒ 不同词义 | 1. 构造语境使用法 A 成真、B 必假；<br>2. 反之亦然 | *serve*<br>• “They rarely **serve** red meat.”（提供食物）<br>• “He **served** his time in prison.”（服刑）<br>两句可同时为真，但无单一情境可兼顾二者&#8203;:contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1} | **优**：最贴近语义学定义<br>**局**：需假设完整语境，实践中难穷尽 |
| **不同的句法行为**<br>(different syntactic behaviors) | 若搭配/句式要求系统性不同，多为不同义 | 比较必选/可选补语、介词、形态等 | *serve*<br>• “serve + NP”<br>• “serve **as** NP”&#8203;:contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3} | **优**：易用语料库自动探测<br>**局**：句法差异亦可能是体裁/方言造成 |
| **语义对立**<br>(antagonistic meanings) | 并列两用法造成语义冲突 ⇒ 概念不同 | 将两候选义放同一并列结构，检验可接受度 | “Does Air France **serve breakfast** and **Philadelphia**?”（不自然，语义冲突）&#8203;:contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5} | **优**：快速共现测试<br>**局**：需能构造自然句；冲突或因常识而非多义 |
| **独立的语义关系**<br>(independent sense relations) | 各自拥有不同的上/下位词、同义词网络 | 借助 WordNet/词典查看超-下位链 | *serve*<br>• “提供”义的上位词 = *provide*<br>• “服刑”义的上位词 = *spend*&#8203;:contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7} | **优**：利于算法化（WordNet 路径）<br>**局**：依赖资源完备度；细粒度义未必收录 |

### Semantic Relations

关系 | 说明 | 例子
---|---|---
Synonymy | 同义 | small, little
Antonymy | 反义 | small, large
Hyponymy | 下位（is-a） | dog ⊂ mammal
Hypernymy | 上位 | mammal ⊃ dog
Meronymy | 部分-整体 | liver ⊂ body
Holonymy | 整体-部分 | body ⊃ liver

### WordNet
- Lexical database organizing word senses into synsets.
- Supports semantic distance calculations.
- Synset: group of word senses that are synonymous
- Synsets are associated to one another by semantic 
relations

WordNet 是一个将单词的不同义项组织成语义网络的词汇知识库。它的基本单位是 **synset（同义词集）**，各个 synset 之间通过语义关系相连。

| 组件 (Component) | 作用 (Role) | 说明 (Description) |
|------------------|-------------|---------------------|
| **Synset** | 最小语义单元，由一组近义词构成 | 如 `dog.n.01` = {dog, domestic dog, Canis familiaris} |
| **Lemma** | 表面词形（如文本中出现的“dogs”, “ran”） | 一个 lemma 可对应多个 synset（多义性） |
| **POS 标签** | 词性标记：n（名词）、v（动词）、a（形容词）、r（副词） | 如 `run.n.01` ≠ `run.v.01` |
| **Pointer** | 连接 synset 的语义关系 | 如 hypernym（上位词）、hyponym（下位词）、meronym（部分）等 |
| **Gloss** | 简短定义 + 示例句 | 如：“A domesticated carnivorous mammal…” |


| 关系 (Relation) | 含义 (Meaning) | 示例 (`dog.n.01`) |
|------------------|----------------|-------------------|
| **Hypernym** | 上位词 / is-a | `canine.n.02`（犬科动物） |
| **Hyponym** | 下位词 / 包含在 | `puppy.n.01`, `lapdog.n.01` |
| **Meronym** | 部分 | `flag.n.07`（尾巴） |
| **Holonym** | 整体 / 所属 | `pack.n.06`（狗群） |
| **Antonym** | 反义词 | 如 `hot` ↔ `cold` |
| **Entailment / Cause** | 动词义项的蕴含/因果 | `snore → sleep`, `kill → die` |

语义距离与相似度 | Semantic Distance & Similarity

- WordNet 可以通过“最短路径长度”计算两个词义之间的**语义距离**
- 两词的“语义距离”= 两者 synset 在 hypernym/hyponym 图中的最短路径长度


### Word Sense Disambiguation (WSD)

Def: 自动判断一个词在上下文中所表达的具体义项。  

**输入**：一个带上下文的句子  
**输出**：WordNet 的 synset 编号（如 `bank.n.01`）

#### Methods
1. 知识库方法（基于规则）
- **Lesk 算法（1986）**：最经典规则法  
  - 对比上下文单词与每个词义的定义（gloss）之间的词重叠  
  - 选择重叠最多的义项  
- 改进版本：extended Lesk、Banerjee & Pedersen

**优点**：不依赖训练语料，适合低资源语言  
**缺点**：精度低，难扩展到深层语义匹配

2. 监督学习方法（统计模型）

- 将 WSD 视为分类问题：给定上下文，预测正确 sense  
- 特征：
  - 上下文单词（窗口）
  - 词性、句法依存关系
  - 词的位置、距离等  
- 模型：决策树、SVM、MaxEnt 等

**优点**：效果好，解释性强  
**缺点**：依赖 sense 标注语料，如 **SemCor**, Senseval

3. 神经网络方法（上下文编码）

A. 基于上下文向量（Contextual Embedding）

- 使用 **BERT / RoBERTa** 等模型提取目标词在上下文中的向量表示
- 方法：
  - 比较上下文向量与每个 synset 的定义向量（通过编码 gloss）进行相似度匹配
  - 或直接 fine-tune WSD 分类头

B. Sequence Labeling

- 将 WSD 作为一个“序列标注任务”：每个词打上 sense 标签  


## 3. Sentence Semantics
>[!note] Def 
>Sentence Semantics is the study of how the meanings of individual words combine to form the meaning of a whole sentence.


### Formal Meaning Representation

- Should be unambiguous, canonical, expressive, and support inference.

### Representations:
- **Database queries**
- **Robot control commands**
- **First-Order Logic (FOL)**

### FOL Examples:
- "Alice is not tall" → ¬Tall(a)
- "Some people like broccoli" → ∃x, Human(x) ∧ Likes(x, br)

## 4. Semantic Graphs

### Types:
- **Flavor 0**: Node = word
- **Flavor 1**: Node = arbitrary sentence part
- **Flavor 2**: Node = unanchored concept

## 5. Semantic Parsing

### Methods:
- **Syntax-driven**: Synchronous context-free grammars (SCFG)
- **Neural approaches**: seq2seq models, graph generation


## Formal Meaning Representation

一个好的语义表示应该具有：

| 属性 | 解释 |
|------|------|
| **Unambiguity（唯一性）** | 一个表示只能表达一个含义 |
| **Canonical Form（规范形式）** | 一个含义应该只有一个形式 |
| **Expressiveness（表达力）** | 能处理广泛主题和语言现象 |
| **Inference Ability（可推理性）** | 能用于知识推理、QA 等任务 |

> ⚠️ 这四者之间存在权衡（越精确越难用）

### 常见的意义表示方法

#### ✅ 特定用途（Special-purpose）

- **数据库查询语言**：将句子翻译为 SQL 语句
- **机器人控制命令**：转为动作指令，如 `(move forward 10m)`
- **语音助手意图表示**：如 `{intent: play_music, song: ...}`

#### ✅ 通用意义表示（General-purpose）

| 表示方式 | 说明 |
|----------|------|
| **一阶逻辑（First-Order Logic, FOL）** | 用谓词逻辑表达语义关系 |
| **语义图（Semantic Graphs）** | 节点表示概念，边表示关系，例如 AMR, EDS |

---

#### 5️⃣ 一阶逻辑表示（FOL）

Term+formula

| 自然语言句子 | 一阶逻辑表示 |
|----------------|----------------|
| Alice is not tall | `¬Tall(a)` |
| Some people like broccoli | `∃x (Human(x) ∧ Likes(x, br))` |
| Every restaurant has a long wait or is disliked by Adrian | `∀x (Restaurant(x) ⇒ LongWait(x) ∨ ¬Likes(a, x))` |

---

#### 6️⃣ 语义图（Semantic Graphs）

语义图通过**图结构**表示句子的逻辑成分，节点为概念或词，边为语义关系。

三种“语义图” flavor：

| Flavor | Node 是什么？ | 示例 |
|--------|---------------|------|
| **Flavor 0** | 词（word） | DM (Minimal Recursion Semantics) |
| **Flavor 1** | 子词/词组 | EDS (Elementary Dependency Structures) |
| **Flavor 2** | 抽象节点（不一定与文本锚定） | AMR (Abstract Meaning Representation) |

---

## Semantic Parsing

> Translating a sentence to its semantic representation
> 将自然语言句子“翻译”为机器可执行的结构化语义表示，如FOL, Semantic Graphs, SQL sentence


### 基于语法驱动（Syntax-driven）

- 遵循组成性原则（Principle of Compositionality）：短语的意义由其子短语的意义决定
	- So we follow a constituency syntactic tree to compose the semantic representation
- 用“同步上下文无关文法Synchronous context-free grammar (SCFG)”构建 NaturalLanguage 和formal language的双语法树：

  - 每条规则生成一对字符串：一个用于 NL，另一个用于语义语言（如逻辑式）
  - 支持结构对齐，如：

    ```
    RULE → if COND , then DIR. / (COND DIR)
    COND → our player 1 has the ball / (bowner our 1)
    ```

---

### 神经网络方法（Neural Semantic Parsing）

| 方法类型 | 描述 |
|----------|------|
| **Seq2Seq** | 将句子线性编码为语义公式或图（如逻辑表达式、DFS 序列） |
| **图生成模型** | 先预测节点，再预测边；类似 dependency parsing |
| **基于过渡系统** | 类似 shift-reduce parser，用操作构建语义图 |

#### 学习策略（Learning Methods）

| 类型 | 说明 |
|------|------|
| **监督学习（Supervised）** | 使用带有语义结构标注的数据训练 |
| **弱监督 / 半监督（Weakly Supervised）** | 仅使用句子和其答案（如 QA 中的答案） |
| **强化学习（Reinforcement Learning）** | 在没有标注语义结构时，通过执行结果反向优化模型 |

## 6. Semantic Role Labeling (SRL)

### Goal:
- Identify predicate-argument structures (who did what to whom, where?)

### Role Resources:
- **PropBank**: Arg0 (agent), Arg1 (patient), ArgM (modifiers)
- **FrameNet**: Role definitions tied to predicates

### Methods:
- **Sequence labeling** (BIO scheme)
- **Graph-based methods**
- **Seq2seq models**

## 7. Information Extraction (IE)

### Tasks:
- **Named Entity Recognition (NER)**
- **Entity Linking**: map to knowledge bases (e.g., Wikidata)
- **Relation Extraction**: find relations between entities
- **Event Extraction**: identify event types, triggers, arguments

### NER Techniques:
- Sequence labeling
- Span classification

### Relation Extraction:
- Predict relations between entity spans using syntactic paths and context

## 8. Summary
- Vector vs. symbolic sentence representations
- Lexical semantics and WordNet
- Word sense disambiguation
- Sentence semantics via formal representations
- Semantic parsing and SRL
- Information extraction techniques