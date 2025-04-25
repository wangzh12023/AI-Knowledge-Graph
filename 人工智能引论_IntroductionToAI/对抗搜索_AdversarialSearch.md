
> [!caution] 
> 本文 **部分** 使用了ChatGPT，请注意识别

# 对抗搜索_AdversarialSearch


> 对抗搜索常用于两人博弈类问题，如国际象棋、围棋、五子棋、井字棋等。此类问题具有非确定性与多方参与的特征，因此需要特别的搜索算法。

---

## 1.Minimax 算法（极小极大算法）

Minimax 是一种在完全信息、回合制、零和博弈中用于决策的策略，假设对手总是做出最优策略。

###  适用场景：
- 完全信息的双人对抗游戏（如井字棋、国际象棋）
- 零和博弈

### 算法原理：
- MAX 玩家试图 **最大化** 自己的得分；
- MIN 玩家试图 **最小化** MAX 的得分；
- 使用递归方式展开整个博弈树，直到叶子节点（终止状态）。

### 核心思想：
```text
MAX 节点：选子节点中最大值
MIN 节点：选子节点中最小值
````

### 示例伪代码：

```python
def minimax(node, depth, isMax):
    if node is terminal or depth == 0:
        return utility(node)

    if isMax:
        value = -inf
        for child in successors(node):
            value = max(value, minimax(child, depth-1, False))
        return value
    else:
        value = +inf
        for child in successors(node):
            value = min(value, minimax(child, depth-1, True))
        return value
```

---

## 2.Expectimax 搜索（期望最大搜索）

当对手的行为具有**随机性**时（不是总是最优），Minimax 就不再合适，此时使用 Expectimax。

### 🤖 特点：

- 适用于**非确定性游戏**（如有骰子、概率事件的游戏）
    
- 引入了 **期望值计算**（不是选最小或最大，而是平均）
    

### 🧠 核心节点类型：

- MAX 节点：尝试最大化自身得分
    
- **Chance 节点**：表示随机行为，返回所有子节点的期望值

>[!warning] 
>chance 节点一般是不进行剪枝操作的，除非题目给定子节点的范围，导致无论如何的expectation比率都不会让上层节点选中这个chance节点。

### 💡 示例伪代码：

```python
def expectimax(node, depth, agent):
    if node is terminal or depth == 0:
        return utility(node)

    if agent == MAX:
        return max(expectimax(child, depth-1, nextAgent) for child in successors(node))
    elif agent == CHANCE:
        total = 0
        for child in successors(node):
            prob = probability(child)
            total += prob * expectimax(child, depth-1, nextAgent)
        return total
```

---

## 3.Alpha-Beta 剪枝


Alpha-Beta 是对 Minimax 算法的剪枝优化，可以大幅减少需要搜索的节点数量。

### 🧪 原理：

- 使用两个边界值：
    
    - α（Alpha）：当前最大值下界（MAX 已知的最好值）
        
    - β（Beta）：当前最小值上界（MIN 已知的最好值）
        
- 若某节点不可能比已知更优，则**剪枝跳过**。
    

### ✂️ 剪枝条件：

```text
当 β ≤ α 时，剪枝（不再展开子节点）
```

### 💡 示例伪代码：

```python
def alphabeta(node, depth, α, β, maximizingPlayer):
    if node is terminal or depth == 0:
        return utility(node)

    if maximizingPlayer:
        value = -inf
        for child in successors(node):
            value = max(value, alphabeta(child, depth-1, α, β, False))
            α = max(α, value)
            if β <= α:
                break  # β剪枝
        return value
    else:
        value = +inf
        for child in successors(node):
            value = min(value, alphabeta(child, depth-1, α, β, True))
            β = min(β, value)
            if β <= α:
                break  # α剪枝
        return value
```

### ✅ 优点：

- 与 Minimax 相同的最优解
    
- 更少的节点展开，时间效率更高（尤其在树剪得深时）

### 补充：穿插Expectimax 搜索的剪枝

当传递到`chance`节点的时候

- If 下层节点是 `MAX` 节点：
	
	- α（Alpha）= max(α, 下层的α值)

- If 下层节点是 `MIN`节点：
- 
	- β（Beta）= min(β, 下层的β值)



## 资源限制(Resources Limited)

### 深度限制搜索（Depth-Limited Search）

- 在搜索到一定深度后停止，**不再**搜索到叶节点，而是通过预设的**深度限制**来结束搜索。
  
- **为什么要限制搜索深度？**
    - 现实中的游戏（如象棋）可能会有庞大的游戏树，包含数百万甚至数十亿个节点。要在合理的时间内完成计算，无法对所有节点进行遍历。
  
- 当搜索在深度限制时停止，我们需要使用**评估函数**来替代终端节点的效用值（即最终结果）。
  
- 评估函数(Evaluation Function)通过计算一些特征来估算当前局面的优劣，比如棋局中的材料差、位置安全性等。


### 评估函数(Evaluation Function)
#### 蒙特卡洛树搜索（MCTS）

- **蒙特卡洛树搜索（MCTS）** 已经成为一种流行的游戏AI方法。MCTS的基本思想是：
    
    1. 随机模拟走子直到游戏结束。
        
    2. 重复这个过程多次，根据这些模拟的结果来评估当前局面的价值。
        
    - **胜率**：MCTS的评估函数基于这些模拟的**胜率**来估算当前局面的优劣。这种方法非常有效，因为它不依赖于固定的评估函数，而是基于大量的模拟结果。


## 分支因子（Branching Factor）

**分支因子**指的是每个状态下的平均可行走子数量。它是决定游戏树复杂度的一个重要因素。

- 较高的分支因子意味着搜索树的增长速度较快，这使得在时间限制内进行深度搜索变得更为困难。

- 分支因子越大，游戏树的复杂度越高，即使使用像**α-β剪枝**这样的高效剪枝技术，也更难在有限的时间内进行深度搜索。

- 为了应对这一挑战，深度限制搜索和评估函数通常会聚焦于最有可能的棋步，从而使得AI可以在实际的时间限制内作出合理决策。（限制分支因子大小）