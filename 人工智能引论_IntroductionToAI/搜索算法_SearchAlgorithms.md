


# 搜索算法_SearchAlgorithms

本章介绍常用的一些搜索算法，内容来自ShanghaiTech CS181 / UCB CS188 / Stuart J.Russell AIMA

---

## Agents classification
- Reflex agents
	Do not consider the future consequences of their actions

- Planning agents
	Decisions based on (hypothesized) consequences of actions

## Search Problems
### Components 
- A state space
- A successor function
- A start state and a goal test
### State Space Graphs

### Search Trees

### Tree Search 树搜索

#### 核心概念

- **Fringe（边缘）**  
  存储待扩展的节点集合。具体实现依赖于搜索策略：  
  - 在 BFS（广度优先搜索）中为 FIFO 队列  
  - 在 DFS（深度优先搜索）中为 LIFO 栈

- **Expansion（扩展）**  
  从 fringe 中取出一个节点，并生成其子节点。

- **Exploration Strategy（探索策略）**  
  决定搜索算法的关键因素：以何种顺序扩展节点。

---

#### 搜索算法的性质（Search Algorithm Properties）

| 属性                   | 说明                  |
| -------------------- | ------------------- |
| **Complete**         | 完备性，是否保证能找到解（如果解存在） |
| **Optimal**          | 是否保证找到代价最小的路径（最优解）  |
| **Time Complexity**  | 算法的时间复杂度            |
| **Space Complexity** | 算法的空间复杂度            |

---

#### 常见搜索算法分析

##### 1. DFS（深度优先搜索）

- **Complete**: 否（若存在无限路径则可能陷入死循环），是（除非限制cycles）
- **Optimal**: 否  
- **Time Complexity**: $O(b^m)$（b 为分支因子，m 为最大深度）  
- **Space Complexity**: $O(bm)$（线性空间）  



---

##### 2. BFS（广度优先搜索）

- **Complete**: 是  
- **Optimal**: 是（仅当路径代价一致时）  
- **Time Complexity**: $O(b^d)$（d 为目标深度）  
- **Space Complexity**: $O(b^d)$（需要存储整层节点）  


---

##### 3. Iterative Deepening（迭代加深）

- **Complete**: 是  
- **Optimal**: 是（与 BFS 同）  
- **Time Complexity**: $O(b^d)$
- **Space Complexity**: $O(bd)$  

结合了 DFS 的空间效率和 BFS 的完备性与最优性。

1. ​**​初始化深度限制​**​：设初始深度 `d = 1`。
2. ​**​执行深度受限搜索（DLS）​**​：
    - 在深度 `d` 内进行DFS，若找到解则返回。
    - 若未找到解，则 `d += 1`，重新搜索。
3. ​**​重复步骤2​**​，直到解被发现或达到最大可行深度。

---

##### 4. Uniform Cost Search（UCS，均价搜索）

- **Complete**: 是  
- **Optimal**: 是（路径代价最小）  
- **Time Complexity**: $O(b^{C^*/ε})$  
- **Space Complexity**: 与时间复杂度相同，因需存储 fringe  

适用于非一致代价图中的最优路径搜索。

---

##### 5. Greedy Best-First Search（贪婪搜索）

- **Complete**: 否（可能陷入局部最优）  
- **Optimal**: 否  
- **Time Complexity**: $O(b^m)$  
- **Space Complexity**: $O(b^m)$  


---

##### 6. A* Search

- **Complete**: 是（若使用一致性启发式）  
- **Optimal**: 是（若启发函数为 admissible）  
- **Time Complexity**: 指数级，但优于 BFS/UCS  
- **Space Complexity**: 高，因需存储所有路径  

A\* 同时考虑路径代价  $g(n)$  和启发代价  $h(n)$ ，即：
```
f(n) = g(n) + h(n)
```

###### 启发式搜索与 Heuristic 设计

- Admissible Heuristic（可接受的启发函数）

	- 永远不高估从当前节点到目标的真实最小代价
	
	- 可确保 A* 在树搜索中是最优的
    

- Consistent Heuristic（一致性启发函数）

- 满足三角不等式：  
    $h(n)≤c(n,n′)+h(n′)$
    
- 可确保 A* 在图搜索中是最优的
    

---

### Graph Search 

在图搜索中，需要避免重复访问已访问节点，否则会影响完备性和最优性。

- A* 在图搜索中要使用一致性启发式才能保持最优性
    
- UCS 是 A* 的特例（当 $h(n)=0$）
    

---

### A* 总结

- A* 综合使用：
    
    - g(n)：从起点到当前节点的路径代价（后向代价）
        
    - h(n)：从当前节点到目标的估计代价（前向代价）
        
- 最优性条件：
    
    - **树搜索**：启发式函数需为 admissible
        
    - **图搜索**：启发式函数需为 consistent
        