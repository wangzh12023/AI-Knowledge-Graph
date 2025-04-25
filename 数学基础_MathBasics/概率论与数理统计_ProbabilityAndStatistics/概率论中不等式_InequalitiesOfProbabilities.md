
>[!note] 
>Reference to ShanghaiTech SI140 L7

### Cauchy-Schwarz Inequality

$|E(X Y)| \leq \sqrt{E\left(X^2\right) E\left(Y^2\right)}$


### Jensen’s Inequality

> Before we  know:

If $f$ is a convex function, $0 \leq \lambda_1, \lambda_2 \leq 1, \lambda_1+\lambda_2=1$, then for any $x_1, x_2$,
$$
f\left(\lambda_1 x_1+\lambda_2 x_2\right) \leq \lambda_1 f\left(x_1\right)+\lambda_2 f\left(x_2\right)
$$

>[!note] Jensen’s Inequality Theorem
>
Let $X$ be a random variable. 
If $g$ is a convex function, then $E(g({X})) \geq g(E(X))$. 
If $g$ is a concave function, then $E({g(X)}) \leq {g(E(X))}$. 
In both cases, the only way that equality can hold is if there are constants $a$ and $b$ such that $g(X)=a+b X$ with probability 1.



### Markov’s Inequality
For any r.v. $X$ and constant $a>0$,
$$
P(|X| \geq a) \leq \frac{E|X|}{a}
$$

>[!todo] 
>太多了，又基本用不到，懒得写了，想了解自己看ppt

