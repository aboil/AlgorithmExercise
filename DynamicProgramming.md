# 动态规划

### 1、斐波那契数列

写一个函数，输入 `n` ，求斐波那契（Fibonacci）数列的第 `n` 项（即 `F(N)`）。斐波那契数列的定义如下：

> F(0) = 0,   F(1) = 1
> F(N) = F(N - 1) + F(N - 2), 其中 N > 1.

斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

```java
public class Solution {
    public int Fibonacci(int n) {
        if(n == 0) return 0;
        int pre = 0;
        int cur = 1;
        for(int i=1; i<n; i++){
            int sum = cur + pre;
            pre = cur;
            cur = sum;
        }
        return cur;
    }
}
```

---

### 2、零钱兑换

给定不同面额的硬币 `coins` 和一个总金额 `amount`。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 `-1`。







