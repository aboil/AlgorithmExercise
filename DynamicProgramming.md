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

**描述：**给定不同面额的硬币 `coins` 和一个总金额 `amount`。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 `-1`。

解：代码第19行：min函数里的第一个参数：dp[i-arr[j]] +1  ，i是当前的钱数（如 i=5），arr[j]是当前面值（如 arr[j]=2），则 找5元零钱相当于当前2元面值的一个和再找5-2=3元零钱的数量。

```java
public class Solution {
    /**
     * 最少货币数
     * @param arr int整型一维数组 the array
     * @param aim int整型 the target
     * @return int整型
     */
    public int minMoney (int[] arr, int aim) {
        // write code here
        // dp[i]的意思是当目标金额是 i 时，至少需要 dp[i] 枚硬币凑出
        int[] dp = new int[aim +1];
        Arrays.fill(dp, aim + 1);    // 填充count数组中的每个元素都是 aim + 1
        //初始化数组
        dp[0] = 0;
        for (int i =1; i<=aim; i++){
            for (int j = 0; j < arr.length; j++){
                if (i >=arr[j]){
                    //当前的钱数-当前面值，为之前换过的钱数，如果能够兑换只需要再加+1即可，如果不能就取aim+1;
                    dp[i] = Math.min(1+dp[i-arr[j]], dp[i]);
                }
            }
        }
        //对应的总数是否能够兑换取决于是否等于aim+1
        return dp[aim] != aim+1 ? dp[aim] :-1;
    }
}
```



待续。。。



