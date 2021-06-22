# 牛客高频200

### N_78 反转链表

**描述：**输入一个链表，反转链表后，输出新链表的表头。

```java
public class Solution {
    public ListNode ReverseList(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }
//         ListNode p = head;
//         ListNode q = head.next;
//         p.next = null;
//         ListNode temp = null;
//         while(q != null){
//             temp = q.next;
//             q.next = p;
//             p = q;
//             q = temp;
//         }
//         return p;
//         1、迭代法
//         ListNode pre = head;
//         ListNode curr = head.next;
//         pre.next = null;
//         while(curr != null){
//             ListNode temp = curr.next;
//             curr.next = pre;
//             pre = curr;
//             curr = temp;
//         }
//         return pre;
        // 2、递归法
        ListNode newHead = ReverseList(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
}
```



### N_140 快速排序

**描述：**给定一个数组，请你编写一个函数，返回该数组排序后的形式。

```java
public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 将给定数组排序
     * @param arr int整型一维数组 待排序的数组
     * @return int整型一维数组
     */
    public int[] MySort (int[] arr) {
        // write code here
        quick(arr, 0, arr.length-1);
        return arr;
    }
    public void quick(int[] list, int left, int right){
        if(left<right){
            int point = partition(list, left, right);
            quick(list, left, point-1);
            quick(list, point+1, right);
        }
    }
    private int partition(int[] list, int left, int right){
        int temp = list[right];
        int point = left-1;
        for(int i=left; i<right; i++){
            if(list[i]<temp){
                point ++;
                swap(list, point, i);
            }
        }
        swap(list, point+1, right);
        return point + 1;
    }
    private void swap(int[] list, int i, int j){
        int temp = list[i];
        list[i] = list[j];
        list[j] = temp;
    }
}
```



---

### N_93 设计缓存结构

**描述：**设计LRU缓存结构，该结构在构造时确定大小，假设大小为K，并有如下两个功能

- set(key, value)：将记录(key, value)插入该结构
- get(key)：返回key对应的value值

[要求]

1. set和get方法的时间复杂度为O(1)
2. 某个key的set或get操作一旦发生，认为这个key的记录成了最常使用的。
3. 当缓存的大小超过K时，移除最不经常使用的记录，即set或get最久远的。

若opt=1，接下来两个整数x, y，表示set(x, y)
若opt=2，接下来一个整数x，表示get(x)，若x未出现过或已被移除，则返回-1
对于每个操作2，输出一个答案

```java
public class Solution {
    /**
     * 对应力扣：P146题
     * lru design
     * @param operators int整型二维数组 the ops
     * @param k int整型 the k
     * @return int整型一维数组
     */
    class DLinkedNode{
        int key;
        int value;
        DLinkedNode pre, next;
        DLinkedNode(){}
        DLinkedNode(int _key, int _value){key=_key; value=_value;}
    }
    private int k;
    private int size;
    private DLinkedNode head = new DLinkedNode(-1, -1);
    private DLinkedNode tail = new DLinkedNode(-1, -1);
    
    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    
    public int[] LRU (int[][] operators, int k) {
        // write code here
        this.size = 0;
        this.k = k;
        head.next = tail;
        tail.pre = head;
        int len = (int)Arrays.stream(operators).filter(x -> x[0]==2).count();
        int[] res = new int[len];
        for(int i=0, j=0; i<operators.length; i++){
            if(operators[i][0]==1){
                set(operators[i][1], operators[i][2]);
            }else{
                res[j++] = get(operators[i][1]);
            }
        }
        return res;
    }
    private void set(int key, int value){
        DLinkedNode node = cache.get(key);
        if(node==null){
            DLinkedNode newNode = new DLinkedNode(key, value);
            cache.put(key,newNode);
            addToHead(newNode);
            size++;
            if(size>k){
                DLinkedNode tailNode = removeTail();
                cache.remove(tailNode.key);
                size--;
            }
        }else{
            node.value = value;
            moveToHead(node);
        }
    }
    private int get(int key){
        DLinkedNode node = cache.get(key);
        if(node==null){
            return -1;
        }
        moveToHead(node);
        return node.value;
    }
    private void addToHead(DLinkedNode node){
        node.next = head.next;
        node.pre = head;
        head.next.pre = node;
        head.next = node;
    }
    private void removeNode(DLinkedNode node){
        node.next.pre = node.pre;
        node.pre.next = node.next;
    }
    private void moveToHead(DLinkedNode node){
        // 顺序不能反！！！！！
        removeNode(node);
        addToHead(node);
    }
    private DLinkedNode removeTail(){
        DLinkedNode node = tail.pre;
        removeNode(node);
        return node;
    }
}
```



---

### N_45 实现二叉树先序、中序、后序遍历

**描述：**分别按照二叉树先序，中序和后序打印所有的节点。

```java
public class Solution {
    /**
     * 
     * @param root TreeNode类 the root of binary tree
     * @return int整型二维数组
     */
    
    public int[][] threeOrders (TreeNode root) {
        // write code here
        List<Integer> pre = new ArrayList<Integer>();
        List<Integer> mid = new ArrayList<Integer>();
        List<Integer> last = new ArrayList<Integer>();
        preorder(root, pre);
        midorder(root, mid);
        lastorder(root, last);
        int[][] orders = new int[3][pre.size()];
        orders[0] = toIntArray(pre);
        orders[1] = toIntArray(mid);
        orders[2] = toIntArray(last);
        return orders;
        
    }
    private int[] toIntArray(List<Integer> list){
        int[] temp = new int[list.size()];
        for(int i=0; i<list.size(); i++){
            temp[i] = list.get(i);
        }
        return temp;
    }
    private void preorder(TreeNode root, List<Integer> pre){
        if(root==null){
            return;
        }
        pre.add(root.val);
        preorder(root.left, pre);
        preorder(root.right, pre);
    }
    private void midorder(TreeNode root, List<Integer> mid){
        if(root==null){
            return;
        }
        midorder(root.left, mid);
        mid.add(root.val);
        midorder(root.right, mid);
    }
    private void lastorder(TreeNode root, List<Integer> last){
        if(root==null){
            return;
        }
        lastorder(root.left, last);
        lastorder(root.right, last);
        last.add(root.val);
    }
}
```



---

### N_119 最小的K个数

**描述：**给定一个数组，找出其中最小的K个数。例如数组元素是4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。如果K>数组的长度，那么返回一个空的数组

```java
public class Solution {
    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
//         先进行快排
        ArrayList<Integer> res = new ArrayList<>();
        if(k<=0 || k>input.length){
            return res;
        }
        
        quick_sort(input, 0, input.length-1, k);
        
        for(int i=0; i<k; i++){
            res.add(input[i]);
        }
        return res;
        
    }
    private void quick_sort(int[] list, int left, int right, int k){
        if(left <= right){
            int position = pertation(list, left, right);
            if(k <= position+1){
                quick_sort(list, left, position-1, k);
            }else{
                quick_sort(list, position+1, right, k);
            }
        }
    }
    
    private int pertation(int[] list, int left, int right){
        int base = list[right];
        int position = left - 1;
        for(int i=left; i<right; i++){
            if(list[i]<base){
                position ++;
                swap(list, position, i);
            }
        }
        swap(list, position+1, right);
        return position+1;
    }
    
    private void swap(int[] list, int i, int j){
        int temp = list[i];
        list[i] = list[j];
        list[j] = temp;
    }
}
```



---

### N_15 二叉树的层序遍历

**描述：**给定一个二叉树，返回该二叉树层序遍历的结果，（从左到右，一层一层地遍历）

```java
public class Solution {
    /**
     * 
     * @param root TreeNode类 
     * @return int整型ArrayList<ArrayList<>>
     */
    public ArrayList<ArrayList<Integer>> levelOrder (TreeNode root) {
        // write code here
        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
        if(root==null){
            return res;
        }
        
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        while(!queue.isEmpty()){
            int currentLevelSize = queue.size();
            ArrayList<Integer> level = new ArrayList<Integer>();
            for(int i=0; i<currentLevelSize; i++){
                TreeNode node = queue.poll();
                level.add(node.val);
                if(node.left != null){
                    queue.offer(node.left);
                }
                if(node.right != null){
                    queue.offer(node.right);
                }
            }
            res.add(level);
        }
        return res;
    }
}
```



---

### N_88 寻找第K大数

**描述：**有一个整数数组，请你根据快速排序的思路，找出数组中第K大的数。

给定一个整数数组a,同时给定它的大小n和要找的K(K在1到n之间)，请返回第K大的数，保证答案存在。

```java
public class Solution {
    public int findKth(int[] a, int n, int K) {
        // write code here
        int res = quick_sort(a, 0, n-1, K);
        return res;
    }
    private int quick_sort(int[] list, int left, int right, int K){
        if(left<=right){
            int position = pratition(list, left, right);
            if(K == position+1){
                return list[position];
            }else if(K < position+1){
                return quick_sort(list, left, position-1, K);
            }
            else{
                return quick_sort(list, position+1, right, K);
            }
        }
        return -1;
    }
    private int pratition(int[] list, int left, int right){
        int base = list[right];
        int pos = left - 1;
        for(int i=left; i<right; i++){
            if(base<list[i]){
                pos ++;
                swap(list, pos, i);
            }
        }
        swap(list, pos+1, right);
        return pos+1;
    }
    private void swap(int[] list, int i, int j){
        int temp = list[i];
        list[i] = list[j];
        list[j] = temp;
    }
}
```



---

### N_61 两数之和

**描述：**给出一个整数数组，请在数组中找出两个加起来等于目标值的数，

你给出的函数 `twoSum` 需要返回这两个数字的下标（`index1`，`index2`），需要满足 `index1` 小于 `index2`。注意：下标是从1开始的

假设给出的数组中只存在唯一解

例如：

给出的数组为 `{20, 70, 110, 150}`,目标值为`90` 
输出 `index1=1, index2=2`

```java
public class Solution {
    /**
     * 
     * @param numbers int整型一维数组 
     * @param target int整型 
     * @return int整型一维数组
     */
    public int[] twoSum (int[] numbers, int target) {
        // write code here
//         int[] res = {0, 0};
//         HashMap<Integer, Integer> mp = new HashMap<Integer, Integer>();
//         for(int i=0; i<numbers.length; i++){
//             mp.put(numbers[i], i);
//         }
        
//         for(int i = 0; i<numbers.length; i++){
//             if(mp.containsKey(target - numbers[i]) && i!=mp.get(target-numbers[i])){
//                 res[0] = i+1;
//                 res[1] = mp.get(target - numbers[i])+1;
//                 return res;
//             }
//         }
//         return res;
        
        HashMap<Integer, Integer> mp = new HashMap<Integer, Integer>();
        for(int cur=0,temp; cur<numbers.length; cur++){
            if(mp.containsKey(target-numbers[cur])){
                return new int[] {mp.get(target-numbers[cur])+1, cur+1};
            }
            mp.put(numbers[cur], cur);
        }
        throw new RuntimeException("results not exits");
    }
}
```



---

### N_33 合并有序链表

**描述：**将两个有序的链表合并为一个新链表，要求新的链表是通过拼接两个链表的节点来生成的，且合并后新链表依然有序。

```java
public class Solution {
    /**
     * 
     * @param l1 ListNode类 
     * @param l2 ListNode类 
     * @return ListNode类
     */
    public ListNode mergeTwoLists (ListNode l1, ListNode l2) {
        // write code here
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        ListNode head = new ListNode(0);
        ListNode temp = head;
        
        while(l1 != null && l2 != null){
            if(l1.val <= l2.val){
                temp.next = l1;
                l1 = l1.next;
            }else{
                temp.next = l2;
                l2 = l2.next;
            }
            temp = temp.next;
        }
        temp.next = (l1==null) ? l2 : l1;
        return head.next;
    }
}
```



---

### N_76 用两个栈实现队列

**描述：**用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

```java
public class Solution {
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    
    public void push(int node) {
        stack1.push(node);
    }
    
    public int pop() {
        if(stack2.size() <= 0){
            while(stack1.size() != 0){
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }
}
```



---

### N_68 跳台阶

描述：一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

```java
public class Solution {
    public int jumpFloor(int target) {
        // 法1：自上向底型递归求解
//         if (target == 1) return 1;
//         if (target == 2) return 2;
//         return jumpFloor(target - 1) + jumpFloor(target - 2);
        
        // 法2：自底向上型循环求解
        int a = 1, b = 1;
        for (int i = 2; i <= target; i++) {
            // 求f[i] = f[i - 1] + f[i - 2]
            a = a + b; // 这里求得的 f[i] 可以用于下次循环求 f[i+1]
            // f[i - 1] = f[i] - f[i - 2]
            b = a - b; // 这里求得的 f[i-1] 可以用于下次循环求 f[i+1]
        }
        return a;
    }
}
```



---

### N_50 链表中的节点每k个一组翻转

**描述：**将给出的链表中的节点每 *k* 个一组翻转，返回翻转后的链表
如果链表中的节点数不是 *k* 的倍数，将最后剩下的节点保持原样
你不能更改节点中的值，只能更改节点本身。
要求空间复杂度 *O(1)* 

例如：

给定的链表是1→2→3→4→5

对于 *k*=2, 你应该返回 2→1→4→3→5

对于 *k*=3, 你应该返回 3→2→1→4→5

```java
public class Solution {
    /**
     * 
     * @param head ListNode类 
     * @param k int整型 
     * @return ListNode类
     */
    public ListNode reverseKGroup (ListNode head, int k) {
        // write code here
        ListNode hair = new ListNode(-1);
        hair.next = head;
        ListNode pre = hair;
        
        while(head != null){
            ListNode tail = pre;
            for(int i=0; i<k; i++){
                tail = tail.next;
                if(tail == null){
                    return hair.next;
                }
            }
            
            ListNode nex = tail.next;
            ListNode[] reverse = myReverse(head, tail);
            head = reverse[0];
            tail = reverse[1];
            
            pre.next = head;
            tail.next = nex;
            pre = tail;
            head = tail.next;
        }
        return hair.next;
    }
    
    public ListNode[] myReverse(ListNode head, ListNode tail){
        ListNode pre = tail.next;
        ListNode p = head;
        while(pre != tail){
            ListNode nex = p.next;
            p.next = pre;
            pre = p;
            p = nex;
        }
        return new ListNode[]{tail, head};
    }
}
```



---

### N_19 子数组的最大累加和问题

描述：给定一个数组`arr`，返回`子数组`的**最大累加和**

例如，`arr = [1, -2, 3, 5, -2, 6, -1]`，所有子数组中，`[3, 5, -2, 6]`可以累加出最大的和`12`，所以返回`12`.

题目保证没有全为负数的数据

[要求]

时间复杂度为*O*(*n*)，空间复杂度为*O (1)* 

```java
public class Solution {
    /**
     * max sum of the subarray
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int maxsumofSubarray (int[] arr) {
        // write code here
        int res = arr[0];
        for(int i=1; i<arr.length; i++){
            arr[i] += arr[i-1] > 0? arr[i-1]:0;
            res = res>arr[i]? res:arr[i];
        }
        return res;
    }
}
```



---

### N_41 最长无重复子串

描述：给定一个数组arr，返回arr的最长无的重复子串的长度(无重复指的是所有数字都不相同)。

```java
public class Solution {
    /**
     * 
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int maxLength (int[] arr) {
        // write code here
        HashMap<Integer, Integer> hm = new HashMap<>();
        int maxLen = 0;
        int start = 0;
        for(int i=0; i<arr.length; i++){
            if(hm.containsKey(arr[i])){
                start = Math.max(start, hm.get(arr[i])+1);
            }
            maxLen = Math.max(maxLen, i-start+1);
            hm.put(arr[i], i);
        }
        return maxLen;
        
    }
}
```



---

### N_4 判断链表中是否有环

描述：判断给定的链表中是否有环。如果有环则返回`true`，否则返回`false`。

你能给出空间复杂度*O (1)* 的解法么？

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        if(head==null || head.next==null){
            return false;
        }
        ListNode slow = head, fast = head.next;
        while(slow != fast){
            if(fast == null || fast.next == null){
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }
}
```



---

### N_22 合并两个有序的数组

描述：给出两个有序的整数数组  **A** 和 **B**，请将数组 **B** 合并到数组 **A** 中，变成一个有序的数组。

```java
public class Solution {
    // 力扣：P88
    public void merge(int A[], int m, int B[], int n) {
        int i=m-1, j=n-1;
        for(int t=m+n-1; j>=0; t--){
            if(i<0){
                A[t] = B[j--];
            }
            else if(A[i]>B[j]){
                A[t] = A[i--];
            }else{
                A[t] = B[j--];
            }
        }
    }
}
```



---

### N_126 换钱的最少货币数

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

---

### NC_3 链表中环的入口节点

题目描述

对于一个给定的链表，返回环的入口节点，如果没有环，返回null

拓展：

你能给出不利用额外空间的解法么？

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode slow=head, fast=head;
        while(fast!=null && fast.next!=null){
            fast = fast.next.next;
            slow = slow.next;
            if(slow==fast){
                fast = head;
                while(slow!=fast){
                    slow = slow.next;
                    fast = fast.next;
                }
                return slow;
            }
        }
        return null;
    }
}
```

---

### NC_52 括号序列

**题目描述：**

给出一个**仅包含**字符'(',')','{','}','['和']',的字符串，判断给出的字符串是否是合法的括号序列
括号必须以正确的顺序关闭，"()"和"()[]{}"都是合法的括号序列，但"(]"和"([)]"不合法。

```java
public class Solution {
    /**
     * 
     * @param s string字符串 
     * @return bool布尔型
     */
    public boolean isValid (String s) {
        // write code here
        Stack<Character> stack = new Stack<Character>();
        for(int i=0; i<s.length(); i++){
            char i_char = s.charAt(i);
            if(stack.empty()){
                stack.push(i_char);
            }else if(stack.peek() == '(' && i_char == ')'){
                stack.pop();
            }else if(stack.peek() == '[' && i_char == ']'){
                stack.pop();
            }else if(stack.peek() == '{' && i_char == '}'){
                stack.pop();
            }else{
                stack.push(i_char);
            }
        }
        return stack.empty();
    }
}
```

---

### NC_53 删除列表的倒数第n个节点

描述：给定一个链表，删除链表的倒数第 n 个节点并返回链表的头指针
例如，

给出的链表为: 1→2→3→4→5, n= 2.
删除了链表的倒数第 n 个节点之后,链表变为 1→2→3→5.

备注：

题目保证 n 一定是有效的
请给出请给出时间复杂度为 O(n)  的算法

```java
public class Solution {
    /**
     * 
     * @param head ListNode类 
     * @param n int整型 
     * @return ListNode类
     */
    public ListNode removeNthFromEnd (ListNode head, int n) {
        // write code here
        int len = 0;
        ListNode p = new ListNode(-1);
        p.next = head;
        while(head!=null){
            head = head.next;
            len++;
        }
        head = p;
        while(true){
            if(len == n){
                if(head.next != null)
                    head.next = head.next.next;
                break;
            }
            head = head.next; 
            len -- ;
        }
            
        return p.next;
        
        
        
        /// 双指针怎么解！！！！
//         ListNode p = new ListNode(-1,head);
//         ListNode t = head;
//         for(int i=0; i<n; i++){
//             t = t.next;
//         }
//         head = p;
//         while(t!=null){
//             head = head.next;
//             t = t.next;
//         }
//         head.next = head.next.next;
//         return p.next;
    }
}
```

注：双指针思路：初始时，让快指针从第n个开始，慢指针从0开始，则当快指针到达最后时，慢指针就到了要被删除的节点位置，

---

### NC_1 大数加法

**描述：**以字符串的形式读入两个数字，编写一个函数计算它们的和，以字符串形式返回。

（字符串长度不大于100000，保证字符串仅由'0'~'9'这10种字符组成）

```java
public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算两个数之和
     * @param s string字符串 表示第一个整数
     * @param t string字符串 表示第二个整数
     * @return string字符串
     */
    public String solve (String s, String t) {
        // write code here
        StringBuilder res = new StringBuilder("");
        int i = s.length() - 1, j = t.length() - 1, carry = 0;
        while(i >= 0 || j >= 0){
            int n1 = i >= 0 ? s.charAt(i) - '0' : 0;
            int n2 = j >= 0 ? t.charAt(j) - '0' : 0;
            int tmp = n1 + n2 + carry;
            carry = tmp / 10;
            res.append(tmp % 10);
            i--; j--;
        }
        if(carry == 1) res.append(1);
        return res.reverse().toString();
    }
}
```

---

### NC_14 二叉树的之字形层序遍历

描述：题目描述

给定一个二叉树，返回该二叉树的之字形层序遍历，（第一层从左向右，下一层从右向左，一直这样交替）
例如：
给定的二叉树是{3,9,20,#,#,15,7},

![img](NowCoder_top200.assets/999991351_1596788654427_630E55F47DBAFBF72C88E265929E43F7)
该二叉树之字形层序遍历的结果是

> [
>
> [3],
>
> [20,9],
>
> [15,7]
>
> ]

```java
public class Solution {
    /**
     * 
     * @param root TreeNode类 
     * @return int整型ArrayList<ArrayList<>>
     */
    public ArrayList<ArrayList<Integer>> zigzagLevelOrder (TreeNode root) {
        // write code here
        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
        if(root==null){
            return res;
        }
        
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        int n_level = 1;
        while(!queue.isEmpty()){
            int currentLevelSize = queue.size();
            ArrayList<Integer> level = new ArrayList<Integer>();
            for(int i=0; i<currentLevelSize; i++){
                TreeNode node = queue.poll();
                if(n_level%2 == 0){
                    level.add(0, node.val);		// 重点在这，在第0位插入
                }else{
                    level.add(node.val);
                }
                
                if(node.left != null){
                    queue.offer(node.left);
                }
                if(node.right != null){
                    queue.offer(node.right);
                }
            }
            res.add(level);
            n_level ++;
        }
        return res;
    }
}
```

---

### NC_127 最长公共子串



**题目描述**

给定两个字符串str1和str2,输出两个字符串的最长公共子串

题目保证str1和str2的最长公共子串存在且唯一。

示例1

**输入**

```
"1AB2345CD","12345EF"
```

**返回值**

```
"2345"
```

代码参考：

```java
public class Solution {
    /**
     * longest common substring
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @return string字符串
     */
    public String LCS (String str1, String str2) {
        // write code here
//         int maxLength = 0;
//         int index = 0;
//         for(int i = 0; i < str2.length(); i++){
//             for(int j = i+1; j <= str2.length(); j++){
//                 if(str1.contains(str2.substring(i, j)) ){
//                     if(maxLength < j-i){
//                         maxLength = j-i;
//                         index = i;
//                     }
//                 } else break;
//             }
//         }
//         if( maxLength == 0 ){ //没有相同的字符串
//             return "-1";
//         }
//         return str2.substring(index,index + maxLength);
        
        
        // 法2、动态规划
        int[][] nums = new int[str1.length()+1][str2.length()+1];	// 里面的值被初始化为0
        if(str1 == null || str2 == null || str1.equals("") || str2.equals("")){
            return "-1";
        }
        int maxLength = 0;   //记录最长公共子串长度
        int end = 0;          //记录最长子串最后一个字符的下标
        int m = str1.length();
        int n = str2.length();

        // 初始化表格边界
		// for(int i = 0; i <= m; ++i) nums[i][0] = 0;
		// for(int j = 0; j <= n; ++j) nums[0][j] = 0;

        // 循环"填表"
        for (int i=1; i<=m; i++){
            for (int j=1; j<=n; j++){
                if (str1.charAt(i-1)==str2.charAt(j-1)){
                    nums[i][j]=nums[i-1][j-1]+1;
                }else {
                    nums[i][j]=0;
                }
                // 记录最长子串的长度和当前下标
                if (nums[i][j] >= maxLength){
                    maxLength = nums[i][j];
                    end = j-1;
                }
            }
        }
        // 如果没有公共子串
        if (maxLength == 0){
            return "-1";
        }else {
            return str2.substring(end-maxLength+1, end+1);
        }
    }
}
```







---

### N_66 两个链表的第一个公共结点

**题目描述：**输入两个无环的单链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）

```java
/*
public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}*/
public class Solution {
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if(pHead1 == null || pHead2 == null){
            return null;
        }
        ListNode p1 = pHead1, p2 = pHead2;
        while(p1 != p2){
            p1 = p1==null?pHead2:p1.next;
            p2 = p2==null?pHead1:p2.next;
        }
        return p1;
    }
}
```



---

### N_40 两个链表生成相加链表

**题目描述**：假设链表中每一个节点的值都在 0 - 9 之间，那么链表整体就可以代表一个整数。

给定两个这种链表，请生成代表两个整数相加值的结果链表。

例如：链表 1 为 9->3->7，链表 2 为 6->3，最后生成新的结果链表为 1->0->0->0。

```java
import java.util.*;

/*
 * public class ListNode {
 *   int val;
 *   ListNode next = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param head1 ListNode类 
     * @param head2 ListNode类 
     * @return ListNode类
     */
    
    public ListNode addInList (ListNode head1, ListNode head2) {
        // write code here
        if(head1 == null) return head2;
        if(head2 == null) return head1;
        Stack<Integer> s1 = new Stack<Integer>();
        Stack<Integer> s2 = new Stack<Integer>();
        while(head1!=null){
            s1.add(head1.val);
            head1 = head1.next;
        }
        while(head2!=null){
            s2.add(head2.val);
            head2 = head2.next;
        }
        ListNode head = new ListNode(-1);
        ListNode temp = head;
        int flag = 0;
        while(!s1.isEmpty() || !s2.isEmpty()){
            int v1 = 0, v2 = 0;
            if(!s1.isEmpty()){
                v1 = s1.pop();
            }
            if(!s2.isEmpty()){
                v2 = s2.pop();
            }
            temp.next = new ListNode((v1+v2 + flag)%10);
            flag = (int)(v1+v2 + flag)/10;
            temp = temp.next;
            
        }
        if(flag == 1)    temp.next = new ListNode(1);
        
        return reverse(head.next);
    }
    private ListNode reverse(ListNode head){
        ListNode p = null, temp = null;
        while(head!=null){
            temp = head.next;
            head.next = p;
            p = head;
            head = temp;
        }
        return p;
    }
}
```



---

### N_102 在二叉树中找到两个节点的最近公共祖先

**题目描述**：给定一棵二叉树(保证非空)以及这棵树上的两个节点对应的val值 o1 和 o2，请找到 o1 和 o2 的最近公共祖先节点。

注：本题保证二叉树中每个节点的val值均不相同。

```java
import java.util.*;

/*
 * public class TreeNode {
 *   int val = 0;
 *   TreeNode left = null;
 *   TreeNode right = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param root TreeNode类 
     * @param o1 int整型 
     * @param o2 int整型 
     * @return int整型
     */
    public int lowestCommonAncestor (TreeNode root, int o1, int o2) {
        // write code here
        return commonAncestor(root, o1, o2).val;
    }
    public TreeNode commonAncestor (TreeNode root, int o1, int o2) {
        if (root == null || root.val == o1 || root.val == o2) { // 超过叶子节点，或者root为p、q中的一个直接返回
            return root;
        }
        TreeNode left = commonAncestor(root.left,o1,o2); // 返回左侧的p\q节点
        TreeNode right = commonAncestor(root.right,o1,o2); // 返回右侧的p\q节点
        if (left == null) {  // 都在右侧
            return right;
        }
        if (right == null) { // 都在左侧
            return left;
        }
        return root; // 在左右两侧
    }
}
```



---

### N_103 反转字符串

**题目描述**：写出一个程序，接受一个字符串，然后输出该字符串反转后的字符串。（字符串长度不超过1000），如输入："abcd"。返回值："dcba"

```java
import java.util.*;


public class Solution {
    /**
     * 反转字符串
     * @param str string字符串 
     * @return string字符串
     */
    public String solve (String str) {
        // write code here
        char[] new_str = str.toCharArray();
        for(int i=str.length(); i>0; i--){
            new_str[str.length()-i] = str.charAt(i-1);
        }
        return new String(new_str);
    }
}
```



---

### N_38 螺旋矩阵

**题目描述**：给定一个m x n大小的矩阵（m行，n列），按螺旋的顺序返回矩阵中的所有元素。

示例1

```
输入：[[1,2,3],[4,5,6],[7,8,9]]
返回值：[1,2,3,6,9,8,7,4,5]
```

```java
import java.util.*;

public class Solution {
    public ArrayList<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> res = new ArrayList<>();
        if(matrix.length == 0)
            return res;
        int top = 0, bottom = matrix.length-1;
        int left = 0, right = matrix[0].length-1;
 
        while( top < (matrix.length+1)/2 && left < (matrix[0].length+1)/2 ){
            //上面  左到右
            for(int i = left; i <= right; i++){
                res.add(matrix[top][i]);
            }
 
            //右边 上到下
            for(int i = top+1; i <= bottom; i++){
                res.add(matrix[i][right]);
            }
 
            //下面  右到左
            for(int i = right-1; top!=bottom && i>=left; i--){
                res.add(matrix[bottom][i]);
            }
 
            //左边 下到上
            for(int i = bottom-1; left!=right && i>=top+1; i--){
                res.add(matrix[i][left]);
            }
            ++top;
            --bottom;
            ++left;
            --right;
        }
        return res;
    }
}
```



---

### N_65 斐波那契数列

**题目描述**：大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。

```java
public class Solution {
    public int Fibonacci(int n) {
        if(n==0) return 0;
        int i=1;
        int res = 1;
        int temp = 0;
        for(int j=3; j<=n; j++){
            temp = res;
            res += i;
            i = temp;
        }
        return res;
    }
}
```



---

### N_17 最长回文子串

**题目描述**：对于一个字符串，请设计一个高效算法，计算其中最长回文子串的长度。

给定字符串**A**以及它的长度**n**，请返回最长回文子串的长度。

示例1

输入：

```
输入："abc1234321ab",12
返回值：7
```

```java
import java.util.*;

public class Solution {
    public int getLongestPalindrome(String A, int n) {
        // write code here
//         int[][] res = new int[n+1][n+1];    // 里面的值默认都是0
//         int maxLen = 0, end = 0;
//         for(int i=1; i<=n; i++){
//             for(int j=1; j<=n; j++){
//                 if(A.charAt(i-1) == A.charAt(n-j)){
//                     res[i][j] = res[i-1][j-1]+1;
//                 }else{
//                     res[i][j]=0;
//                 }
//                 if(maxLen<res[i][j]){
//                     int beforeRev = n - j;
//                     // 考虑到abc12cba这种字符串，所以要进行下标的对应判断
//                     if(beforeRev + res[i][j] == i){    // 注意要进行判断，判断下标是否对应，否则像abc45cba这种字符串就会返回3
//                         maxLen = res[i][j];
//                         end = i;
//                     }
//                 }
//             }
//         }
//         return maxLen;
//         return A.substring(end-maxLen, end);
        
        
        
            //边界条件判断
            if (n < 2)
                return A.length();
            //maxLen表示最长回文串的长度
            int maxLen = 0;
            for (int i = 0; i < n; ) {
                //如果剩余子串长度小于目前查找到的最长回文子串的长度，直接终止循环
                // （因为即使他是回文子串，也不是最长的，所以直接终止循环，不再判断）
                if (n - i <= maxLen / 2)
                    break;
                int left = i;
                int right = i;
                while (right < n - 1 && A.charAt(right + 1) == A.charAt(right))
                    ++right; //过滤掉重复的

                //下次在判断的时候从重复的下一个字符开始判断
                i = right + 1;
                //然后往两边判断，找出回文子串的长度
                while (right < n - 1 && left > 0 && A.charAt(right + 1) == A.charAt(left - 1)) {
                    ++right;
                    --left;
                }
                //保留最长的
                if (right - left + 1 > maxLen) {
                    maxLen = right - left + 1;
                }
            }
            //截取回文子串
            return maxLen;
        }
}
```



---

### N_54 数组中相加和为0的三元组

**题目描述**：给出一个有n个元素的数组S，S中是否有元素a,b,c满足a+b+c=0？找出数组S中所有满足条件的三元组。

注意：三元组（a、b、c）中的元素必须按非降序排列。（即a≤b≤c）解集中不能包含重复的三元组。

示例

```
输入：[0]
返回值：[]

输入：[-2,0,1,1,2]
返回值：[[-2,0,2],[-2,1,1]]

输入：[-10,0,10,20,-10,-40]
返回值：[[-10,-10,20],[-10,0,10]]
```

```java
import java.util.*;

public class Solution {
    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        Arrays.sort(num);
        ArrayList<ArrayList<Integer>> list = new ArrayList<>();
        for (int i = 0; i < num.length - 2; ++i) {
            if (i != 0 && num[i - 1] == num[i]) continue;
            int k = num.length - 1;
            for (int j = i + 1; j < num.length - 1; ++j) {
                if (j != i + 1 && num[j] == num[j - 1]) continue;
                while (j < k && num[i] + num[j] + num[k] > 0) --k;
                if (j == k) break;
                if (num[i] + num[j] + num[k] == 0) {
                    ArrayList<Integer> ele = new ArrayList<>();
                    ele.add(num[i]);
                    ele.add(num[j]);
                    ele.add(num[k]);
                    list.add(ele);
                }
            }
        }
        return list;
    }
}
```



---

### N_12 重建二叉树

**题目描述**：

输入某二叉树的**前序遍历**和**中序遍历**的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

```java
/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
import java.util.*;

public class Solution {
    HashMap<Integer, Integer> memo = new HashMap<>();
    int[] preorder;
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        for(int i=0; i<in.length; i++){
            memo.put(in[i],i);
        }
        preorder = pre;
        TreeNode root = buildTree(0, in.length-1, 0, pre.length-1);
        return root;
    }
    
    public TreeNode buildTree(int is, int ie, int ps, int pe){
        if(ie<is || pe<ps){
            return null;
        }
        int root_value = preorder[ps];
        int ri = memo.get(root_value);
        
        TreeNode node = new TreeNode(root_value);
        node.left = buildTree(is, ri-1, ps+1, ps+ri-is);
        node.right = buildTree(ri+1, ie, ps+ri-is+1, pe);
        return node;
    }
}
```



---

### N_91 最长递增子序列

**题目描述**：给定数组arr，设长度为n，输出arr的最长递增子序列。（如果有多个答案，请输出其中 按数值(注：区别于按单个字符的ASCII码值)进行比较的 字典序最小的那个）

示例

```
输入：[2,1,5,3,6,4,8,9,7]
返回值：[1,3,4,8,9]

输入：[1,2,8,6,4]
返回值：[1,2,4]
其最长递增子序列有3个，（1，2，8）、（1，2，6）、（1，2，4）其中第三个 按数值进行比较的字典序 最小，故答案为（1，2，4） 
```

```java

```



---

### N_32 求平方根

**题目描述**：实现函数 int sqrt(int x)。计算并返回x的平方根（向下取整）

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param x int整型 
     * @return int整型
     */
    public int sqrt (int x) {
        // write code here
        for(int i=1; i<=x/2+1; i++){
            if((i+1)*(i+1)>x && i*i <= x){
                return i;
            }
        }
        return 0;
        
        // 二分法
//         if (x <= 0)
//             return x;
//         long left = 1;
//         long right = x;
//         while(left < right) {
//             long middle = (left + right) / 2;
//             if (middle * middle <= x && (middle + 1) * (middle + 1) > x) {
//                 return (int)middle;
//             } else if (middle * middle < x) {
//                 left = middle;
//             } else {
//                 right = middle;
//             }
//         }
//         return (int) left;
    }
}
```



---

### N_48 在旋转过的有序数组中寻找目标值

**题目描述**：给定一个整数数组nums，按升序排序，数组中的元素各不相同。

nums数组在传递给search函数之前，会在预先未知的某个下标 t（0 <= t <= nums.length-1）上进行旋转，让数组变为 [nums[t], nums[t+1], ..., nums[nums.length-1], nums[0], nums[1], ..., nums[t-1]]。

比如，数组 [0,2,4,6,8,10] 在下标 2 处旋转之后变为 [6,8,10,0,2,4]

现在给定一个旋转后的数组 nums 和一个整数 target ，请你查找这个数组是不是存在这个target，如果存在，那么返回它的下标，如果不存在，返回-1

示例

```
输入：[6,8,10,0,2,4],10
返回值：2

输入：[6,8,10,0,2,4],3
返回值：-1

```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param nums int整型一维数组 
     * @param target int整型 
     * @return int整型
     */
    public int search (int[] nums, int target) {
        // write code here
        int left = 0 , right = nums.length - 1;
        while(left <= right){
            int mid = left + ((right - left) / 2);//等价于 (right + left)/2
            if(nums[mid] == target) 
                return mid;
            // 右边有序
            if(nums[mid] < nums[right]){
                if(target > nums[mid] && target <= nums[right]){
                    left = mid + 1;
                }else{
                    right = mid - 1;
                }
            }else{
                if(target >= nums[left] && target < nums[mid]){
                    
                    right = mid - 1;
                }else{
                    left = mid + 1;
                }
            }
        }
        return -1;
    }
}
```



---

### N_90 设计getMin功能的栈

**题目描述**：实现一个特殊功能的栈，在实现栈的基本功能的基础上，再实现返回栈中最小元素的操作。

示例1

```
输入：[[1,3],[1,2],[1,1],[3],[2],[3]]
返回值：[1,2]
```

备注：有三种操作种类，op1表示push，op2表示pop，op3表示getMin。你需要返回和op3出现次数一样多的数组，表示每次getMin的答案

```java
import java.util.*;


public class Solution {
    /**
     * return a array which include all ans for op3
     * @param op int整型二维数组 operator
     * @return int整型一维数组
     */
    Stack<Integer> stack1 = new Stack<>();
    Stack<Integer> stack2 = new Stack<>();
    ArrayList<Integer> temp = new ArrayList<>();
    public int[] getMinStack (int[][] op) {
        // write code here
        for(int i=0; i<op.length; i++){
            if(op[i][0]==1){
                push(op[i][1]);
            }else if(op[i][0]==2){
                pop();
            }else{
                temp.add(getMin());
            }
        }
        int[] res = new int[temp.size()];
        for(int i=0; i<temp.size(); i++){
            res[i] = temp.get(i);
        }
        return res;
    }
    
    public void push(int value){
        stack1.push(value);
        if(stack2.isEmpty() || stack2.peek()>=value){//条件是大于等于
            stack2.push(value);
        }
    }
    
    public int pop(){
        if(stack1.peek().equals(stack2.peek())){
            stack2.pop();
        }
        return stack1.pop();
    }
    
    public int getMin(){
        return stack2.peek();
    }
}
```



---

### N_7 股票（一次交易）

更难的 [股票（两次交易）]()

**题目描述**：假设你有一个数组，其中第 *i* 个元素是股票在第 *i* 天的价格。
你有一次买入和卖出的机会。（只有买入了股票以后才能卖出）。请你设计一个算法来计算可以获得的最大收益。

**示例：**

```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```

```java
import java.util.*;

public class Solution {
    /**
     * 
     * @param prices int整型一维数组 
     * @return int整型
     */
    public int maxProfit (int[] prices) {
        // write code here
        int cost = Integer.MAX_VALUE, profit = 0;
        for(int price : prices) {
            // 前i日最大利润=max(前(i−1)日最大利润, 第i日价格−前i日最低价格)
            cost = Math.min(cost, price);//前i日最低价格
            profit = Math.max(profit, price - cost);
        }
        return profit;
    }
}
```



### N_51 合并k个有序链表

**题目描述**：合并 *k* 个已排序的链表并将其作为一个已排序的链表返回。分析并描述其复杂度。

示例：

```
输入：  [{1,2,6},{},{3,5,6,7}]
返回值：{1,2,3,4,5,6,7}
```

参考代码，自己写的

```java
import java.util.*;
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode mergeKLists(ArrayList<ListNode> lists) {

        Queue<ListNode>  queue = new LinkedList<ListNode>();
        int j = 0;
        ListNode node = null;    //用于保存剩余节点中，最小的节点
        for(; j<lists.size(); j++){
            if(lists.get(j) != null){
                node = lists.get(j);
                break;
            }
        }
        if(node == null){
            return null;
        }
        
        ListNode root = new ListNode(-1);
        ListNode p = root;
        ListNode temp;
        for(int i =j+1; i<lists.size(); i++){
            temp = lists.get(i);
            if(temp == null){
                continue;
            }
            if(node.val > temp.val){
                queue.offer(node);
                node = temp;
            }else{
                queue.offer(temp);
            }
        }
        if(node.next != null){
            queue.offer(node.next);
        }
        
        p.next = node;
        p = p.next;
        
        while(!queue.isEmpty()){
            int n = queue.size()-1;
            node = queue.poll();
            for(int i=0; i<n; i++){
                temp = queue.poll();
                if(node.val > temp.val){
                    queue.offer(node);
                    node = temp;
                }else{
                    queue.offer(temp);
                }
            }
            if(node.next!=null){
                queue.offer(node.next);
            }
            p.next = node;
            p = p.next;
        }
        return root.next;
    }
}
```

### N_121 字符串的排列







### N_128 容器盛水问题