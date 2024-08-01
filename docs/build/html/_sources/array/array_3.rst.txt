Array Questions Part 3
======================
56. (LC 1) Two Sum
--------------------
`1. Two Sum <https://leetcode.com/problems/two-sum/submissions/1340375300/>`_
Easy

**Python3**

::

    # My V (LC accepted 99, 50)
    class Solution:
        def twoSum(self, nums: List[int], target: int) -> List[int]:
            hash_t = {}
            for i, n in enumerate(nums):
                pair = target - n
                if pair in hash_t:
                    return [hash_t[pair], i]
                hash_t[n] = i

**C++** [:ref:`14 <ref-label>`]

| nums={3,2,10,11,7,15}, target = 9
| **The naive approach**
| Iterate nums, for each number iterate the rest of nums and look for the pair_value=target-num.
| E.g. 3, 9-3=6, look for 6 to the right of 3.
| 2, 9-2=7, look for 7 to the right of 2.

.. code-block:: cpp

    class Solution {
    public:
        vector<int> twoSum(vector<int>& nums, int target) {
            vector<int> result;  //for two answer indices
            for (auto it1 = nums.begin(); it1 != nums.end(); ++it1){
                auto it2 = find(it1+1, nums.end(), target - *it1); //it1+1 to look to the right of cur num only
                if (it2 != nums.end()){  //found the number
                    result.push_back(it1 - nums.begin());  //iterators subtraction allows to obtain a proper number of an index, not iterator
                    result.push_back(it2 - nums.begin());
                    break;   //cannot use return here, otherwise error: non-void function does not return a value in all control paths 
                }
            }
            return {}; //returning empty vector if complement was not found
        }
    };

| **Approach 2**
| Using hash map.
| keys: nums values, values: their position
| Insert all elems into the map. (The catch is not to insert all values, then traverse 
| again and search, but do it simultaneously.)

.. code-block:: cpp

    class Solution {
    public:
        vector<int> twoSum(vector<int>& nums, int target) {
            unordered_map<int, int> _map;
            for (int i = 0; i < nums.size(); ++i){
                int num = nums[i];
                int complement = target - num;
                auto it = _map.find(complement); //returns iter if finds key in map, end of container iter otherwise. Iter points to key, value pair.
                if(it != _map.end()){ //found
                    return {it->second, i};  //iter points at the (key,value) pair, *iter is tha pair. (*iter).mem, we want the second, i.e. value. i is the index of cur num
                }
                _map[num] = i;  //insert into hash map, like in Python
            }
            return {}; //returning empty vector if complement was not found
        }
    };

| Note how we iterate:
| Approach 1 -> vector -> iterators
| Approach 1 -> unordered_map -> subscripting

57. (LC 15) 3Sum
-------------------
`15. 3Sum <https://leetcode.com/problems/3sum/description/>`_
*(Medium)*

**Solution 2 (Sorting + pointers)**
Internally uses Two Sum II - Input Array Is Sorted.

Here we are not given a sorted input. We sort before proceeding to the algorithm.
(LC Topics hints that we should use sorting + two pointers.)

| Complexity: 
| Time: O(n log n) + O(n^2) = O(n^2)
| (sorting + we still use nested loops)
| Space: O(1) or O(n) depending on the implementation of sorting (language internals).
 
Keys::

    # Visualization
    # [-1,0,1,2,-1,-4] sort ->
    # [-4,-1,-1,0,1,2] use pointers ->
    # [-4,-1,-1,0,1,2] 
    # cur  L        R

**My V3** (Solution2, dropping extra optimizations, the core algorithm. 
Just for understanding. Not efficient enough for Leetcode.) ::

    def f(a):
        if len(a) < 3:
            return []
        a.sort()
        ans = []
        for i, v in enumerate(a):
            lp = i + 1
            rp = len(a) - 1
            while lp < rp:
                s = v + a[lp] + a[rp]
                if s > 0:
                    rp -= 1
                elif s < 0:
                    lp += 1
                else:
                    res = [v, a[lp], a[rp]]
                    res.sort()
                    if res not in ans:
                        ans.append(res)
                    lp += 1
                    rp -= 1
        return ans

**My V2** (Solution2 with edge cases. Leetcode accepted.) ::

    class Solution:
        def threeSum(self, a: List[int]) -> List[List[int]]:
            a.sort()
            ans = []
            if len(a) < 3:
                return []
            if len(a) == 3:
                return [a] if sum(a) == 0 else []
            for i in range(len(a) - 2):
                if a[i] > 0:
                    break
                if i > 0 and a[i] == a[i - 1]:
                    continue
                cur = a[i]
                lp = i + 1
                rp = len(a) - 1
                while lp < rp:
                    s = sum([cur, a[lp], a[rp]])
                    if s < 0:
                        lp += 1
                    elif s > 0:
                        rp -= 1
                    else:
                        ans.append([cur, a[lp], a[rp]])
                        lp += 1
                        rp -= 1
                        while a[lp] == a[lp - 1] and lp < rp:
                            lp += 1
            return ans

**Solution 2 explained** ::

    class Solution:
        def threeSum(self, nums: List[int]) -> List[List[int]]:
            res = []
            nums.sort()

            for i, a in enumerate(nums):
                # Skip positive integers
                if a > 0:                        #0
                    break
                if i > 0 and a == nums[i - 1]:   #1
                    continue
                l, r = i + 1, len(nums) - 1   #2
                while l < r:
                    threeSum = a + nums[l] + nums[r]
                    if threeSum > 0:
                        r -= 1
                    elif threeSum < 0:
                        l += 1
                    else:
                        res.append([a, nums[l], nums[r]])
                        l += 1
                        r -= 1
                        while nums[l] == nums[l - 1] and l < r:  #3
                            l += 1
            return res

#0 If current number (a) is positive, then all the following nums are positive 
as well (sorted array), and we won't sum them to 0.

| #1 To avoid duplicates
| if it is the same number as prev, e.g. [-2,-2,0,3..], we don't use it.
| I.e. we don't use it as the <first number> again for triplets like [-2,0,2], [-2,0,2].
| (We can use it as a second like [-2,-2,4].)

#2 3 nums = Our current value and two pointers::

    # [-4,-1,-1,0,1,2] 
    #   a  L        R
 
| #3 To avoid duplicates
| Again if we have 

::

    # [-1,-1,-1,0,3..]
    #   a   L

If we move L+=1, it will move to the same value, so we move L till it is a different value, or L meets R.  

| **Solution 1 (Nested loops)**
| Uses Two Sum internally.
| So time will be the same as for Solution 2 (sorting+nested loops).
| Space will definitely be O(n). (While in Solution 2 space can be O(1).)
 
| Account for edge cases, sort, use 2sum internally.
| 2sum:
| -Main loop for each item in a, num1.
| -Yes each time in the main loop make a new hash for all items except current.
| -Yes, there is another nested loop for num2 (in a[i+1:])
| -you are looking for num3 = -(num1+num2)
| -If found triplet, account for permutations of the same (sort, check if not in ans).
| -else add to hash num (of the 2nd loop)

::

    class Solution(object):
        def threeSum(self, nums):
            # edge cases
            if not nums or len(nums) < 3:
                return []
            if len(nums) == 3:
                return [nums] if sum(nums) == 0 else []
            if nums.count(0) == len(nums):
                return [[0,0,0]]
            res = []
            nums.sort()

            for i in range(len(nums)):
                cur = nums[i]
                # 2 sum
                d = {}
                for j, x in enumerate(nums[i+1:]):
                    # cur + x + y = 0
                    # -> y = -x - cur
                    if -x-cur in d:
                        tmp = [cur, x, -x-cur]
                        tmp.sort()  
                        if tmp not in res:
                            res.append(tmp)
                    else:
                        d[x] = j
            return res

    nums = [-1,0,1,2,-1,-4]
    print(threeSum(nums))    # [[-1, 0, 1], [-1, -1, 2]]

My V
(Brute force, use Python std lib.) ::

    from itertools import combinations
    def sums_to_zero(a):
        # Out of all combinations with size 3, choose those that sum to 0.
        combos = [c for c in combinations(a, 3) if sum(c) == 0]
        # Choose only unique combinations
        ans = []
        for c in combos:
            c = list(c)
            c.sort()
            if c not in ans:
                ans.append(c)
        return ans

    nums = [-1, 0, 1, 2, -1, -4]
    print(sums_to_zero(nums))  #[[-1, 0, 1], [-1, -1, 2]]

58. (LC 16) 3Sum Closest
--------------------------
`16. 3Sum Closest <https://leetcode.com/problems/3sum-closest/submissions/1189356541/>`_
*(Medium)*

| **Keys**:
| -sort
| -three pointers, diff var
| -if threeSum > Greater than target, move RP

::

    #      <----|
    #  [-4,-1,1,2]
    #           R

-if threeSum < Less than target, move MidPoint ::

    #       |--->
    #  [-4,-1,1,2]
    #       M

**Solutions**::

    ### My V3 (LC accepted, 16, 74%)
    def threeSumClosest(a, t):
        a.sort()
        ans = 0
        dif = float("inf")
        for lp in range(len(a) - 2):
            rp = len(a) - 1
            mp = lp + 1
            while mp < rp:
                summing = sum([a[lp], a[mp], a[rp]])
                cur_dif = abs(summing - t)
                if cur_dif < dif:
                    dif = cur_dif
                    ans = summing
                if summing > t:
                    rp -= 1
                elif summing < t:
                    mp += 1
                else:
                    return t
        return ans

::

    # My V (Stdlib)
    import itertools as it

    def sum_closest(a, t):
        combos = it.combinations(a, 3)
        sums = [sum(c) for c in combos]
        ans = sums[0]
        dif = abs(t - ans)
        for s in sums:
            if abs(t - s) < dif:
                ans = s
        return ans

    nums = [-1, 2, 1, -4]
    target = 1
    print(sum_closest(nums, target))  # 2

Solutions Time:  O(n^2)::

    ### 1
    def threeSumClosest(nums, target):
        N = len(nums)
        nums.sort()
        res = float('inf') # sum of 3 numbers
        for t in range(N):
            i, j = t + 1, N - 1
            while i < j:
                _sum = nums[t] + nums[i] + nums[j]
                if abs(_sum - target) < abs(res - target):
                    res = _sum
                if _sum > target:
                    j -= 1
                elif _sum < target:
                    i += 1
                else:
                    return target
        return res

    ### 2 (pretty much the same)
    def threeSumClosest(num, target):
        num.sort()
        mindiff = 100000
        res = 0
        for i in range(len(num)):
            left = i + 1
            right = len(num) - 1
            while left < right:
                sum = num[i] + num[left] + num[right]
                diff = abs(sum - target)
                if diff < mindiff:
                    mindiff = diff
                    res = sum
                if sum == target:
                    return sum
                elif sum < target:
                    left += 1
                else:
                    right -= 1
        return res

59. (LC 989) Add to Array-Form of Integer
-------------------------------------------
*(Easy)*
The array-form of an integer num is an array representing its digits in left to right order.
For example, for num = 1321, the array form is [1,3,2,1].
Given num, the array-form of an integer, and an integer k, return the array-form of the integer num + k.

| Example 1:
| Input: num = [1,2,0,0], k = 34
| Output: [1,2,3,4]
| Explanation: 1200 + 34 = 1234
 
| Example 2:
| Input: num = [2,7,4], k = 181
| Output: [4,5,5]
| Explanation: 274 + 181 = 455
 
| Example 3:
| Input: num = [2,1,5], k = 806
| Output: [1,0,2,1]
| Explanation: 215 + 806 = 1021

::

    ### My v
    def add_to_array(a, n):
        a = [0] + a
        for i in range((len(a) - 1), -1, -1):
            a[i] = a[i] + (n % 10)   #4+(181%10)=4+1=5
            a[i - 1] += a[i] // 10   #7+5//10, i.e. +carry
            a[i] = a[i] % 10         #if there was carry on a[i], chop it off
            n = n // 10              #chop of right digit from 181, leaving 18
        if a[0] == 0:
            return a[1:]
        return a

    num = [2, 7, 4]
    k = 181
    print(add_to_array(num, k))  #[4, 5, 5]

    ### Solution 1
    # (operation on array)
    class Solution:
        def addToArrayForm(self, num: List[int], k: int) -> List[int]:
            s = ""
            for i in num:
                s += str(i)       
            answer = int(s) + k
            return  list("".join(str(answer)))  #why not list(str(answer))

    # Using list comprehension
    class Solution:
        def addToArrayForm(self, A: List[int], K: int) -> List[int]:
            return [int(x) for x in str(int(''.join(str(x) for x in A))+K)]

| ``divmod(a,b)``
| Given two numbers (a=what you want to divide, b=divide by )
| Gives as result (quotient, remainder)

>>> divmod(26, 5)
(5, 1)

::

    ### Solution 2
    class Solution:
        def addToArrayForm(self, num: List[int], k: int) -> List[int]:
            i, carry = len(num) - 1, 0
            ans = []
            while i >= 0 or k or carry:
                carry += (0 if i < 0 else num[i]) + (k % 10)
                carry, v = divmod(carry, 10)
                ans.append(v)
                k //= 10
                i -= 1
            return ans[::-1]

| **Explained**
| E.g., Input: num = [1,2,0,0], k = 34
 
|     i, carry = len(num) - 1, 0
| # We start at the LSB, i.e. last index i of array 'num'.
| Here at first iteration i=4
| Set carry to 0.
 
|     while i >= 0 or k or carry:
| # Because we need to carry on if k > number in array.
| 1)No worries, we won't do i=-1 lookups in array nums. carry=0 if i < 0.
| 2)Strangely we set carry to be the result of normal sum of num[i] + k%10.
| FYI k%10 is the LSB of k, here 34%10=4
| First loop, i=3, carry = num[3] + 4 = 4
| We set this right in the next step.
| 3)
|     carry, v = divmod(carry, 10)
 
 >>> divmod(4, 10)
 (0, 4)
 
| Now carry is 0, v=4
| FYI, if instead of 4, we had 18, then we get our carry=1 with:
 
 >>> divmod(18, 10)
 (1, 8)
 
| 4)
|     ans.append(v)
| 5)
|     k //= 10
|     i -= 1
| Remove k's LSB (34//10 = 3)
| Move to the next index.
| Next we will be adding 3 to num[3-1].

::

    ### Solution 3
    class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        for i in reversed(range(len(num))):
        k, num[i] = divmod(num[i] + k, 10)

        while k > 0:
        num = [k % 10] + num
        k //= 10

        return num

60. (LC 419) Battleships in a Board
-------------------------------------
`419. Battleships in a Board <https://leetcode.com/problems/battleships-in-a-board/>`_
Medium

| **Solution 1**
| **Keys:**

-Just check for each cell that has 'X' if the cell <immediately above> or the cell 
<immediately to the left> also has 'X'. Means you already counted that 'X', so you can continue. 

::

    class Solution(object):
        def countBattleships(self, board):
            """
            :type board: List[List[str]]
            :rtype: int
            """
            h = len(board)
            w = len(board[0]) if h else 0

            ans = 0
            for x in range(h):
                for y in range(w):
                    if board[x][y] == 'X':
                        if x > 0 and board[x - 1][y] == 'X':  #if there is a ship above
                            continue
                        if y > 0 and board[x][y - 1] == 'X':  #if there is a sip to the left
                            continue
                        ans += 1
            return ans

| Note, 
| h (height) is x (first index in matrix)

| **Solution 2**
| If you overdid problems on graphs. 

::

    ### My V (LC accepted 5, 8% slow)
    class Solution:
        def countBattleships(self, board: List[List[str]]) -> int:
            rows = len(board)
            cols = len(board[0])
            visited = set()
            ships = 0

            def dfs(r,c):
                if r not in range(rows) or c not in range(cols) or (
                        board[r][c] == '.' or (r,c) in visited):
                    return
                visited.add((r,c))
                dfs(r, c+1)
                dfs(r, c-1)
                dfs(r+1, c)
                dfs(r-1, c)

            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == 'X' and (r,c) not in visited:
                        dfs(r,c)
                        ships +=1
            return ships

61. (LC 121) Best Time to Buy and Sell Stock
------------------------------------------------
`121. Best Time to Buy and Sell Stock <https://leetcode.com/problems/best-time-to-buy-and-sell-stock>`_
*(Easy)*

In short: buy and sell once, return max profit.

You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and 
choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. 
If you cannot achieve any profit, return 0.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

::

    ### My V
    def buy_sell(a):
        max_pofit, min_price = 0, a[0]
        for p in a:
            min_price = min(min_price, p)
            max_pofit = max(max_pofit, p - min_price)
        return max_pofit

    ### Solution 1
    class Solution(object):
        def maxProfit(self, prices):
            if len(prices) == 0:
                return 0
            ### NOTE : we define 1st minPrice as prices[0]
            minPrice = prices[0]
            maxProfit = 0
            ### NOTE : we only loop prices ONCE
            for p in prices:
                # only if p < minPrice, we get minPrice
                if p < minPrice:
                    minPrice = p
                ### NOTE : only if p - minPrice > maxProfit, we get maxProfit
                elif p - minPrice > maxProfit:
                    maxProfit = p - minPrice
            return maxProfit

    ### Other Solutions
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            ans, mi = 0, inf
            for v in prices:
                ans = max(ans, v - mi)
                mi = min(mi, v)
            return ans

    class Solution(object):
        # @param prices, a list of integers
        # @return an integer
        def maxProfit(self, prices):
            max_profit, min_price = 0, float("inf")
            for price in prices:
                min_price = min(min_price, price)
                max_profit = max(max_profit, price - min_price)
            return max_profit

62. (LC 309) Best Time to Buy and Sell Stock with Cooldown
------------------------------------------------------------
`309. Best Time to Buy and Sell Stock with Cooldown 
<https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/>`_
*(Medium)*
::

    # 1
    class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        sell = 0
        hold = -math.inf
        prev = 0

        for price in prices:
        cache = sell
        sell = max(sell, hold + price)
        hold = max(hold, prev - price)
        prev = cache

        return sell

    # 2
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            f, f0, f1 = 0, 0, -prices[0]
            for x in prices[1:]:
                f, f0, f1 = f0, max(f0, f1 + x), max(f1, f - x)
            return f0

# 3 Dynamic programming, O(n) [:ref:`10 <ref-label>`]::

    from typing import List

    def maxProfit(prices: List[int]) -> int:
        # State: Buying or Selling?
        # If Buy -> i + 1
        # If Sell -> i + 2   # +2 because +cooldown day

        dp = {}  # key=(i, buying) val=max_profit, dp implements cashing

        def dfs(i, buying):
            if i >= len(prices):
                return 0
            if (i, buying) in dp:
                return dp[(i, buying)]

            cooldown = dfs(i + 1, buying)
            if buying:
                buy = dfs(i + 1, not buying) - prices[i]
                dp[(i, buying)] = max(buy, cooldown)
            else:
                sell = dfs(i + 2, not buying) + prices[i]
                dp[(i, buying)] = max(sell, cooldown)
            return dp[(i, buying)]

        return dfs(0, True)

    prices = [1, 2, 3, 0, 2]
    print(maxProfit(prices))

63. (LC 122) Best Time to Buy and Sell Stock II
-------------------------------------------------
`122. Best Time to Buy and Sell Stock II
<https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/>`_
*(Medium)*

| Key is that you can buy and sell on the same day.
| -Basically you can sell each time you meet a higher price. 
| -If successful sell, then set <cur buy price> = <cur price> (so sell and buy on the same day).
| -Add up the results.
| E.g. prices=[1,2,3,4,5]
| You don't have to look for the best option, which is here buy at 1, sell at 5.
| You can buy at 1, sell at 2. Then buy at 2, sell at 3 etc.

**Solution 1** [:ref:`2 <ref-label>`] ::

    ### Solution 1
    from typing import List
    import itertools
    def maxProfit(prices: List[int]) -> int:
        return sum(max(0, b - a) for a, b in itertools.pairwise(prices))

    prices = [7,1,5,3,6,4]
    print(maxProfit(prices)) # 7

| # tools
| ``itertools.pairwise(iterable)``
| Roughly equivalent to:
| pairwise('ABCDEFG') --> AB BC CD DE EF FG

**Solution 2** [:ref:`10 <ref-label>`] ::

    ### Solution 2
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            max_profit = 0
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    max_profit += prices[i] - prices[i-1]
            return max_profit

**My V** (LC accepted 50, 70%) ::

    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            cur_min = prices[0]
            total_profit = 0
            for price in prices:
                cur_min = min(cur_min, price)
                profit = price - cur_min
                if profit > 0:
                    cur_min = price
                total_profit += profit
            return total_profit

Emulating as close as possible the classic buy-sell stock.

64. (LC 1014) Best Sightseeing Pair
-------------------------------------
`1014. Best Sightseeing Pair <https://leetcode.com/problems/best-sightseeing-pair/>`_
*(Medium)*

| # In short
| Given an array, return the highest 
| values[i] + values[j] + i - j
 
| # Keys
| i - j is the distance between the sightseeing spots.

::

    ### Solution 1
    class Solution:
        def maxScoreSightseeingPair(self, A: List[int]) -> int:
            n = len(A)
            pre = A[0] + 0
            res = 0
            for i in range(1, n):
                res = max(res, pre + A[i] - i)
                pre = max(pre, A[i] + i)
            return res

    # The same (breaking down the steps)
    from typing import List
    def f(A: List[int]) -> int:
        n = len(A)
        pre = A[0] + 0
        res = 0
        for i in range(1, n):
            cur_res = pre + A[i] - i
            res = max(res, cur_res)
            possible_pre = A[i] + i
            pre = max(pre, possible_pre)
        return res 

| # Explained solution 1       
| ``res = max(res, pre + A[i] - i)``
| Final response, check if we found a greater 
| (previous spot + current spot - distance between them)
 
| # - i, + i confusion
| it might seem unfair that in 
| ``res = max(res, pre + A[i] - i)``
| We each time subtract the full index, not the net distance (i - j).
| But actually it is because in the second line:
| ``pre = max(pre, A[i] + i)``
| A[i] + i
| + i means the value at i will carry with it its distance.
| So if our new previous = value + 3 (it is at index 3).
| Then the next time we calculate response, e.g. at i=4, 
| max(res, value+3 - 4)
| We see that if they are only 1 place apart, we end up subtracting only that 1, not 4.
| ==>previous CARRIES its distance with its value.

| E.g. A = [2,4,10]
| pre=A[0]=2, res=0
| i=1
| res=max(0, 2+4-1), res=5
| pre=max(2, 4+1), pre=5
| i=2
| res=max(5, 5+10-2), res=13 (so really 5+10-2=4+10-1)
| pre=max(5, 10+2), pre=12  ==>10 carries the weight of where it is at, i.e. index 2

65. (LC 605) Can Place Flowers
---------------------------------
| *(Easy)*
| You have a long flowerbed in which some of the plots are planted, and some are not. 
| However, flowers cannot be planted in adjacent plots.

Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, 
and an integer n, return true if n new flowers can be planted in the flowerbed without 
violating the no-adjacent-flowers rule and false otherwise.

| Example 1:
| Input: flowerbed = [1,0,0,0,1], n = 1
| Output: true
 
| Example 2:
| Input: flowerbed = [1,0,0,0,1], n = 2
| Output: false

::

    ### My V
    def can_plant(a, n):
        a = [0] + a + [0]
        cnt = 0
        for i in range(1, len(a) - 1):
            if not a[i] & 1:
                if a[i - 1] == 0 and a[i + 1] == 0:
                    cnt += 1
                    a[i] = 1
        return cnt >= n

    flowerbed = [1, 0, 0, 0, 1]
    print(can_plant(flowerbed, 1))  # True
    print(can_plant(flowerbed, 2))  # False

    ### Solution 1
    class Solution:
        def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
            flowerbed = [0] + flowerbed + [0]
            for i in range(1, len(flowerbed) - 1):
                if sum(flowerbed[i - 1 : i + 2]) == 0:
                    flowerbed[i] = 1
                    n -= 1
            return n <= 0

| ### Explained
| (See explanation for solution 2 in addition.)
| #Here we check if values at [i=1, i=2, i=3] all add up to 0, none is set to 1 in one go.
| #We also account for the fact that we may have A = [0,0,1,0,1], 
| so we may plant at i=0.
| Because we do:
|     flowerbed = [0] + flowerbed + [0]
| We start the loop for i in range(1..), but we actually start at original i=0, 
| which is now i=1, because we prepended with\appended to array 0s.

::

    ### Solution 2
    class Solution(object):
        def canPlaceFlowers(self, flowerbed, n):
            """
            :type flowerbed: List[int]
            :type n: int
            :rtype: bool
            """
            for i, num in enumerate(flowerbed):
                if num == 1: continue
                if i > 0 and flowerbed[i - 1] == 1: continue
                if i < len(flowerbed) - 1 and flowerbed[i + 1] == 1: continue
                flowerbed[i] = 1
                n -= 1
            return n <= 0

| ### Explained
| 1) If num at i is 1, continue
| 2) Check adjacent values to the left and right of the current i, see if they are 1,
| then we cannot plant.
 
| if i > 0 and flowerbed[i - 1] == 1: continue
| # If it is not the first element (at i=0), check that element to the left (i-1)
| is not 1. Else continue the loop.
 
| if i < len(flowerbed) - 1 and flowerbed[i + 1] == 1: continue
| # If we are looking not at the last element of the array (len(A)-1),
| (then it has no elements to the right)
| then check if element to the right (at i+1) is 1. 

:: 

    ### Solution 3
    class Solution(object):
        def canPlaceFlowers(self, flowerbed, n):
            """
            :type flowerbed: List[int]
            :type n: int
            :rtype: bool
            """
            flowerbed = [0] + flowerbed + [0]
            N = len(flowerbed)
            res = 0
            for i in range(1, N - 1):
                if flowerbed[i - 1] == flowerbed[i] == flowerbed[i + 1] == 0:
                    res += 1
                    flowerbed[i] = 1
            return res >= n

