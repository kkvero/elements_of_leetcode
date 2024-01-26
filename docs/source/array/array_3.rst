Array Questions Part 3
======================
56. (LC 1) Two Sum
--------------------
*(Easy)*
Given an array of integers nums and an integer target, return indices of the two 
numbers such that they add up to target.
(You may assume that each input would have exactly one solution, and you may not use the same element twice.)

| # Example 1:
| Input: nums = [2,7,11,15], target = 9
| Output: [0,1]
| Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

::

    ### V my
    def two_sum(a, t):
        d = {}
        for i, n in enumerate(a):
            if t - n in d:
                return i, d[t - n]
            d[n] = i
        return False

    nums = [2, 7, 11, 15]
    print(two_sum(nums, 9)) # (1, 0)

    ### Solution 1   
    def two_sum(array, target):
        dic = {}
        for i, num in enumerate(array):
            if num in dic:
                return dic[num], i
            else:
                dic[target - num] = i
        return None

57. (LC 15) 3Sum
-------------------
*(Medium)*
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] 
such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets,
although the input may contain duplicates.

| # Examples:
| Input: nums = [-1,0,1,2,-1,-4]
| Output: [[-1,-1,2],[-1,0,1]]
| Explanation: 
| nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
| nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
| nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0. 
| The distinct triplets are [-1,0,1] and [-1,-1,2].
| # Input: nums = [0,1,1]
| Output: []
| # Input: nums = [0,0,0]
| Output: [[0,0,0]]

::

    ### Solution 
    # Account for edge cases, sort, use 2sum internally

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

    ### My V
    # (Brute force, use Python std lib.)

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
*(Medium)*
Given an integer array nums of length n and an integer target, find three integers 
in nums such that the sum is closest to target.
Return the sum of the three integers.
(You may assume that each input would have exactly one solution.)

| # Example 1:
| Input: nums = [-1,2,1,-4], target = 1
| Output: 2
| Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

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

    ### 2 (pretty much the same, more compact)
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

