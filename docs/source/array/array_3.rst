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












