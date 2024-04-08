Array Questions Part 7
======================
96. (LC 945) Minimum Increment to Make Array Unique
-----------------------------------------------------
`945. Minimum Increment to Make Array Unique <https://leetcode.com/problems/minimum-increment-to-make-array-unique/>`_
Medium ::

    class Solution(object):
    def minIncrementForUnique(self, A):
        A.sort()
        ans = 0
        for i in range(1, len(A)):
        if A[i] > A[i - 1]: 
            continue
        ans += A[i - 1] - A[i] + 1
        A[i] = A[i - 1] + 1
        return ans

    def minIncrementForUnique(nums) -> int:
        nums.sort()
        print(nums)
        ans = 0
        for i in range(1, len(nums)):
            if nums[i] <= nums[i - 1]:
                d = nums[i - 1] - nums[i] + 1
                nums[i] += d
                ans += d
        print(nums)
        return ans

97. (LC 209) Minimum Size Subarray Sum
-----------------------------------------
`209. Minimum Size Subarray Sum <https://leetcode.com/problems/minimum-size-subarray-sum/>`_
Medium

| **Key**
| -Add to sum (sum+=a[rp]) as you iterate through the array.
| sum-=a[lp] as you drop values on the left.

::

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        ans = n + 1
        s = j = 0
        for i, x in enumerate(nums):
            s += x
            while j < n and s >= target:
                ans = min(ans, i - j + 1)
                s -= nums[j]
                j += 1
        return ans if ans <= n else 0

j is pointer1 (start of substr), i is pointer2 (end).
::

    ### My V (LC accepted 30, 38%, 18,11%)
    class Solution:
        def minSubArrayLen(self, t: int, a: List[int]) -> int:
            ans = float("inf")
            lp = 0
            s = 0  #sum
            for rp in range(len(a)):
                s += a[rp]
                while (s - a[lp]) >= t and lp <= rp:
                    s -= a[lp]
                    lp += 1
                if s >= t:
                    ans = min(ans, rp - lp + 1)
            return ans if ans < float("inf") else 0

| **Explained** (main V)
| Example:
| Input: target = 7, nums = [2,3,1,2,4,3]
| Output: 2
 
|     ``n = len(nums)``
|     ``ans = n + 1``
| Basically we set answer to a length greater than the len of array of nums.
| I.e. making it unsatisfactory for the end result.
 
|     ``for i, x in enumerate(nums):``
|         ``s += x``
| We accumulate s to achieve the target. When, and only when, s >= target,
| we enter the while loop:
|         ``while j < n and s >= target:``
 
|             ``ans = min(ans, i - j + 1)``
| Upon entering the while loop, our first step is to 
| <calculate the len of subarray that initially got us inside the loop>.
| E.g. nums = [2,3,1,2,4,3], our first time in while loop is when i=3, j=0
| ans=min(ans, 3-0+1), min(7, 4)=4  --->min len(subarray)=4
 
|             ``s -= nums[j]``
|             ``j += 1``
| What the rest of this while loop does - is it tries to see if there is a shorter
| subarray by 
| 1)dropping values from the left. If [2,3,1,2], drop 2.
| 2)by incrementing j+1, we decrement len(subarray), because i-j in
| ans = min(ans, i - j + 1).
| (j-the start of substring, i - the end of substring).
 
| BUT it makes sense to continue with this dropping only while after dropping nums
| on the left, (s is still >= target). That's what the while loop has.
 
| So it looks like:
| [2,3,1,2,4,3]
| [2,3,1,2] s=8, enter while, drop left nums: [3,1,2] s=6, s<7 so return to main loop
| [3,1,2,4], s=10, enter while, [1,2,4], s=7, [2,4], s<7,
| [1,2,4,3], s=10, enter while, [2,4,3], s=9, [4,3]. len=2.

Also note that we continue the main loop with left numbers dropped.

98. (LC 3) Longest Substring Without Repeating Characters
------------------------------------------------------------
`3. Longest Substring Without Repeating Characters <https://leetcode.com/problems/longest-substring-without-repeating-characters/>`_
Medium
::

    ### My V
    def longest_unique(s):
        ans = cur = 0
        d = {}
        for i in range(len(s)):
            if s[i] not in d:
                d[s[i]] = True
                cur += 1
            else:
                ans = max(ans, cur)
                cur = 0
                d = {}
        return max(ans, cur)

    ### Solution
    class Solution:
        def lengthOfLongestSubstring(self, s: str) -> int:
            ss = set()
            i = ans = 0
            for j, c in enumerate(s):
                while c in ss:
                    ss.remove(s[i])
                    i += 1
                ss.add(c)
                ans = max(ans, j - i + 1)
            return ans

| i and j represent the start and end positions of the non-repeating substring.
| If c not in ss set, we add c to ss set. Calculate ans.
 
| As soon as we meet a dup, we start to remove from set.
| Note two cases:
| E.g. 'abca' and 'abcc'.
| 1)'abca'
| Upon encountering 'a', we remove it from ss. {b,c,a}, means we can straight away
| continue building, i.e. adding to ss.
| 2) 'abcc'
| Encountering 'c', we remove fromm ss value at s[i], i.e. 'a', yes not 'c', then 
| remove 'b'.
| It it exactly what we want because when having 'abcc', it only to make sense 
| to move i all the way to c, i.e. to drop everything on the left, 
| and start building from 'c...' to the right.

99. (LC 163) Missing Ranges
-------------------------------

**Task**
You are given an inclusive range [lower, upper] and a sorted unique integer array nums, 
where all elements are within the inclusive range.

A number x is considered missing if x is in the range [lower, upper] and x is not in nums.

Return the shortest sorted list of ranges that exactly covers all the missing numbers. 
That is, no element of nums is included in any of the ranges, and each missing 
number is covered by one of the ranges.

| Example 1:
| Input: nums = [0,1,3,50,75], lower = 0, upper = 99
| Output: [[2,2],[4,49],[51,74],[76,99]]
| Explanation: The ranges are:
| [2,2]
| [4,49]
| [51,74]
| [76,99]

| **Tools**
| itertools.pairwise(iterable)
| pairwise('ABCDEFG') --> AB BC CD DE EF FG

::

    ### Solution
    from itertools import pairwise
    class Solution:
        def findMissingRanges(
            self, nums: List[int], lower: int, upper: int
        ) -> List[List[int]]:
            n = len(nums)
            if n == 0:
                return [[lower, upper]]
            ans = []
            if nums[0] > lower:
                ans.append([lower, nums[0] - 1])
            for a, b in pairwise(nums):     #**
                if b - a > 1:
                    ans.append([a + 1, b - 1])
            if nums[-1] < upper:
                ans.append([nums[-1] + 1, upper])
            return ans

    ### My V (without using stdlib)
    def find_missing_ranges(a, l, u):
        mr = []  # missing ranges
        for i in range(len(a)):
            if i == 0:
                if a[i] > l:
                    mr.append([l, a[i] - 1])
            elif i == len(a) - 1:
                if a[i] < u:
                    mr.append([a[i] + 1, u])
            elif a[i + 1] != a[i] + 1:
                mr.append([a[i] + 1, a[i + 1] - 1])
        return mr

    ### V2
    def f(a, l, u):
        ans = []
        for i in range(len(a)):
            cur_range = []
            if i == 0:
                if a[i] > l:
                    cur_range.append(l)
                    cur_range.append(a[i] - 1)
                else:
                    continue  #to avoid adding to ans empty []
            elif i == len(a) - 1:
                if a[i] < u:
                    cur_range.append(a[i] + 1)
                    cur_range.append(u)
            elif (a[i] + 1) < a[i + 1]:
                cur_range.append(a[i] + 1)
                cur_range.append(a[i + 1] - 1)
            ans.append(cur_range)
        return ans

    nums = [0, 1, 3, 50, 75]
    lower = 0
    upper = 99
    # expect [[2,2],[4,49],[51,74],[76,99]]
    print(f(nums, lower, upper))  # [[2, 2], [4, 49], [51, 74], [76, 99]]

100. (LC 238) Product of Array Except Self
---------------------------------------------
`238. Product of Array Except Self <https://leetcode.com/problems/product-of-array-except-self/>`_
Medium

| **Idea**
| Two Passes.
| We first traverse the array from left to right. Then in reverse order.

::

    class Solution:
        def productExceptSelf(self, nums: List[int]) -> List[int]:
            n = len(nums)
            ans = [0] * n
            left = right = 1
            for i, x in enumerate(nums):
                ans[i] = left
                left *= x
            for i in range(n - 1, -1, -1):
                ans[i] *= right
                right *= nums[i]
            return ans

**Explained**

Traversing forward we set a[0] to 1, i.e. we "set a delay" by 1 item, 
when calculating each accumulated product.
So after forward iteration we have ans = [1,1,2,6]  (initial nums = [1,2,3,4])

When we iterate backwards, because right is initially set to 1, we have a delay by
1 item from the right.
So having  ans = [1,1,2,6], on first iteration backwards we multiply 6 by 1 
(only in the next move we multiply by 4).

**My V (Brute force, using stdlib)**
Compute accumulated product for array that each time excludes one item (use slicing
for this). Then for the final list we use just the last item, [-1], of the accumulated list. ::

    import itertools as it
    def prod_except_self(a):
        return [
            list(it.accumulate((a[:i] + a[i + 1 :]), lambda x, y: x * y))[-1]
            for i in range(len(a))
        ]

    nums = [1, 2, 3, 4]
    print(prod_except_self(nums)) #[24, 12, 8, 6]










