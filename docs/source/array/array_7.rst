Array Questions Part 7
======================
96. (LC 945) Minimum Increment to Make Array Unique
-----------------------------------------------------
`945. Minimum Increment to Make Array Unique <https://leetcode.com/problems/minimum-increment-to-make-array-unique/>`_
Medium ::

    ### My V (LC accepted, 40, 90)
    class Solution:
        def minIncrementForUnique(self, nums: List[int]) -> int:
            nums.sort()
            moves = 0
            for i in range(1, len(nums)):
                if nums[i] <= nums[i-1]:
                    dif = nums[i-1] - nums[i] + 1
                    nums[i] += dif
                    moves += dif
            return moves

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

101. (LC 53) Maximum Subarray
---------------------------------
`53. Maximum Subarray <https://leetcode.com/problems/maximum-subarray/>`_
Medium

**Keys**

| -If cur sum drops to a negative value, drop the entire cur subarray and start building from scratch.
| +Don't calculate the subarry sum each time. Do just += num
| -Edge case when adjacent nums like [-3,4].
| If going forward their sum is 1.
| If calculating the sum first, s=-3, less than 0, so drop to 0 first, then add +4.
| OR set the initial sum to 0 from the start. And drop it to 0 again if cur_s < 0.

::

    ### My V (LC accepted 50, 50%)
    def f(a):
        max_s = cur_s = a[0]
        for n in a[1:]:
            cur_s = max(0, cur_s)  #drop sum to 0 Before adding the next num
            cur_s += n
            max_s = max(max_s, cur_s)
        return max_s

    ### Solution 2 (neetcode, LC accepted 75, 98%)
    def f(a):
        max_s = a[0]
        cur_s = 0
        for n in a:
            cur_s += n
            max_s = max(max_s, cur_s)
            if cur_s < 0:
                cur_s = 0
        return max_s

    ### Solution 1
    class Solution:
        def maxSubArray(self, nums: List[int]) -> int:
            ans = f = nums[0]
            for x in nums[1:]:
                f = max(f, 0) + x   #**
                ans = max(ans, f)
            return ans

    #** max out of sum of previous numbers (because f is built out of +x0+x1..), 
    or 0+x, i.e. x alone. 
    I.e. if previous sum of numbers is lower than 0.

102. (LC 152) Maximum Product Subarray
-------------------------------------------
`152. Maximum Product Subarray <https://leetcode.com/problems/maximum-product-subarray/>`_
Medium

| **Keys:**
| -track both max, min. Min, max selected from (cur_num, max*cur_num, min*cur_num) Greedy select answer.

| **Brute force** 
| Making all possible subarrays would be O(n**2)

::

    # Illustration
    # [2,3,-2,4]     [2,3,-2,4]
    # |-|              |-|
    # |---| etc        |----| etc

| **Recognize the pattern**
| Multiplying nums with dif signs (-/+) flips the product.
| So keep track of both min, max product so far.
| The final answer will depend on whether the next num is -/+.
 
| //next_num (+): makes max bigger, min smaller
| //next_num (0): resets min, max (then min=max=0)
| //next_num (-): max, min swap places (because the sign gets flipped)
 
::

#     [-3, 2,-4, 6, 0, -8, 5]
# max: -3, 2,24,144,0,  0, 5
# mix: -3,-6,-8,-48,0, -8,-40
 
| At -3
| max_so_far=-3
| min_so_far=-3
| max_overall=-3
| At 2
| max_so_far=2  arr=[2]
| min_so_far=-6 arr=[-3,2]
| max_overall=2
| At -4
| max_so_far=24 arr=[-3,2,-4]
| min_so_far=-8 arr=[2,-4]
| max_overall=24
| At 6
| max_so_far=144 arr=[-3,2,-4,6]
| min_so_far=-48 arr=[2,-4,6]
| max_overall=144
| etc.

**Solution 1** [:ref:`10 <ref-label>`]
::

    ### Solution 1
    class Solution:
        def maxProduct(self, nums: List[int]) -> int:
            # O(n)/O(1) : Time/Memory
            res = nums[0]
            curMin, curMax = 1, 1
            for n in nums:
                tmp = curMax * n                         #1
                curMax = max(n * curMax, n * curMin, n)
                curMin = min(tmp, n * curMin, n)
                res = max(res, curMax)
            return res

| #1
| tmp variable because the code could have been:
|     curMax = max(n * curMax, n * curMin, n)
|     curMin = max(n * curMax, n * curMin, n)
| But when computing curMin, we want the old value of curMax, not the reassigned in the step above.

**C++** [:ref:`14 <ref-label>`]

.. code-block:: cpp

    //C++
    class Solution {
    public:
        int maxProduct(vector<int>& nums) {
            int max_overall = nums[0];
            int max_ending_here = nums[0], min_ending_here = nums[0];
            
            for(int i = 1; i < nums.size(); ++i){
                int tmp = max_ending_here;
                max_ending_here = max({nums[i], nums[i]*max_ending_here, nums[i]*min_ending_here});
                min_ending_here = min({nums[i], nums[i]*tmp, nums[i]*min_ending_here});
                max_overall = max(max_overall, max_ending_here);
            }
            return max_overall;
        }
    };

| max({item1, item2, item3})
| When >2 params, include initializer braces {}.

103. (LC 217) Contains Duplicate
--------------------------------------
`217. Contains Duplicate <https://leetcode.com/problems/contains-duplicate/>_`
Easy ::

    ### Solution 1 (LC accepted, most efficient)
    class Solution:
        def containsDuplicate(self, nums: List[int]) -> bool:
            return len(set(nums)) < len(nums)

    ### Solution 2 (my V, LC accepted 2nd in efficiency)
    class Solution:
        def containsDuplicate(self, nums: List[int]) -> bool:
            cnt = collections.Counter(nums)
            dups = [k for k, v in cnt.items() if v > 1]
            return len(dups) > 0

    ### Solution 3 (LC accepted least efficient)
    import itertools
    class Solution:
        def containsDuplicate(self, nums: List[int]) -> bool:
            return any(a == b for a, b in itertools.pairwise(sorted(nums)))

104. (LC 33) Search in Rotated Sorted Array
-----------------------------------------------
`33. Search in Rotated Sorted Array <https://leetcode.com/problems/search-in-rotated-sorted-array/>`_
Medium

Hint: Binary search ::

    ### V2 Neetcode
    class Solution:
        def search(self, nums: List[int], target: int) -> int:
            l, r = 0, len(nums) - 1

            while l <= r:            #if a=[1]
                mid = (l + r) // 2
                if target == nums[mid]:
                    return mid

                # left sorted portion
                if nums[l] <= nums[mid]:  #if this side of array is sorted
                    if target > nums[mid] or target < nums[l]:
                        l = mid + 1
                    else:
                        r = mid - 1
                # right sorted portion
                else:
                    if target < nums[mid] or target > nums[r]:
                        r = mid - 1
                    else:
                        l = mid + 1
            return -1

| O(logN) means we would use binary search.
| a=[4,5,6,7,0,1,2]
| In an array sorted with an offset/rotation, we will have 2 portions that are 
| exactly sorted.
| But using M in binary search, we could end up with 2 portions, that are not sorted.
| To combat this fact:
| 1)Having L,M,R. t=0. 
| We first check if portion L-M is sorted. How: we check if a[L] <= a[M].
| if it is sorted, and our t>a[M] then we can be certain, we need to search M-R.

| +EDGE CASE. t<a[M] could be both on left and right sides.
| => If t<a[M] and t<a[L], then we know we have to search M-R.

::

    ### V 1
    class Solution:
        def search(self, nums: List[int], target: int) -> int:
            n = len(nums)
            left, right = 0, n - 1
            while left < right:
                mid = (left + right) >> 1
                if nums[0] <= nums[mid]:
                    if nums[0] <= target <= nums[mid]:
                        right = mid
                    else:
                        left = mid + 1
                else:
                    if nums[mid] < target <= nums[n - 1]:
                        left = mid + 1
                    else:
                        right = mid
            return left if nums[left] == target else -1

    ### My V
    def f(a, n):
        try:
            ans = a.index(n)
        except:
            ans = -1
        return ans

105. (LC 11) Container With Most Water
--------------------------------------------
`11. Container With Most Water <https://leetcode.com/problems/container-with-most-water/>`_
Medium

| **Keys**
| -Two pointers.
| -Volume = min(values)*(dif of indices)

::

    ### My V2 (Two Pointers) LC accepted 20,40%
    def f(a):
        l = 0
        r = len(a) - 1
        vmax = 0
        while l < r:
            v = min(a[l], a[r]) * (r - l)  #Volume current
            vmax = max(vmax, v)
            if a[l] < a[r]:
                l += 1
            else:
                r -= 1
        return vmax

    ### Solution
    class Solution:
        def maxArea(self, height: List[int]) -> int:
            i, j = 0, len(height) - 1
            ans = 0
            while i < j:
                t = (j - i) * min(height[i], height[j])
                ans = max(ans, t)
                if height[i] < height[j]:
                    i += 1
                else:
                    j -= 1
            return ans

    ### My V 1
    # (left to right iteration)

    def f(a):
        m1 = (0, 0)
        vmax = 0
        for i in range(len(a)):
            v = min(m1[0], a[i]) * (i - m1[1])
            vmax = max(vmax, v)
            if a[i] - i > m1[0]:
                m1 = (a[i], i)
        return vmax

    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    print(f(height)) #49
    height2 = [1, 1]
    print(f(height2)) #1

| We record the max height as m1 tuple, (value, index).
| We always need to know the index of max value because Volume = height*width 
| (width = current i - m1 index)


