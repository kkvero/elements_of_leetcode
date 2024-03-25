Array Questions Part 6
======================
86. (LC 860) Lemonade Change
-------------------------------
`860. Lemonade Change <https://leetcode.com/problems/lemonade-change/>`_
Easy
::

    class Solution:
        def lemonadeChange(self, bills: List[int]) -> bool:
            five = ten = 0
            for v in bills:
                if v == 5:
                    five += 1
                elif v == 10:
                    ten += 1
                    five -= 1
                else:
                    if ten:
                        ten -= 1
                        five -= 1
                    else:
                        five -= 3
                if five < 0:
                    return False
            return True

    ### My V
    def f(a):
        fives = []
        tens = []
        for n in a:
            if n == 5:
                fives.append(n)
            elif n == 10:
                if len(fives) > 0:
                    fives.pop()
                    tens.append(10)
                else:
                    return False
            elif n == 20:
                if len(tens) > 0 and len(fives) >= 1:
                    tens.pop()
                    fives.pop()
                elif len(fives) >= 3:
                    fives = fives[:-3]
                else:
                    return False
        return True

87. (LC 531) Lonely Pixel I
------------------------------
**Task**

Given an m x n picture consisting of black 'B' and white 'W' pixels, 
return the number of black lonely pixels.

A black lonely pixel is a character 'B' that located at a specific position where 
the same row and same column don't have any other black pixels.

My Note: lonely means no other B pixel on the ENTIRE row or column
(do not mistake with adjacency).

| Input: picture = [
|     ["W","W","B"],
|     ["W","B","W"],
|     ["B","W","W"]]
| Output: 3
| Explanation: All the three 'B's are black lonely pixels.

| Input: picture = [
|     ["B","B","B"],
|     ["B","B","W"],
|     ["B","B","B"]]
| Output: 0

**Solution**

| Idea:
| -2 traversals
| -1st: initiate arrays rows = [0,0,0], cols=[0,0,0], record +1 to that row, col if you met 'B'
| -2nd trav: if 'B', is the value at that row==1, at that col==1.

::

    def findLonelyPixel(self, picture: List[List[str]]) -> int:
        w, h = len(picture), len(picture[0])
        rows, cols = [0] * w, [0] * h
        for x in range(w):
            for y in range(h):
                if picture[x][y] == 'B':
                    rows[x] += 1        
                    cols[y] += 1        
        ans = 0
        for x in range(w):
            for y in range(h):
                if picture[x][y] == 'B':
                    if rows[x] == 1:     
                        if cols[y] == 1: 
                            ans += 1
        return ans

| **Explained**
|     ``rows, cols = [0] * w, [0] * h``
| Create records for all rows and columns. Initiate rows = [0,0,0], cols=[0,0,0]

|     rows[x] += 1        
|     cols[y] += 1        
| Again, we don't check the adjacent values for an item.
| If we encounter a "B", we record that on that row and on that column there is +1 of B.
| Hence if the value will be 2 when we check later, then that pixel is not lonely.

::

    ### My V2
    # (count(), and flip rows/columns with list(zip(*m)) )
    def f(m):
        w = len(m)
        h = len(m[0])
        lone = 0
        for i in range(w):
            for j in range(h):
                if m[i][j] == "B":
                    cnt1 = m[i].count("B")
                    cnt2 = list(zip(*m))[i].count("B")
                    if cnt1 + cnt2 == 2:
                        lone += 1
        return lone

    ### My V1
    # Idea:
    # -use count()
    # -use transposed version of the matrix

    def f(m):
        ans = 0
        transposed = list(zip(*m))
        for n in m:
            cnt = n.count("B")
            if cnt == 1:
                i = n.index("B")
                if transposed[i].count("B") == 1:
                    ans += 1
        return ans

    picture = [["W", "W", "B"], ["W", "B", "W"], ["B", "W", "W"]]
    print(f(picture))
    picture2 = [["B", "B", "B"], ["B", "B", "W"], ["B", "B", "B"]]
    print(f(picture2))
    picture3 = [["W", "W", "B"], ["W", "B", "B"], ["B", "W", "W"]]
    print(f(picture3))
    #3
    #0
    #1

88. (LC 674) Longest Continuous Increasing Subsequence
---------------------------------------------------------
`674. Longest Continuous Increasing Subsequence <https://leetcode.com/problems/longest-continuous-increasing-subsequence/>`_
Easy ::

    def longest_subsequence(a):
        lmax, lcur = 1, 1  # length max and current
        for i in range(1, len(a)):
            if a[i] > a[i - 1]:
                lcur += 1
            else:
                lmax = max(lmax, lcur)
                lcur = 1
        return max(lmax, lcur)

    nums = [1, 3, 5, 4, 7]
    print(longest_subsequence(nums)) #3

Note:
we calculate max length "lmax" only when we encounter a value that breaks the sequence.
So if there are several sequences, and the longest is the last one, we never calculate max for it, 
so we do that in the return statement. 

89. (LC 128) Longest Consecutive Sequence
--------------------------------------------
`128. Longest Consecutive Sequence <https://leetcode.com/problems/longest-consecutive-sequence/>`_
Medium ::

    ### Solution (my v).
    def f(a):
        d = {}
        ans = 0
        for n in a:
            d[n] = True
        for k in d:
            if k + 1 not in d:
                cnt = 0
                while k in d:
                    k -= 1
                    cnt += 1
                ans = max(ans, cnt)
        return ans

    ### V2
    def sequence(a):
        d = {}
        ans = 0
        for n in a:
            d[n] = True
        for n in a:
            if n - 1 not in d:
                seq = []
                while n in d:
                    seq.append(n)
                    n += 1
                ans = max(ans, len(seq))
        return ans

    nums = [100, 4, 200, 1, 3, 2]
    nums2 = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
    print(f(nums))  # 4
    print(f(nums2))  # 9

| Explained (V1):
|    if k + 1 not in d:

Start building sequence only starting from the topmost number. E.g. a=[2,3,100, 2,99, 4],
do not start building sequence from 2, because there is 2+1=3 in a. Start only from 4.

|    ans = max(ans, cnt)

Even though we are building the efficient way, for greatest n in sequence. 
There might be different unrelated sequences, like in a=[2,3,100, 2,99, 4].
So we build an inner count "cnt" for each sequence, and choose max.

90. (LC 229) Majority Element II
---------------------------------
`229. Majority Element II <https://leetcode.com/problems/majority-element-ii/>`_
Medium ::

    import collections
    def majority_element(a):
        times = len(a) / 3
        cnt = collections.Counter(a)
        return [x for x in cnt if cnt[x] > times]

    #OR
    def maj_elem(a):
        return [
            x for x in collections.Counter(a) if collections.Counter(a)[x] > (len(a) / 3)
        ]

    nums = [3, 2, 3]
    nums2 = [1]
    print(majority_element(nums))  # [3]
    print(majority_element(nums2))  # [1]

91. (LC 643) Maximum Average Subarray I
------------------------------------------
`643. Maximum Average Subarray I <https://leetcode.com/problems/maximum-average-subarray-i/>`_
Easy ::

    ### Solution 1
    # (more of a sliding window technique)
    class Solution:
        def findMaxAverage(self, nums: List[int], k: int) -> float:
            s = sum(nums[:k])
            ans = s
            for i in range(k, len(nums)):
                s += nums[i] - nums[i - k]
                ans = max(ans, s)
            return ans / k

| **Explained**
|     ``s += nums[i] - nums[i - k]``
| nums[i] adds 1 item to the right of initial slice nums[:k]
| But we also should take care to take away 1 item to the left of initial slice,
| to preserve len of slice s = k,
| this is exactly what "- nums[i-k]" does. 
| E.g. if k=4, first i=4, then [i-k]=4-4=0, so we subtract item at index [0].
| This achieves a sliding window effect.

::

    ### My V2
    # (Classic sliding window.)
    def f(a, k):
        lp = 0
        ans = 0
        for rp in range(len(a)):
            if rp - lp + 1 == k:
                res = sum(a[lp : rp + 1]) / k
                ans = max(ans, res)
                lp += 1
        return "{:.5f}".format(ans)

    ### My V
    # (More of a Python slicing technique.)
    def f(a, k):
        ans = 0
        for i in range(len(a) - (k - 1)):
            avg = sum(a[i : i + k]) / k
            ans = max(ans, avg)
        return ans

92. (LC 624) Maximum Distance in Arrays
------------------------------------------
| **Task**
| You are given m arrays, where each array is sorted in ascending order.

You can pick up two integers from two different arrays (each array picks one) and 
calculate the distance. We define the distance between two integers a and b to be 
their absolute difference \|a - b\|.

Return the maximum distance.

Example 1:
Input: arrays = [[1,2,3],[4,5],[1,2,3]]
Output: 4
Explanation: One way to reach the maximum distance 4 is to pick 1 in the first or third array and pick 5 in the second array.

::

    # Solution 1
    class Solution:
        def maxDistance(self, arrays: List[List[int]]) -> int:
            ans = 0
            mi, mx = arrays[0][0], arrays[0][-1]    #**1
            for arr in arrays[1:]:
                a, b = abs(arr[0] - mx), abs(arr[-1] - mi)  #**2
                ans = max(ans, a, b)
                mi = min(mi, arr[0])
                mx = max(mx, arr[-1])
            return ans

| **Keys**
| #**
| 1)Work with the fact that we are given SORTED arrays.
| 2)Make sure each time we pick min and max from two <different arrays>.

::

    # Solution 2
    class Solution:
        def maxDistance(self, arrays):
            """
            :type arrays: List[List[int]]
            :rtype: int
            """
            res, curMin, curMax = 0, 10000, -10000
            for a in arrays :
                res = max(res, max(a[-1]-curMin, curMax-a[0]))
                curMin, curMax = min(curMin, a[0]), max(curMax, a[-1])
            return res

93. (LC 670) Maximum Swap
------------------------------
`670. Maximum Swap <https://leetcode.com/problems/maximum-swap/>`_
Medium
::

    ### Solution 2 No tools. (LC accepted 73,35%, Time and Space O(log m))
    class Solution:
        def maximumSwap(self, num: int) -> int:
            s = list(str(num))
            n = len(s)
            d = list(range(n))
            for i in reversed(range(n-1)):
                if s[i] <= s[d[i + 1]]:
                    d[i] = d[i + 1]
            for i, j in enumerate(d):
                if s[i] < s[j]:
                    s[i], s[j] = s[j], s[i]
                    break
            return int(''.join(s))

| **In short**
| -two traversals: 
| 1)right to left; array d to record index of max value to the right of current
| 2)left to right traverse d, if index<value, swap 

| **Full elaboration**
| E.g. num = 98368 -> 98863
| -convert num to s=['9', '8', '3', '6', '8']
| -n=len(s)=5
| -d=[0,1,2,3,4]
| ---Traverse right to left: for i 3,2,1,0:

| i=3
| s=['9', '8', '3', '6', '8']
| d=[0,1,2,3,4]
| If s[i]<=max at right (max at right is stored in d)
|             if s[i] <= s[d[i + 1]]:
| 6 < 8
| d=[0,1,2,4,4]
 
| ---Traverse left to right d
| If s at index d index < s at index d value: swap
 
| s=['9', '8', <'3'>, '6', '8']
|     0    1     2
| d=[0,1,<4>,4,4]
| Here value at index 2 < value at index 4.

::

    ### My V (LC accepted: 50,90%)
    def swap(n):
        s = str(n)
        L = [int(n) for n in s]
        L2 = sorted(L)
        for i, num in enumerate(L):
            max_val = L2.pop()
            if num < max_val:
                ind = s.rindex(str(max_val))
                L[i], L[ind] = L[ind], L[i]
                break
        L3 = map(str, L)
        return int("".join(L3))

    print(swap(937))  # 973
    print(swap(2736))  # 7236
    print(swap(90979))  # 99970
    print(swap(93774))  # 97374 after using str.rindex(max_val) 97734

| Note the case when n=93774
| 1)we swap not the first integer (here at i=2) 
| 2)there are several occurrences of the max_value 7.
| We have to look for the rightmost index of the occurrence of max_value.
| So that after the swap the number is bigger.
 
| # Here str.rindex(value) was used.
| [Credit: https://stackoverflow.com/questions/6890170/how-to-find-the-last-occurrence-of-an-item-in-a-python-list ]

For the more general case you could use list.index on the reversed list:

>>> len(li) - 1 - li[::-1].index('a')
6

The slicing here creates a copy of the entire list. That's fine for short lists, 
but for the case where li is very large, efficiency can be better with a lazy approach::

    def list_rindex(li, x):
        for i in reversed(range(len(li))):
            if li[i] == x:
                return i
        raise ValueError("{} is not in list".format(x))

    # One-liner version:
    next(i for i in reversed(range(len(li))) if li[i] == 'a')

94. (LC 4) Median of Two Sorted Arrays
------------------------------------------
`4. Median of Two Sorted Arrays <https://leetcode.com/problems/median-of-two-sorted-arrays/>`_
Hard

The thing here is that the task asks for a solution o(log(n+m)).
It is easier to find solutions O(n+m). Several listed below.
While O(log(n+m) is more complicated, uses binary search.

::

    ### Solution 1
    # (Merging, i.e. here nums1+nums2 is O(n+m). 
    # Despite that, submitting to Leetcode gives a pretty good acceptance statistics - 
    # Time beats 99.28% of users with Python3. Memory beats 61%.)

    class Solution:
        def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:      
            concat = sorted(nums1+nums2) 
            if len(concat)%2 == 1:
                med = concat[int(len(concat)/2)]
            else:
                tot_len = int(len(concat)/2)
                med = (concat[tot_len-1]+concat[tot_len]) / 2
            return med

    ### Solution 2 (less efficient according to Leetcode)
    import statistics
    class Solution:
        def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
            nums3 = nums1 + nums2
            nums3 = sorted(nums3)
            return statistics.median(nums3)

    #OR (Runtime Beats80.53%of users with Python3, Memory Beats86.27%of users with Python3)
    import statistics
    class Solution:
        def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
            return statistics.median(sorted(nums1+nums2))


    ### Solution 3 (Leetcode editorial, O(n+m))
    class Solution:
        def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
            m, n = len(nums1), len(nums2)
            p1, p2 = 0, 0
            
            # Get the smaller value between nums1[p1] and nums2[p2].
            def get_min():
                nonlocal p1, p2
                if p1 < m and p2 < n:
                    if nums1[p1] < nums2[p2]:
                        ans = nums1[p1]   #**1
                        p1 += 1
                    else:
                        ans = nums2[p2]
                        p2 += 1
                elif p2 == n:
                    ans = nums1[p1]
                    p1 += 1
                else:
                    ans = nums2[p2]
                    p2 += 1
                return ans
            
            if (m + n) % 2 == 0:
                for _ in range((m + n) // 2 - 1):
                    _ = get_min()
                return (get_min() + get_min()) / 2
            else:
                for _ in range((m + n) // 2):  #**2
                    _ = get_min()
                return get_min()

| 1# Gets us values at current p1,p2 first (so at 0 initially), only then increments +1.
| 2# These loops
| for _ in range((m + n) // 2 - 1)
| Iterate that many times before the actual answer that we need.
| E.g. [1,2], [3]
| total len=3, mid =1
| for _ in range(1) -> iterates once.
| return get_min() -> iterates once the 2nd time
| So we get ans at p1=1 which nums1[p1]=value 2.

::

    ### Solution 3, My V (Leetcode checked, beets 75% of Py3 solutions both T&M.)
    class Solution:
        def findMedianSortedArrays(self, a1: List[int], a2: List[int]) -> float:
            n = len(a1)
            m = len(a2)
            total_len = m + n
            mid_index = total_len // 2
            p1, p2 = 0, 0

            # returns CURRENT min values at p1 or p2, then moves +1 p1 or p2
            def get_min():
                nonlocal p1, p2
                if p1 == n:
                    ans = a2[p2]
                    p2 += 1
                elif p2 == m:
                    ans = a1[p1]
                    p1 += 1
                else:  # both p1 and p2 are within array lens
                    if a1[p1] < a2[p2]:
                        ans = a1[p1]
                        p1 += 1
                    else:
                        ans = a2[p2]
                        p2 += 1
                return ans

            if total_len & 1:  # odd
                for _ in range(mid_index):
                    _ = get_min()
                return get_min()
            else:  # even
                for _ in range(mid_index - 1):
                    _ = get_min()
                return (get_min() + get_min()) / 2

95. (LC 921) Minimum Add to Make Parentheses Valid
-----------------------------------------------------
`921. Minimum Add to Make Parentheses Valid <https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/>`_
Medium

NOTE:
The simplistic examples of the task hide the fact that you should account for cases
like ')('. So it is not enough to count parentheses. We should keep track of them in order.
(i.e. use stack). ::

    class Solution(object):
        def minAddToMakeValid(self, s):
            """
            :type S: str
            :rtype: int
            """
            # left is length of stack
            left = right = 0
            
            for char in s:
                if char == '(':
                    left += 1
                else:
                    if left:
                        left -= 1
                    else:
                        right += 1
            return left + right

    class Solution:
        def minAddToMakeValid(self, s: str) -> int:
            stk = []
            for c in s:
                if c == ')' and stk and stk[-1] == '(':
                    stk.pop()
                else:
                    stk.append(c)
            return len(stk)

    ### My V (LC accepted)
    def make_valid(s):
        stack = []  # for '('
        cnt = 0     # for ')' without pair
        for c in s:
            if c == "(":
                stack.append(c)
            else:
                if len(stack) > 0:
                    stack.pop()
                else:
                    cnt += 1
        return cnt + len(stack)



















