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







