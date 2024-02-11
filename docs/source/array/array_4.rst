Array Questions Part 4
======================
66. (LC 1109) Corporate Flight Bookings
----------------------------------------
1109. `Corporate Flight Bookings <https://leetcode.com/problems/corporate-flight-bookings/>`_
*Medium*

**My version 1** ::

    def corpFlightBookings(bookings: List[List[int]], n: int) -> List[int]:
        ans = [0] * n
        for first, last, seats in bookings:
            ans[first - 1] += seats
            while last > first:
                ans[last-1] += seats
                last -= 1
        return ans

    bookings = [[1,2,10],[2,3,20],[2,5,25]]
    n = 5
    print(corpFlightBookings(bookings, n))  # [10, 55, 45, 25, 25]

    ### V2
    def f2(data, n):
        ans = [0] * n
        for first, last, seats in data:
            for i in range(first - 1, last): #again indexing in our ans is first-1
                ans[i] += seats
        return ans

**Solution** ::

    def corpFlightBookings(bookings: List[List[int]], n: int) -> List[int]:
        ans = [0] * n
        for first, last, seats in bookings:
            ans[first - 1] += seats
            if last < n:
                ans[last] -= seats
        return list(itertools.accumulate(ans))

    bookings = [[1,2,10],[2,3,20],[2,5,25]]
    n = 5
    print(corpFlightBookings(bookings, n))  # [10, 55, 45, 25, 25]

| **Explained**
| ans[first - 1] because we initiate ans = [0,0,0,0,0] where indexing starts at 0, while in our bookings indexing start at 1.

| # Seeing inside the loop
| 10, 0, -10, 0, 0
| 10,20, -10,-20, 0
| 10,45, -10,-20, 0 
| Notes: 45=20+25, we don't have at i=5, -25, because the condition is if last < n.
| Accumulation on ans array gives [10, 55, 45, 25, 25]

67. (LC 697) Degree of an Array
----------------------------------
697. `Degree of an Array <https://leetcode.com/problems/degree-of-an-array/>`_

| NOTE 
| The answer in the second example might seem crazy, until  you realize that
| you are asked to give the shortest CONTIGUOUS subarray.
| 
| KEYS
| -Use cnt=collections.Counter(array) to count occurrences for all nums
| -degree=max(cnt.values())
| -Recognize that there can be more than one number with the highest degree.
| - make 2 dicts left, right = {}, {}
| {number: index}.
| left - records when you encounter a number for the first time (the leftmost encounter).
| right - the rightmost encounter of a number.
| E.g. nums = [1,2,2,3,1]
| left = {1:0, 2:1, 3:3}
| right = {1:4, 2:2, 3:3}
| For that use: enumerate(array), 
|     left.setdefault(n, i)
|     right[n] = i
| -For each num with highest degree calculate subarray len via: right[num] - left[num] + 1

::

    ### Solution 1 (left, right dicts)
    import collections
    class Solution(object):
        def findShortestSubArray(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            counts = collections.Counter(nums)
            left, right = {}, {}
            for i, num in enumerate(nums):
                left.setdefault(num, i)
                right[num] = i
            degree = max(counts.values())
            return min(right[num]-left[num]+1 \
                    for num in counts.keys() \
                    if counts[num] == degree)

    ### My remake of S1, left, right dicts (Final efficiency+readability balance)
    import collections
    def f(a):
        cnt = collections.Counter(a)
        max_cnt = max(cnt.values())  # find most frequent count, e.g. 2 times
        max_nums = [
            k for k, v in cnt.items() if v == max_cnt
        ]  # nums for most freq count, e.g. [1,2]
        left, right = {}, {}
        for i, n in enumerate(a):
            left.setdefault(n, i)
            right[n] = i
        lengths = []
        for num in max_nums:
            length = right[num] - left[num] + 1
            lengths.append(length)
        return min(lengths)

    nums = [1, 2, 2, 3, 1]
    nums2 = [1, 2, 2, 3, 1, 4, 2]
    print(f(nums))  # 2
    print(f(nums2))  # 6

    ### My V (indexing)
    import collections
    def f(a):
        cnt = collections.Counter(a)
        max_cnt = max(cnt.values())  # find most frequent count, e.g. 2 times
        max_nums = [
            k for k, v in cnt.items() if v == max_cnt
        ]  # nums for most freq count, e.g. [1,2]
        subarrays = []
        for n in max_nums:  # for each of the most freq numbers
            start = a.index(n)
            end = len(a) - a[::-1].index(n) - 1
            subarrays.append(len(a[start : end + 1]))  # append len of subarray

        return min(subarrays)

| **Logic to solution 1**
| Iterate through the array, keep two dics: left and right, {number: index}.
| left - records when you encounter a number for the first time (the leftmost encounter).
| right - the rightmost encounter of a number.
| E.g. nums = [1,2,2,3,1]
| left = {1:0, 2:1, 3:3}
| right = {1:4, 2:2, 3:3}

| degree - max in collections.Counter(nums), here degree=2 
| Number we encounter most of the time. (To satisfy the first condition of the task.)

| return min(right[num]-left[num]+1 \\
|             for num in counts.keys() \\
|             if counts[num] == degree)
| E.g. nums = [1,2,2,3,1], counts = {1:2, 2:2, 3:1}
| 1)For keys in counts - just all our unique numbers.
| 2)if, i.e. look at only those that we encounter most of the time, here just 
| numbers 1,2, their values in counter = degree = 2
| 3)look up indexes for these numbers in right and left, the difference will tell us
| how far apart they are. Choose the minimum.
| Here we calculate for num=1, num=2
| r[1] - l[1] +1 = 4-0+1=5
| r[2] - l[2] +1 = 2-2+1=2
| We got our winner, the answer is 2.

**Tools** 
How do we make the 'left' dictionary. To record only the first time we encounter
a number.

``dict.setdefault(key[, default])``
If key is in the dictionary, return its value. If not, insert key with a value of 
default and return default. (default defaults to None.)

>>> d = {30:45}
>>> d.setdefault(25, 50)  #new key
50
>>> d
{30: 45, 25: 50}    #OK, sets new key with value
>>> d.setdefault(25, 60)   #key already in dict
50
>>> d
{30: 45, 25: 50}    #Not OK, keep the old value 

::

    ### Solution with "no tricks" (the least efficient for that)
    import collections
    def f(a):
        cnt = collections.Counter(a)
        values = []   # Because there can be several values with the same degree
        degree = 0
        for v in cnt.values():  #OR degree=max(cnt.values())
            if v > degree:
                degree = v
        [values.append(k) for k, v in cnt.items() if v == degree]
        ans = []
        for value in values:
            subarray_len = 0
            for n in a:
                if n == value:
                    subarray_len += 1
                    degree -= 1
                elif n != value and degree > 0 and subarray_len > 0:
                    subarray_len += 1
            ans.append(subarray_len)
        return min(ans)

68. (LC 498) Diagonal Traverse
--------------------------------
`498. Diagonal Traverse <https://leetcode.com/problems/diagonal-traverse/>`_
Medium

| Main points
| Keep in mind:
| i j
| 0 0 01 02
| So i is width, first index, j is height, 2nd index.

**Solution**::

    class Solution:
        def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
            m, n = len(mat), len(mat[0])  #n is matrix width,
            ans = []
            for k in range(m + n - 1):
                t = []
                i = 0 if k < n else k - n + 1  #after k>n, i grows +1
                j = k if k < n else n - 1      #after k>n, j will be static, =2
                while i < m and j >= 0:
                    t.append(mat[i][j])
                    i += 1
                    j -= 1
                if k % 2 == 0:
                    t = t[::-1]
                ans.extend(t)
            return ans

| **Explained**
| # m, n = matrix width, length
| # k is the number of diagonals we can make in the matrix.
|     for k in range(m + n - 1):
| E.g. in a 3x3 matrix we can make m+n-1=5 diagonals. Take a look:
| 1 2 3
| 4 5 6
| 7 8 9
| So our main loop is k (0, 5).
| # t is each diagonal, e.g. here t=[1], t=[2,4] etc
| # We are going to collect our diagonals all in one direction (top-down), 
| reverse if k is even (0,2,4)
| if k % 2 == 0:
|     t = t[::-1]
| #
|     i = 0 if k < n else k - n + 1
| Diagonals start at row index=0, until we reach the end of row 0, i.e. n=3, 
| when k > n, our 4th (k=3) diagonal cannot start at i=0, which has only 3 elements. 
| Then we start on the next row i+1, i.e. k-n+1 (e.g. 3-3+1=1=i,4-3+1=2=i)













