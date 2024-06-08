Array Questions Part 5
======================
76. (LC 896) Monotonic Array
-------------------------------
`896. Monotonic Array <https://leetcode.com/problems/monotonic-array/>`_
Easy
::

    class Solution:
        def isMonotonic(self, nums: List[int]) -> bool:
            isIncr = isDecr = False
            for i, v in enumerate(nums[1:]):
                if v < nums[i]:
                    isIncr = True
                elif v > nums[i]:
                    isDecr = True
                if isIncr and isDecr:
                    return False
            return True

    class Solution:
        def isMonotonic(self, nums: List[int]) -> bool:
            incr = all(a <= b for a, b in pairwise(nums))
            decr = all(a >= b for a, b in pairwise(nums))
            return incr or decr

    ### My V1 (LC accepted, pretty efficient, beats T 90%, S 67%)
    def f(nums):
        if nums[0] > nums[len(nums) - 1]:
            up, down = False, True
        else:
            up, down = True, False
        for i in range(1, len(nums)):
            if up:
                if nums[i] < nums[i - 1]:
                    return False
            elif down:
                if nums[i] > nums[i - 1]:
                    return False
        return True

    ### My V2
    import itertools
    def monotonic2(a):
        L2 = [a <= b for a, b in itertools.pairwise(a)]
        L3 = [a >= b for a, b in itertools.pairwise(a)]
        if all(L2) or all(L3):
            return True
        return False

77. (LC 293) Flip Game
------------------------
You are playing a Flip Game with your friend.
You are given a string currentState that contains only '+' and '-'. 
You and your friend take turns to flip two consecutive "++" into "--". 
The game ends when a person can no longer make a move, and therefore the other person will be the winner.

Return all possible states of the string currentState after one valid move. 
You may return the answer in any order. If there is no valid move, return an empty list [].

Example 1:
Input: currentState = "++++"
Output: ["--++","+--+","++--"]

Example 2:
Input: currentState = "+"
Output: []

::

    class Solution:
        def generatePossibleNextMoves(self, currentState: str) -> List[str]:
            s = list(currentState)
            ans = []
            for i, c in enumerate(s[:-1]): 
                if c == "+" and s[i + 1] == "+":
                    s[i] = s[i + 1] = "-"
                    ans.append("".join(s))
                    s[i] = s[i + 1] = "+"
            return ans

    # With comments
    class Solution:
        def generatePossibleNextMoves(self, currentState: str) -> List[str]:
            s = list(currentState)
            ans = []
            for i, c in enumerate(s[:-1]): #len(s)-1, because we will do i+1
                if c == "+" and s[i + 1] == "+":
                    s[i] = s[i + 1] = "-"   #change in place to --
                    ans.append("".join(s))  #add to ans changed and the items we don't know
                    s[i] = s[i + 1] = "+"   #change back to ++
            return ans

    ### My V2
    def f(s):
        ans = []
        for i in range(len(s) - 1):
            if s[i : i + 2] == "++":
                ans.append([s[:i] + "--" + s[i + 2 :]])
        return ans

    ### My V1
    def f(s):
        ans = []
        L = list(s)
        for i in range(len(L) - 1):
            if L[i] == "+" and L[i + 1] == "+":
                ans_c = L[:i] + ["-"] + ["-"] + L[i + 2 :]
                ans.append("".join(ans_c))
        return ans

78. (LC 832) Flipping an Image
--------------------------------
`832. Flipping an Image <https://leetcode.com/problems/flipping-an-image/>`_
Easy

**Solutions** ::

    # My V
    def flip_img(m):
        for i in range(len(m)):
            m[i] = m[i][::-1]
            for j in range(len(m[i])):
                m[i][j] ^= 1
        return m

    ### Solution 1
    class Solution:
        def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
            n = len(image)
            for row in image:
                i, j = 0, n - 1
                while i < j:
                    if row[i] == row[j]:
                        row[i] ^= 1
                        row[j] ^= 1
                    i, j = i + 1, j - 1
                if i == j:
                    row[i] ^= 1
            return image

| **Explained**
| row - is each inner list of the image, image - is list of lists
 
|     ``for row in image:``
| -We traverse each inner list.
 
|     ``i, j = 0, n - 1``
|       ``while i < j:``
| -Comparing left most, right most items. Moving towards the list center.
 
|     ``if row[i] == row[j]:``
| -If items are the same, e.g. [1,0,0,1], 1) it means there is no need to swap them
| (swapping 1 with 1), 2)then we can straight away flip them, i.e. do the 2nd step. 
 
|     ``i, j = i + 1, j - 1``
| -Keep moving towards the center
| #**
 
|     ``if i == j:``
|         ``row[i] ^= 1``
| -If a list has an odd number of items, then we look at the center item when i==j
| (e.g. [1,1,0]).
| We flip it no matter what.

#**It might look like we are missing the case when items at i, j are different, e.g.
[1,1,0] we neither swap nor flip them. But there is no need! 1 and 0 swapped and flipped
is still 1 and 0. 
We need to flip only if items are the same, 1,1 or 0,0.

::

    ### Solution 2
    (Unlike solution 1, here we swap/reverse all, flip all, even if there is no need, 
    we don't check the actual values.)
    class Solution:
        def flipAndInvertImage(self, A):
            """
            :type A: List[List[int]]
            :rtype: List[List[int]]
            """
            rows = len(A)
            cols = len(A[0])
            for row in range(rows):
                A[row] = A[row][::-1]   #reverse all
                for col in range(cols):
                    A[row][col] ^= 1    #flip all
            return A


| Also there is no need for different rows and col variables, as we are told that
| n == image.length
| n == image[i].length
| So row=col=n
| (Num of lists = items in inner list. E.g. 3 lists, 3 items in each list.)

79. (LC 48) Rotate Image
--------------------------
`48. Rotate Image <https://leetcode.com/problems/rotate-image/>`_
Medium

::

    ### Solution 1
    def rotate(matrix):
        return [list(reversed(x)) for x in zip(*matrix)]

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(rotate(matrix)) #[[7, 4, 1], [8, 5, 2], [9, 6, 3]]

.. admonition:: Transpose vs. rotate 90

    There is a difference.
    To transpose a matrix (for rows to become columns).
    The transposed matrix is not rotated but mirrored on the diagonal (i.e. columns and rows are swapped). 

    Compare and visualize::

        # original-transposed-rotated
        # 1 2 3   1 4 7   7 4 1
        # 4 5 6   2 5 8   8 5 2
        # 7 8 9   3 6 9   9 6 3 

    | So if the task was to transpose, we would do:
    | ``list(zip(*matrix))``
    | To rotate, note the Visualization above, we just need to reverse the rows in transposed result.
    | ``[list(reversed(x)) for x in zip(*matrix)]``

::

    ### Solution 2
    class Solution:
        def rotate(self, matrix: List[List[int]]) -> None:
            n = len(matrix)
            for i in range(n >> 1):
                for j in range(n):
                    matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
            for i in range(n):
                for j in range(i):
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

*Explanation*

According to the requirements of the problem, we actually need to rotate 
``matrix[i][j]`` to ``matrix[j][n - i - 1]``.
We can first flip the matrix upside down, that is, swap ``matrix[i][j]`` and 
``matrix[n - i - 1][j]``, and then flip the matrix along the main diagonal, that is, 
swap ``matrix[i][j]`` and ``matrix[j][i]``. This way we can rotate ``matrix[i][j]`` to ``matrix[j][n - i - 1]``.

Time O(N**2), N is the length of the matrix, space O(1).

80. (LC 334) Increasing Triplet Subsequence
----------------------------------------------
`334. Increasing Triplet Subsequence <https://leetcode.com/problems/increasing-triplet-subsequence/>`_
Medium

::

    ### Solution 1
    def increasingTriplet(nums):
            first = float('inf')
            second = float('inf')
            for num in nums:
                if num <= first:     # min num
                    first = num
                elif num <= second:  # 2nd min num, i.e. mid
                    second = num
                else:                # 3rd min num, i.e. max
                    return True      
            return False

    ### Solution 1 My V1 (LC accepted)
    def f(nums):
        min1 = float("inf")
        mid = min1
        for i in range(len(nums)):
            if nums[i] <= min1:   #IMPORTANT < or =
                min1 = nums[i]    #do not be tempted to do also mid=min1
            elif nums[i] <= mid:
                mid = nums[i]
            else:
                return True
        return False

    ### Solution 2
    class Solution:
        def increasingTriplet(self, nums: List[int]) -> bool:
            mi, mid = inf, inf
            for num in nums:
                if num > mid:    # i.e we found max, triplet is complete
                    return True
                if num <= mi:
                    mi = num
                else:
                    mid = num
            return False

81. (LC 56) Merge Intervals
------------------------------
`56. Merge Intervals <https://leetcode.com/problems/merge-intervals/>`_
Medium

| *Side note. Sorting list of lists.*
| Sorts on the first element.

>>> L3
[[2, 1], [1, 3]]
>>> L3.sort()
>>> L3
[[1, 3], [2, 1]]


| **Task gotchas**
| -Task examples don't make it clear that you might be given unsorted intervals
| [[1,4],[0,4]]
| Or 
| [[1,4],[2,3]]
| So sort the input first.

**Solution** ::

    class Solution:
        def merge(self, intervals: List[List[int]]) -> List[List[int]]:
            intervals.sort()
            ans = [intervals[0]]
            for s, e in intervals[1:]:
                if ans[-1][1] < s:
                    ans.append([s, e])
                else:
                    ans[-1][1] = max(ans[-1][1], e)
            return ans

    ### My V (LC accepted 85, 32%)
    class Solution:
        def merge(self, intervals: List[List[int]]) -> List[List[int]]:
            intervals.sort()
            new = [intervals[0]]
            for start, end in intervals:
                if start <= new[-1][1]:
                    if end > new[-1][1]:
                        new[-1][1] = end
                else:
                    new.append([start, end])
            return new

| **Example**
| Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
| Output: [[1,6],[8,10],[15,18]]
| Merging intervals [1,3] and [2,6] into [1,6].
 
|     ans = [intervals[0]]
| Put the first interval into answer.
| ans = [[1,3]]
 
|     for s, e in intervals[1:]:
| For all the rest intervals, look at start, end for each.
 
|     if ans[-1][1] < s:
|         ans.append([s, e])

If end of the last interval in answer, here [1,3], end=3 is < start of the interval
we are looking at, here [2,6], s=2. So if next interval starts later than the previous
ends, we would've appended that WHOLE interval to the answer. 

|    else:
|        ans[-1][1] = max(ans[-1][1], e)

If [2,6] starts within [1,3], then we don't append the whole interval [2,6]
but instead check if it finishes within interval already in answer, i.e. [1,3].
Check WHICH END IS GREATER, make that new end for existing interval, [1,3] -> [1,6]

82. (LC 57) Insert Interval
------------------------------
`57. Insert Interval <https://leetcode.com/problems/insert-interval/>`_
Medium

**Logic**
We append the given interval to the list of intervals, then call the merge()
method on it. merge() will first sort the intervals, us having added a new interval,
then perform the unchanged logic of merge from the previous question.

**Solution** ::

    class Solution:
        def insert(
            self, intervals: List[List[int]], newInterval: List[int]
        ) -> List[List[int]]:
            def merge(intervals: List[List[int]]) -> List[List[int]]:
                intervals.sort()
                ans = [intervals[0]]
                for s, e in intervals[1:]:
                    if ans[-1][1] < s:
                        ans.append([s, e])
                    else:
                        ans[-1][1] = max(ans[-1][1], e)
                return ans

            intervals.append(newInterval)
            return merge(intervals)

*My attempt (LC accepted 60,65%)*
I insert the interval into the right spot, keeping the intervals sorted. ::

    class Solution:
        def insert(self, m: List[List[int]], inter: List[int]) -> List[List[int]]:
            if len(m)==0:
                return [inter]
            # insert new interval
            for i in range(len(m)):
                if inter[0] < m[i][0]:
                    m = m[:i] + [inter] + m[i:]
                    break
                elif i==len(m)-1:
                    m.append(inter)

            # merge intervals
            new_m = []
            new_m.append(m[0])
            for j in range(1, len(m)):
                #omit interval completely, it is within prev interval
                if new_m[-1][0] <= m[j][0] and new_m[-1][1] >= m[j][1]:
                    continue
                #append interval completely
                elif new_m[-1][1] < m[j][0]:
                    new_m.append(m[j])
                #intervals intersect
                elif new_m[-1][1] >= m[j][0]:
                    new_m[-1][1] = m[j][1]
            return new_m


83. (LC 215) Kth Largest Element in an Array
-----------------------------------------------
`215. Kth Largest Element in an Array <https://leetcode.com/problems/kth-largest-element-in-an-array/>`_
Medium

The task asks not to use sorting.

**Sort** ::

    def kth(a, k):
        return sorted(a)[-k]

    def kth2(a, k):
        a.sort(reverse=True)
        return a[k - 1]

    class Solution1:
        def findKthLargest(self, nums: List[int], k: int) -> int:
            nums.sort()
            return nums[len(nums) - k]

**No sorting** ::

    # Idea : remove max 
    def findKthLargest(nums, k):
        for i in range(k - 1):
            nums.remove(max(nums))
        return max(nums)

    # The same my V
    def f2(a, k):
        for _ in range(k):
            m = max(a)
            a.remove(m)
        return m

**Heap**

Complexity:
Making a heap is O(N). Popping from a heap one time is logN, so KlogN for k pops.
Overall = N+KlogN.
Which is a bit better than sorting with NlogN, depending on k. ::

    from heapq import heapify, heappop
    def f4(a, k):
        max_heap = [n * (-1) for n in a]  #alt. -int(n) for..
        heapify(max_heap)
        for _ in range(k):
            ans = heappop(max_heap)
        return ans * (-1)

**Quickselect**

# Complexity [:ref:`10 <ref-label>`]
Quickselect average case O(N), worst case O(N**2).
Compare with quicksort avg. O(NlogN). Because quicksort performs the search on both 
sides of the partitioned array. In quick select we search only in one partition 
(because we compare pivot index with target k and know on which one side we have to search.)
More precisely, its going to be n + n/2 + n/4 .. infinite series = 2n = O(n)

What is the worst case.
When each time we pick a pivot, it happens to be the greatest number, landing at right side (_,_,_,_,P). 
Meaning we would decrease our search not by half, but only by -1 item.
Ending up with O(N**2)

We don't have to sort the entire array to give the answer.

Quickselect can be thought of as a hybrid of Quicksort and binary search.
Like Quicksort, Quickselect relies on partitioning.

After a partition, the pivot value ends up in the appropriate spot in the
array. So if we end up with pivot at 5th place, then this is our
5th lowest value in the array. For good!

==> So after each partition, we check where the pivot ended up, i.e. we check pivot's index.
If it is not entirely what we were looking for, then we keep partitioning
only that halve of array, to the left or right of pivot, depending whether
we were looking for the Nth lowest or highest value. 

Solution [:ref:`12 <ref-label>`]::

    class Solution:
        def partition(self, a, left_pointer, right_pointer):
            pivot_index = right_pointer
            pivot = a[pivot_index]
            right_pointer -= 1
            while True:
                while a[left_pointer] < pivot:
                    left_pointer += 1
                while a[right_pointer] > pivot:
                    right_pointer -= 1
                if left_pointer >= right_pointer:
                    break
                else:
                    a[left_pointer], a[right_pointer] = (
                        a[right_pointer],
                        a[left_pointer],
                    )
                    left_pointer += 1
            a[left_pointer], a[pivot_index] = (
                a[pivot_index],
                a[left_pointer],
            )
            return left_pointer

        def quickselect(self, a, k, left_index, right_index):
            k = len(a) - k  # kth lowest equivalent of kth largest
            if right_index - left_index <= 0:
                return a[left_index]
            pivot_index = self.partition(a, left_index, right_index)
            if k < pivot_index:  # search left side       #2
                return self.quickselect(a, k, left_index, pivot_index - 1)
            elif k > pivot_index:  # search right side
                return self.quickselect(a, k, pivot_index + 1, right_index)
            else:
                return a[pivot_index]

    nums = [3, 2, 3, 1, 2, 4, 5, 5, 6]
    k = 4
    S = Solution()
    # print(S.quickselect(nums, k, 0, len(nums) - 1))  # 4
    nums2 = [3, 2, 1, 5, 6, 4]
    k2 = 2
    print(S.quickselect(nums2, k2, 0, len(nums2) - 1))  #5

**Explained**

#1 The adaptation line.
Initially the algorithm is to search for the kth_lowest.
The sorting is in normal, lowest to highest order. [1,2,3,4]
Array 6 items long, 4th largest is 6-4=2 normal index. 

| # kth_lowest_value
| Equivalent to the index
| # if right_index - left_index <= 0:
| Base case
| else # if kth_lowest_value == pivot_index

84. (LC 747) Largest Number At Least Twice of Others
-------------------------------------------------------
`747. Largest Number At Least Twice of Others <https://leetcode.com/problems/largest-number-at-least-twice-of-others/>`_
Easy ::

    ### Solution 1
    class Solution:
        def dominantIndex(self, nums: List[int]) -> int:
            mx = mid = 0
            ans = -1
            for i, v in enumerate(nums):
                if v > mx:
                    mid, mx = mx, v
                    ans = i
                elif v > mid:
                    mid = v
            return ans if mx >= 2 * mid else -1

    ### My V1
    #(perhaps .index() lookup is not as efficient as using enumerate)

    def twice_as(a):
        max1 = -float("inf")
        for n in a:
            if n > max1:
                max2 = max1
                max1 = n
            elif n > max2:
                max2 = n
        if (max1 / max2) >= 2:
            return a.index(max1)
        return -1

    nums1 = [3, 6, 1, 0]
    nums2 = [1, 2, 3, 4]
    print(twice_as(nums1))  # 1
    print(twice_as(nums2))  # -1

    ### My V2 - heap
    (Uses extra space)
    from heapq import heapify, heappop
    def f(a):
        nums = [-int(x) for x in a]
        heapify(nums)
        max1 = heappop(nums)
        max2 = heappop(nums)
        if max1 / max2 >= 2:
            return a.index(-max1)
        return -1

    ### My V3 - max
    def f(a):
        max1 = max(a)
        i = a.index(max1)
        a.remove(max1)
        max2 = max(a)
        if max1 / max2 >= 2:
            return i
        return -1

    ### My V4
    def f(a):
        m = max(a)
        for n in a:
            if n != m and n != 0:
                if m / n < 2:
                    return -1
        return a.index(m)

85. (LC 949) Largest Time for Given Digits
----------------------------------------------
`LC 949. Largest Time for Given Digits <https://leetcode.com/problems/largest-time-for-given-digits/>`_
Medium

| 2 main versions:
| 1-make all possible time strings from the given 4 digits. Use itertools.permutations()
| 2-greedy

::

    # My V2
    import itertools as it
    def latest_time(a):
        valid_time = [
            x
            for x in it.permutations(a, 4)
            if ((x[0] * 10 + x[1]) < 24) and ((x[2] * 10 + x[3]) < 60)
        ]
        if valid_time:
            max_time = max(valid_time)  # we can find max of tuples without any conversions
            return "%02d:%02d" % (
                max_time[0] * 10 + max_time[1],
                max_time[2] * 10 + max_time[3],
            )
        return ""

    arr = [1, 2, 3, 4]
    arr2 = [1, 2]
    arr3 = [5, 5, 6, 6]
    print(latest_time(arr))  #23:41
    print(latest_time(arr2)) #''
    print(latest_time(arr3)) #''

    ### My V1
    import itertools

    def latest_time(a):
        L = list(itertools.permutations(a))
        L2 = L[:]
        # print(L)
        for i in L:
            if (i[0] > 2) or (i[0] == 2 and i[1] >= 4) or (i[2] >= 6):
            #OR if i[0]*10 + i[1] < 24 and i[2]*10 + i[3] < 60:
                L2.remove(i)
        # print(L2)
        if L2:
            mt = max(L2)  # max time
            return f"{mt[0]}{mt[1]}:{mt[2]}{mt[3]}"
        return ""

    arr1 = [1, 2, 3, 4]
    arr2 = [5, 5, 5, 5]
    print(latest_time(arr1))  # 23:41
    print(latest_time(arr2))  # ''

    ### My V (greedy)
    # (Probably needs more testing)
    def latest_time(a):
        h1 = h2 = m1 = m2 = 0
        for n in a:
            if h1 <= n and n < 3:
                if h2 < h1:
                    h2 = h1
                h1 = n
            elif h2 < n and (h1 * 10 + n) <= 23:
                if m1 < h2:
                    m1 = h2
                h2 = n
            elif m1 < n and (n * 10 + m1) < 60:
                if m2 < m1:
                    m2 = m1
                m1 = n
            elif m2 < n:
                m2 = n
            else:
                return ""
        return f"{h1}{h2}:{m1}{m2}"

    arr = [1, 2, 3, 4]
    print(latest_time(arr))
    arr2 = [5, 5, 5, 5]
    print(latest_time(arr2))
    arr3 = [0, 9, 9, 5]
    print(latest_time(arr3))
    #23:41
    #""
    #09:59


