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










