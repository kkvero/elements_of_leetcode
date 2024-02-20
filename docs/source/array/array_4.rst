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

69. (LC 888) Fair Candy Swap
-----------------------------
`LC 888 Fair Candy Swap <https://leetcode.com/problems/fair-candy-swap/>`_
Easy

| Example
| Input: aliceSizes = [1,2], bobSizes = [2,3]
| Output: [1,2]
 
| In short.
| The goal - Alice and Bob should have the same number of candies.
| Alice has 1 candy in box 1, 2 candies in box 2. [1,2]
| Bob has 2 candies in box 1, 3 candies in box 2. [2,3]
| Output  [1,2] is, i=0 is how many candies Alice should give to Bob, 
| i=1 i how many candies Bob should give to Alice 
| so that they both have the same number of candies.
| (If box has 2 candies, box is not divisible, both candies should be given.)

**Solutions** ::

    ### V3
    def candy_swap(A, B):
        mid = int((sum(A + B)) / 2)
        for n in A:
            pair = mid - (sum(A) - n)
            if pair in B:
                return [n, pair]

    ### V0
    class Solution:
        def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
            diff = (sum(aliceSizes) - sum(bobSizes)) >> 1 
            s = set(bobSizes)
            for a in aliceSizes:
                target = a - diff
                if target in s:
                    return [a, target]

**Explained**

``diff = (sum(aliceSizes) - sum(bobSizes)) >> 1``
We are ensured that there is a solution, means the total number of candies both
kids have is an even number. Means it is divisible by 2. The most efficient way to divide
by 2 is to remove one LSB from the even number (LSB in even numbers is always 0).
Removing LSB 0 amounts to dividing by 2. 

>>> bin(6)
'0b110'
>>> 6 >> 1
3
>>> bin(3)
'0b11'

diff - is rather the num of candies each kid will have, when they both have the same num.

**More solutions** ::

    ### V1
    class Solution(object):
        def fairCandySwap(self, A, B):
            """
            :type A: List[int]
            :type B: List[int]
            :rtype: List[int]
            """
            sum_A, sum_B, set_B = sum(A), sum(B), set(B)
            target = (sum_A + sum_B) / 2
            for a in A:
                b = target - (sum_A - a)
                if b >= 1 and b <= 100000 and b in set_B:
                    return [a, b]

    ### V2
    class Solution(object):
        def fairCandySwap(self, A, B):
            """
            :type A: List[int]
            :type B: List[int]
            :rtype: List[int]
            """
            diff = (sum(A)-sum(B))//2
            setA = set(A)
            for b in set(B):
                if diff+b in setA:
                    return [diff+b, b]
            return []

70. (LC 442) Find All Duplicates in an Array
----------------------------------------------
`442. Find All Duplicates in an Array <https://leetcode.com/problems/find-all-duplicates-in-an-array/>`_
Medium ::

    ### V1
    class Solution(object):
        def findDuplicates(self, nums):
            """
            :type nums: List[int]
            :rtype: List[int]
            """
            ans = []
            for n in nums:
                if nums[abs(n) - 1] < 0:
                    ans.append(abs(n))
                else:
                    nums[abs(n) - 1] *= -1
            return ans

| # Logic - mark met elements at index of the value.
| (It uses the fact that the constraint is our array values are in range [1, n],
| where n == nums.length, 
| i.e. 1 <= nums[i] <= n ).
| That's like a straight hint that all values are also valid indices for that array
| (more precisely i=value-1).
| We iterate through the array numbers. 
| We lookup elements at index of the current value.
| A = [4, 3, 2, 7, 8, 2, 3, 1]
| n=4, A[n-1]=7
| We actually don't care what the value is at index A[n-1].
| It's just that if there are 2 equal items in A, then we will be SENT to the same
| index TWICE, its like THE SAME ADDRESS.
| So what do we do, when we are sent to an address, we mark that we've been there.
| How: just value at A[n-1] =* (-1).
| So really the first step is to check - have we been there? (is A[n-1] < 0).
| Otherwise mark we've been there.
| (If we've been there just once, then A[n-1] < 0, if we've been there twice, or not once yet, then value is > 0).

::

    ### V0
    Note, this doesn't satisfy O(1) space of the task
    from collections import Counter
    class Solution(object):
        def findDuplicates(self, nums):
            return [elem for elem, count in Counter(nums).items() if count == 2]

71. (LC 448) Find All Numbers Disappeared in an Array
--------------------------------------------------------
`448. Find All Numbers Disappeared in an Array <https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/>`_
Easy ::

    ### V1
    def findDisappearedNumbers2(nums):
        return list(set(range(1, len(nums) + 1)) - set(nums))

    ### V2
    def findDisappearedNumbers(self, nums):
        n = set(nums)
        new = set(range(1, len(nums) + 1))
        return list(new - n)   # works on "set" data structure 

    ### V my
    def find_missing(a):
        ans = list(range(1, len(a) + 1))
        for i in a:
            if i in ans:
                ans.remove(i)
        return ans

72. (LC 724) Find Pivot Index
--------------------------------
`724. Find Pivot Index <https://leetcode.com/problems/find-pivot-index/>`_
Easy ::

    # V1
    class Solution:
        def pivotIndex(self, nums: List[int]) -> int:
            left, right = 0, sum(nums)
            for i, x in enumerate(nums):
                right -= x         #(right=(right - value_of_pivot))
                if left == right:
                    return i
                left += x         #(left + value_of_pivot)
            return -1

Note:
    ``right -= x``

| When testing if i is pivot.
| We subtract value at i (at pivot) from the sum on the right.
| But we do not yet add value at pivot to left.
| We first test if left == right,
| only then add value at i (possible pivot) to left: left += x
| E.g. [1,7,3,5]. When looking at 7, we subtract value 7 from right, but we do not yet add it to left,
| because at pivot 7 we compare sums "1" and "3+5".

::

    # V2
    class Solution(object):
        def pivotIndex(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            sums = sum(nums)
            total = 0
            for x, n in enumerate(nums):
                if sums - n == 2 * total: 
                    return x
                total += n
            return -1

    # My V1
    def find_pivot(a):
        a = [0] + a + [0]
        p = -1
        for i in range(1, len(a) - 1):
            sl = sum(a[0:i])
            sr = sum(a[i + 1 : len(a)])
            if sl == sr:
                p = i - 1
        return p

    nums = [1, 7, 3, 6, 5, 6]  #3
    nums2 = [2, 1, -1]         #0
    print(find_pivot(nums))
    print(find_pivot(nums2))

    # V2
    def find_pivot(a):
        a = [0] + a + [0]
        for i in range(1, len(a) - 1):
            if sum(a[0:i]) == sum(a[(i + 1) : len(a)]):
                return i - 1
        return -1

    nums = [1, 7, 3, 6, 5, 6]
    print(find_pivot(nums)) #3

    # V3
    def f(a):
        s = sum(a)
        for i in range(len(a)):
            if i == 0:
                s_left = 0
            else:
                s_left = sum(a[0:i])
            if i == len(a) - 1:
                s_right = 0
            else:
                s_right = s - s_left - a[i]
            if s_left == s_right:
                return i
        return -1

73. (LC 1275) Find Winner on a Tic Tac Toe Game
----------------------------------------------------
`1275. Find Winner on a Tic Tac Toe Game <https://leetcode.com/problems/find-winner-on-a-tic-tac-toe-game/>`_
Easy

.. admonition:: A catch in creating multidimensional arrays

    This type of declaration will not create m*n spaces in memory; rather, only one integer 
    will be created and referenced by each element of the inner list.

    >>> grid = [[""] * 3] * 3
    >>> grid
    [['', '', ''], ['', '', ''], ['', '', '']]

    The usual assignment will give you what you expect.

    >>> grid[0][0] = 'x'
    >>> grid
    [['x', '', ''], ['x', '', ''], ['x', '', '']]

    Use generator, then elements will be independent.

    >>> grid = [[0]*3 for i in range(3)] # Generalizing, [ [0] * c for i in range(r) ]
    >>> grid
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    >>> grid[0][0]= 4
    >>> grid
    [[4, 0, 0], [0, 0, 0], [0, 0, 0]]

::

    # IDEA : RECORD EACH MOVE
    def tictactoe(moves):
        n = 3  # size of the board
        rows, cols = [0] * n, [0] * n
        diag = anti_diag = 0

        # player A will have value 1, player B value -1, player A starts.
        player = 1

        for row, col in moves:
            # Using data from the given array 'moves' record the move of a player.
            rows[row] += player
            cols[col] += player

            # If this move is placed on diagonal or anti-diagonal,
            # we shall update the relative value as well.
            if row == col:
                diag += player
            if row + col == n - 1:
                anti_diag += player

            # check if this move meets any of the winning conditions.
            if any(abs(line) == n for line in (rows[row], cols[col], diag, anti_diag)):
                return "A" if player == 1 else "B"

            # If no one wins so far, change to the other player.
            player *= -1

        # If all moves are completed and there is still no result, we shall check if
        # the grid is full or not. If so, the game ends with draw, otherwise pending.
        return "Draw" if len(moves) == n * n else "Pending"

    moves = [[0, 0], [2, 0], [1, 1], [2, 1], [2, 2]]
    print(tictactoe(moves))  #A


    # The same without comments 
    def tictactoe(moves):
        n = 3  
        rows, cols = [0] * n, [0] * n
        diag = anti_diag = 0
        player = 1
        for row, col in moves:
            rows[row] += player
            cols[col] += player
            if row == col:
                diag += player
            if row + col == n - 1:
                anti_diag += player

            if any(abs(line) == n for line in (rows[row], cols[col], diag, anti_diag)):
                return "A" if player == 1 else "B"

            player *= -1

        return "Draw" if len(moves) == n * n else "Pending"


    moves = [[0, 0], [2, 0], [1, 1], [2, 1], [2, 2]]
    print(tictactoe(moves))  #A

BRUTE FORCE::

    ### My V3
    def tic_tac(moves):
        grid = [[0] * 3 for i in range(3)]
        player = 1
        for m in moves:
            grid[m[0]][m[1]] = player
            player *= -1
        search1 = find_winner(grid)
        rotated_grid = list(list(reversed(x)) for x in zip(*grid))
        search2 = find_winner(rotated_grid)
        diag1 = diag2 = []
        for i in range(3):
            diag1.append(grid[i][i])
            diag2.append(rotated_grid[i][i])
        search3 = calc_sum(diag1)
        search4 = calc_sum(diag2)
        result = [search1, search2, search3, search4]
        if any(result):
            for r in result:
                if r:
                    return r
        elif len(moves) < 9:
            return "Pending"
        else:
            return "Draw"


    def calc_sum(row):
        if sum(row) == 3:
            winner = "A"
        elif sum(row) == -3:
            winner = "B"
        else:
            winner = None
        return winner


    def find_winner(grid):
        for row in grid:
            winner = calc_sum(row)
            if winner:
                return winner

    ### My V2
    def tic_tac_toe(moves):
        grid = [[0] * 3 for i in range(3)]
        mark = 1
        for move in moves:
            grid[move[0]][move[1]] = mark
            mark *= -1
        # print(grid)
        winner1 = get_winner(grid)
        winner2 = get_winner(list(zip(*grid)))
        winner3 = get_winner(get_diagonals(grid))
        # print(winner1, winner2, winner3)
        winners = [winner1, winner2, winner3]
        for w in winners:
            if w != None:
                return w
        if len(moves) < 9:
            return "Pending"
        else:
            return "Draw"


    def get_winner(grid):
        winner = None
        for row in grid:
            if sum(row) == 3:
                winner = "A"
            elif sum(row) == -3:
                winner = "B"
        return winner


    def get_diagonals(grid):
        d1 = d2 = []
        for i in range(3):
            d1.append(grid[i][i])
            d2.append(grid[i][abs(i - 2)])
        diagonals = [d1, d2]
        return diagonals


    moves = [[0, 0], [2, 0], [1, 1], [2, 1], [2, 2]]
    moves2 = [[0, 0], [2, 0], [1, 1], [2, 1], [1, 2], [2, 2]]
    moves3 = [[0, 0], [1, 1], [2, 0], [1, 0], [1, 2], [2, 1], [0, 1], [0, 2], [2, 2]]
    print(tic_tac_toe(moves))  # A
    print(tic_tac_toe(moves2))  # B
    print(tic_tac_toe(moves3))  # Draw


    ### My V1
    def tic_tac_toe(a):
        grid = [[0] * 3 for i in range(3)]
        for i in range(len(a)):
            if i % 2 == 0:
                mark = 1
            else:
                mark = -1
            grid[a[i][0]][a[i][1]] = mark
        # print(grid)
        # print(list(zip(*grid)))
        ans1 = find_winner(grid)
        ans2 = find_winner(list(zip(*grid)))  # transpose grid and do the same check
        ans3 = find_winner2(grid)  # check winner in diagonals
        ans = [ans1, ans2, ans3]
        if any(ans):
            return ans
        if len(a) < 9:
            return "Pending"
        return "Draw"

    def find_winner(grid):
        for i in grid:
            if sum(i) == 3:
                return "A"
            elif sum(i) == -3:
                return "B"

    def find_winner2(grid):
        diag1 = diag2 = []
        for i in range(3):
            diag1.append(grid[i][i])
            diag2.append(grid[i][abs(i - 2)])
        if sum(diag1) == 3 or sum(diag2) == 3:
            return "A"
        elif sum(diag1) == -3 or sum(diag2) == -3:
            return "B"

    moves = [[0, 0], [2, 0], [1, 1], [2, 1], [2, 2]]
    moves2 = [[0, 0], [2, 0], [1, 1], [2, 1], [1, 2], [2, 2]]
    moves3 = [[0, 0], [1, 1], [2, 0], [1, 0], [1, 2], [2, 1], [0, 1], [0, 2], [2, 2]]
    print(tic_tac_toe(moves))   # [None, None, 'A']
    print(tic_tac_toe(moves2))  # ['B', None, None]
    print(tic_tac_toe(moves3))  # Draw

74. (LC 287) Find the Duplicate Number
------------------------------------------
`287. Find the Duplicate Number <https://leetcode.com/problems/find-the-duplicate-number/>`_
Medium

Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.

Example 1:
Input: nums = [1,3,4,2,2]
Output: 2

**Easier solutions** ::

    # set
    def find_dups(a):
        return sum(a) - sum(set(a))

    # Counter
    from collections import Counter
    def find_dups2(a):
        return [k for k, v in Counter(a).items() if v > 1]

| **Satisfactory solutions**
| **Key points**

| If the number of elements in [1,..x] is greater than x, then the duplicate number must be in [1,..x].
| E.g. x=3, len([1,2,3])=3, len([1,2,3,3])=4

| ``sum(v <= x for v in nums)``
| here we are actually counting the True statements, not summing [1,2,3..]
| But if 1<=x, 2<=x, then +1+1. If we found 3 numbers in nums that are <= x, then sum=3,
| if there was a dup, then sum=4.
| - binary search gives us O(n log n), space O(1)

::

    ### Solution 1 (using builtin for bin search)
    from typing import List
    from bisect import bisect_left

    def findDuplicate(nums: List[int]) -> int:
        def f(x: int) -> bool:
            return sum(v <= x for v in nums) > x  #* 
        return bisect_left(range(len(nums)), True, key=f)  #**

#* mostly returns False, but for one True  

#** for [1,3,4,2,2] , range(0, 5), lookup for [0,1,2,3,4] gives [False, False, True, False..] 
we search for True, bisect returns index 2. (So the f function takes as arguments nums from 
the range?)

| nums = [1, 3, 4, 2, 2]
| print(findDuplicate(nums)) #2
| # key=f is the key function, a callable that returns a value used for sorting or ordering

::

    ### Solution 2 (Implement bin search manually)
    #V1 
    class Solution(object):
        def findDuplicate(self, nums):
            low, high = 1, len(nums) - 1
            while low <= high:
                mid = (low + high) >> 1
                cnt = sum(x <= mid for x in nums)
                if cnt > mid:
                    high = mid - 1
                else:
                    low = mid + 1
            return low

    #V2
    def findDuplicate(nums):
        L, R = 1, len(nums) - 1
        while L <= R:
            mid = (L + R) >> 1
            cnt = sum([1 for num in nums if num <= mid])

            if cnt <= mid:
                L = mid + 1
            else:
                R = mid - 1
        return L

    ### Solution 3
    # IDEA : TWO POINTERS
    class Solution(object):
        def findDuplicate(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            slow , fast = nums[0] , nums[nums[0]]
            while slow != fast:
                slow = nums[slow]
                fast = nums[nums[fast]]

            slow = 0
            while slow != fast:
                slow = nums[slow]
                fast = nums[fast]
            return slow


**Not acceptable solutions**::
    
    # Using dict lookup, but space
    # addresses, but changes given array
    def find_dups3(a):
        for i in a:
            if a[abs(i) - 1] < 0:
                return i
            a[i - 1] *= -1

    # Sort array first, but changes array
    class Solution:
        def findDuplicate(self, nums):
            nums.sort()
            for i in range(1, len(nums)):
                if nums[i] == nums[i-1]:
                    return nums[i]

75. (LC 412) Fizz Buzz
-----------------------------
`412. Fizz Buzz <https://leetcode.com/problems/fizz-buzz/>`_
Easy ::

    class Solution(object):
        def fizzBuzz(self, n):
            """
            :type n: int
            :rtype: List[str]
            """
            ans = []
            for x in range(1, n + 1):
                n = str(x)
                if x % 15 == 0:
                    n = "FizzBuzz"
                elif x % 3 == 0:
                    n = "Fizz"
                elif x % 5 == 0:
                    n = "Buzz"
                ans.append(n)
            return ans

