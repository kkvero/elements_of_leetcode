Bit Manipulation Questions Part 4
=================================

30. (LC 462) Minimum Moves to Equal Array Elements II
-----------------------------------------------------
Given a non-empty integer array, find the minimum number of moves required to 
make all array elements equal, where a move is incrementing a selected element 
by 1 or decrementing a selected element by 1.

| Example.
| Input: [1,2,3]
| Output: 2
| Explanation:
| Only two moves are needed (remember each move increments or decrements one element):
| [1,2,3]  =>  [2,2,3]  =>  [2,2,2]

::

    def minMoves2(nums):
        nums.sort()
        median = nums[len(nums) / 2]
        return sum(abs(num - median) for num in nums)

    # Less magic
    def f(a):
        moves = 0
        value = int(sum(a) / len(a))
        for n in a:
            moves += abs(value - n)
        return moves

35. (LC 371) Sum of Two Integers
---------------------------------
(Medium)
Given two integers a and b, return the sum of the two integers without using the operators + and -.
Constraints:
-1000 <= a, b <= 1000 ::

    # Solution 1
    class Solution(object):
        def getSum(self, a, b):
            return sum([a,b])

    # Solution 2 rewrite
    import math
    def summing(m, n):
        return math.log(math.e**m * math.e**n)
    print(summing(5, 4)) #9.0

    # Solution 2
    class Souluton:
        def getSum(self, a, b):
            tmp = math.exp(a) * math.exp(b)
            r = int(math.log(tmp))
            return r

(see :ref:`math-kb1-label`)
::

    # Solution 3
    # Time: O(32)O(32)
    # Space: O(1)O(1)

    class Solution:
        def getSum(self, a: int, b: int) -> int:
            # 32 bit mask in hexadecimal
            mask = 0xffffffff
            # works both as while loop and single value check 
            while (b & mask) > 0:
                carry = ( a & b ) << 1
                a = (a ^ b) 
                b = carry
            # handles overflow
            return (a & mask) if b > 0 else a

Note 1, this algorithm does not handle the case when it is given a negative b initially.
getSum(-7, 3) Ok, getSum(-7,-3) Nope, returns -7.
Note 2. We are given the constraint a,b between 1000, -1000, so why accounting for overflow?
Negative integers.
For using mask here see :ref:`mask-label`.

| # Actual algorithm points here:
| - XOR adds correctly, if there was no carry
| - AND (a & b) correctly returns the carry, but we just have to move it one position to the left, hence << 1
| So we continue until there is no carry (carry = 0, and b = carry = 0)





