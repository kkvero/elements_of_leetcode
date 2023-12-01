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

31. (LC 1573) Number of Ways to Split a String
----------------------------------------------
Given a binary string s, you can split s into 3 non-empty strings s1, s2, and s3 
where s1 + s2 + s3 = s.
Return the number of ways s can be split such that the number of ones is the same 
in s1, s2, and s3. Since the answer may be too large, return it modulo 10^9 + 7.

| Example 1:
| Input: s = "10101"
| Output: 4
| Explanation: There are four ways to split s in 3 parts where each part contain the same number of letters '1'.
| "1|010|1"
| "1|01|01"
| "10|10|1"
| "10|1|01"
 
| Example 2:
| Input: s = "1001"
| Output: 0
 
| Example 3:
| Input: s = "0000"
| Output: 3
| Explanation: There are three ways to split s in 3 parts.
| "0|0|00"
| "0|00|0"
| "00|0|0"
 
| Constraints:
|     3 <= s.length <= 105
|     s[i] is either '0' or '1'.

**Solution** ::
    
    def numWays(s):
        mod = 10**9+7
        cnt = s.count('1')
        if cnt == 0: return (len(s)-1)*(len(s)-2)//2 % mod
        if cnt % 3 != 0: return 0
        ones = []
        for i,x in enumerate(s):
            if x == '1': ones.append(i)
        return (ones[cnt//3] - ones[cnt//3-1]) * (ones[2*cnt//3]- ones[2*cnt//3-1]) % mod

        # possible rewrite for the last part (n_ones - is number of ones in each cut, 
        # indices - indices where we have 1s)
        n_ones = int(ones / 3) 
        return (indices[n_ones] - indices[n_ones - 1]) + (
            indices[2 * n_ones] - indices[2 * n_ones - 1]

**Explanations of the basic math.**

1. How to count the number of possible ways to cut a string.

| s = "100100010100110"
| num_of_ones = 6
| For a valid way to split,each part must have 2 ones.
| Number of ones encountered so far at each index:
| ones = [1,1,1,2,2,2,2,3,3,4,4,4,5,6,6]
| We can make the first cut at indices from 3-6 (the first part should end there).
| So,we have 4 ways to choose for the end index of the 1st cut.
| Similarly,for the 2nd part,it may end at any index from 9-11 and still be a valid cut i.e. 3 ways.
| Multiplying these,we get 4*3 = 12 ways.

| *But we use a different algorithm in the solution.*
| s = "100100010100110"
| cnt = s.count('1')  #cnt=6
| # Record at what indices we encounter '1'.
| ones = [0,3,7,9,12,13]  #x=1 at these indexes (x is value at index i)
 
| Breaking down this:
| ``return (ones[cnt//3] - ones[cnt//3-1]) * (ones[2*cnt//3]- ones[2*cnt//3-1]) % mod``
| 1) (ones[cnt//3] - ones[cnt//3-1])
| cnt//3  #cnt=6, so 6//3 = 2
| ones[2] - ones[1] #values are ones at these indices 
| 7 - 3 = 4  #4 places where we could cut the first part 
| (literally, we could cut anywhere between indices 3 and 7: at 3,4,5,6=4 places)
| # In other words:
| (ones[cnt//3] - ones[cnt//3-1])
| is (end of first cut - start of first cut) 
| 2) (ones[2*cnt//3]- ones[2*cnt//3-1])
| Times 2, because for the second cut at the indices we should have double the number
| of 1s.
| ones[2*2] - ones[2*2-1]
| 12 - 9 = 3
| Overall ways = 4 * 3 (corresponds to the explanation of the basic math)

2. Ways to cut a string with only 0s

| s = '000000'
| total no. of positions available for cut1: (n-1)
| total no. of positions available for cut2: (n-2), since one position is taken by cut1.
 
| hence total no. of ways to place two cuts: (n-1)*(n-2)
 
| But it includes duplicate cases, as cut1 at position 1 and cut2 at position 2 
| will be the same as cut1 at position 2 and cut2 at position 1. 
| Both will give the same substrings of s. ('0|0|0000')
| So, total num of ways to cut a string in 3 parts is: (n-1)*(n-2)/2

32. (LC 342) Power of Four
--------------------------
(easy)
Given an integer n, return true if it is a power of four. Otherwise, return false.
An integer n is a power of four, if there exists an integer x such that n == 4**x.

| Example 1:
| Input: n = 16
| Output: true
| Example 2:
| Input: n = 5
| Output: false
| Example 3:
| Input: n = 1
| Output: true
| # Time:  O(1)
| # Space: O(1)

::

    ### Solution 1 (recursion)
    def isPowerOfFour(num):
        if num <= 0: return False
        if num == 1: return True
        if num % 4 == 0:
            return isPowerOfFour(num / 4)
        return False


    ### Solution 2 (BIT MANIPULATION)
    def isPowerOfFour(self, num):
        return num > 0 and (num & (num - 1)) == 0 and (num & 0x55555555) != 0

1. ``(num & 0x55555555) != 0``

>>> bin(0x55555555)
'0b1010101010101010101010101010101'

If we look at examples of numbers that are powers of 4:

>>> 4**2
16
>>> 4**3
64
>>> bin(4)
'0b100'
>>> bin(16)
'0b10000'
>>> bin(64)
'0b1000000'

We see that such numbers always have an even number of zeros, and 1 is at an odd 
position.
Hence when & compared with a mask ..1010101, the 1 in 10000 should align with a 1
in the mask.

| E.g.
| 10000 &
| 10101
| 10000

2. ``(num & (num - 1))``

Again, because a num power of 4 is of format a 1 and all zeros like 10000,
10000 & 10000-1 = 0, 10000 & 1111 = 0

33. (LC 762) Prime Number of Set Bits in Binary Representation
--------------------------------------------------------------
(Easy)
Given two integers left and right, return the count of numbers in the inclusive 
range [left, right] having a prime number of set bits in their binary representation.

Recall that the number of set bits an integer has is the number of 1's present when written in binary.
For example, 21 written in binary is 10101, which has 3 set bits.

| #Example 1:
| Input: left = 6, right = 10
| Output: 4
| Explanation:
| 6  -> 110 (2 set bits, 2 is prime)
| 7  -> 111 (3 set bits, 3 is prime)
| 8  -> 1000 (1 set bit, 1 is not prime)
| 9  -> 1001 (2 set bits, 2 is prime)
| 10 -> 1010 (2 set bits, 2 is prime)
| 4 numbers have a prime number of set bits.
 
| #Example 2:
| Input: left = 10, right = 15
| Output: 5
| Explanation:
| 10 -> 1010 (2 set bits, 2 is prime)
| 11 -> 1011 (3 set bits, 3 is prime)
| 12 -> 1100 (2 set bits, 2 is prime)
| 13 -> 1101 (3 set bits, 3 is prime)
| 14 -> 1110 (3 set bits, 3 is prime)
| 15 -> 1111 (4 set bits, 4 is not prime)
| 5 numbers have a prime number of set bits.

Also recall that 1 is not a prime, 2 is the smallest prime. ::

    ### Solution 1 (sort of my version)
    # (In this version we count the number of set bits in the main function.
    # In the helper function we identify if that number is prime.)
    import math
    def is_prime(n):
        if n==1:
            return False
        elif n==2:
            return True
        for i in range(2, int(math.sqrt(n))+1):
            if n % i == 0:
                return False
        return True
        
    def count_set_bits(n1, n2):
        primes = [x for x in range(n1, n2+1) if is_prime(bin(x)[2:].count('1'))]
        return len(primes)

    print(count_set_bits(6, 10))  #4
    print(count_set_bits(10, 15)) #5

    ### Solution 2
    # (In this version the helper function both counts the set bits using bit operators.
    # The main function only makes a list of valid results, calling the helper function.)
    import math
    def f17(x):
        '''number of set bits is a prime number'''
        count = 0
        while x:
            x &= (x-1)
            count +=1
        if count == 1:
            return False
        for i in range(2, int(math.sqrt(count))+1):
            if count % i == 0:
                return False
        return True

    # Verify f17
    print(f17(300))  #'0b100101100'
    print(f17(21))  #'0b10101'
    print(f17(6))  #'0b110'
    # OUT
    # False
    # True
    # True

    def f18(x,y):
        '''For numbers in range x, y: how many nums have prime number of set bits'''
        res = [i for i in range(x, y+1) if f17(i)]
        print(res)
        return len(res)

    print(f18(6, 10))
    # OUT
    # [6, 7, 9, 10]
    # 4

34. (LC 645) Set Mismatch
-------------------------
(Easy)
You have a set of integers <s>, which originally contains all the numbers from 1 to n. 
Unfortunately, due to some error, one of the numbers in s got duplicated to 
another number in the set, which results in 
<repetition of one number> and <loss of another> number.

You are given an integer array <nums> representing the data status of this set after the error.
Find the number that occurs twice and the number that is missing and return them in the form of an array.

| Example 1:
| Input: nums = [1,2,2,4]
| Output: [2,3]
 
| Example 2:
| Input: nums = [1,1]
| Output: [1,2]
 
| Constraints:
|     2 <= nums.length <= 104
|     1 <= nums[i] <= 104

*Solutions seem to assume that the given range definitely starts with 1.
And the missing number comes right after the duplicate.* ::

    # using set, sum
    ### Solution 1
    def findErrorNums(nums):
        return [sum(nums) - sum(set(nums)), sum(range(1, len(nums)+1)) - sum(set(nums))]

    # OR
    def find_dup_mis(a):
        dup = sum(a) - sum(set(a))
        mis = sum(list(range(1, len(a) + 1))) - sum(set(a))
        return [dup, mis]

    # Breaking it down
    def find_error(a):
        sum1 = sum(a)
        sum2 = sum(set(a))
        dup = sum1 - sum2
        sum3 = sum(range(len(a)+1))
        missing = sum3 - sum2
        return [dup, missing]

    a = [1,2,3,4,4,6,7]
    print(find_error(a))  #[4, 5]

    # using bit manipulation
    def find_dup_mis2(a):
        # finding mis (make list = our a + full list = [1,1,2,2,>3<,4,4])
        mis = 0
        L = list(set(a)) + list(range(1, len(a) + 1))
        mis = 0
        for n in L:
            mis ^= n
        # finding dup (use num^num=0)
        for i in range(len(a)):
            if a[i] ^ a[i + 1] == 0:
                dup = a[i]
                break
        return [dup, mis]

    a = [1, 2, 2, 4]
    print(find_dup_mis2(a)) #[2, 3]

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





