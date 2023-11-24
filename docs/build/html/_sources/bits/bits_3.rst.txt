Bit Manipulation Questions Part 3
=================================

20. Reverse bits
----------------
Reverse bits of a given 32 bits unsigned integer.
For example, given input 43261596 (represented in binary as 
00000010100101000001111010011100), return 964176192 (represented in binary as 
00111001011110000010100101000000).

**Bit manipulation** ::

    def reverse_bits(n):
        m = 0
        i = 0
        while i < 32:            
            m = (m << 1) + (n & 1)
            n >>= 1
            i += 1
        return m

    # or while n:  then remove i+=1
    def reverse_bits2(n):
        m = 0
        while n:
            m = (m << 1) + (n & 1)
            n >>= 1
        return m

    print(reverse_bits2(678))

| With m we start building a new number from scratch.
| m<<1 adding a 0 on the right (First iteration, m=0, 0<<1=0)
| n&1 we get the rightmost digit of the original number (e.g. 1011 & 1 = 1)
| m<<1 + n&1 Add 0 or 1 to m (n&1 will result in either 1 or 0)
| (Reminder e.g. 10 + 1 = 11, 10 + 0 = 10)

| n>>1 
| we remove the last right digit, so we deal with the next right digit of the original number.

::

    def reverseBits(n):
        res = 0
        for i in range(32):
            res <<= 1
            res |= ((n >> i) & 1)
        return res

**String manipulation.**
The fuss of padding a given number with 0s.
If we want bin(3) in python, it returns "11" and not "000...11".
Then if we reverse it i.e., reverse of "11" is "11" but expected answer 
is "110000... "(reverse of 000...11). ::

    class Solution:
        def reverseBits(n):
            s = bin(n)[2:]
            s = "0"*(32 - len(s)) + s  #to pad with 0s
            t = s[::-1]
            return int(t,2)            #to return in binary format

| Alternatively to bin(n)[2:] => bin(n).replace("0b", "")
| Also could add a test => if (len(s) != 32):

**Another string manipulation.**
Pad with spaces, replace spaces with 0. ::

    class Solution:
        def reverseBits(n):
            n = bin(n)[2:]        
            n = '%32s' % n 
            n = n.replace(' ','0') 
            return int(n[::-1],2)

    # using [:1:-1] instead of [:2] and [::-1]
    def reverseBits(n):
        """
        :type n: int
        :rtype: int
        """
        b = bin(n)[:1:-1]
        return int(b + '0'*(32-len(b)), 2)

21. Single number
-----------------
Given an array of integers, every element appears twice except for one. Find that single one.

(Note, This also works for finding a number occurring odd number of times, 
where all the other numbers appear even number of times.
Note: Your algorithm should have a linear runtime complexity. Could you implement 
it without using extra memory?)

**BIT XOR**
(Without using extra memory.) ::

    def find_single(a):
        ans = 0
        for n in a:
            ans ^= n
        return ans

| if all numbers appear twice, returns 0
| Recall the following two properties of XOR:
| It returns zero if we take XOR of two same numbers.
| It returns the same number if we XOR with zero.
| x ^ 0 = x 
| x ^ x = 0
| So we can XOR all the numbers in the input; 
| duplicate numbers will zero out each other and we will be left with the single number.

*Note.*
It relies on the fact that the duplicate numbers appear exactly twice (or other 
even number), not three times e.g.

**Python stdlib** ::

    import collections 
    def singleNumber(nums):
        num_count = collections.Counter(nums)
        return [x for x in num_count if num_count[x] == 1] 

Reminder of how collections.Counter(L) works:
>>> import collections
>>> L = [2,4,5,2]
>>> collections.Counter(L)
Counter({2: 2, 4: 1, 5: 1})

**using set()** ::

    def singleNumber(nums):
        return 2 * sum(set(nums)) - sum(nums)

**loop, extra array (remove, append)** ::

    def singleNumber(nums):
        no_repeat_array=[]
        for index, item in enumerate(nums):
            if item in no_repeat_array:
                no_repeat_array.remove(item)
            else:
                no_repeat_array.append(item)
        return no_repeat_array.pop()

22. Single number2
------------------
Given an array of integers, every element appears three times except for one, 
which appears exactly once. Find that single one.

| Example:
| Input: [2,2,3,2]
| Output: 3

(Note: Your algorithm should have a linear runtime complexity. 
Could you implement it without using extra memory?)

**Solution** ::

    def singleNumber(self, nums):
        # for [a,a,b,a] array
        # 3*(a+b) - (a+a+b+a) = 2b
        return int((3*(sum(set(nums))) - sum(nums))//2)  #generator doesn't use space

    # The same, but more elaboration
    def f2(a):
        s_set = sum(set(a)) * 3
        s_a = sum(a)
        res = (s_set - s_a) / 2
        return res

    # Python stdlib (but uses memory)
    import collections 
    def singleNumber(nums):
        num_count = collections.Counter(nums)
        return [x for x in num_count if num_count[x] == 1]

23. Single number3
------------------
Given an array of numbers nums, in which exactly two
elements appear only once and all the other elements
appear exactly twice. Find the two elements that appear only once.

| For example:
| Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].
| Note:
| The order of the result is not important. So in the above example, [5, 3] is also correct.
 
| Your algorithm should run in linear runtime complexity.
| Could you implement it using only constant space complexity?
| Time: O(n) Space: O(1)

::

    # Stdlib
    import collections 
    def singleNumber(nums):
        num_count = collections.Counter(nums)
        return [x for x in num_count if num_count[x] == 1]

24. (LC 401) Binary Watch
-------------------------
| *Description*
| A binary watch has 4 LEDs on the top which represent the hours (0-11), and the 6 LEDs on the bottom represent the minutes (0-59).
| # A binary watch looks like this:
| 8 4 2 1
| 32 16 8 4 2 1
| E.g. to show the time 3:25 the lighted numbers would be
| 2 1
| 16 8 1 

| *Task*
| Given a non-negative integer n which represents the number of LEDs that are currently on, return all possible times the watch could represent.
| Example:
| Input: n = 1
| Return: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]

*Note.*
The hour must not contain a leading zero, for example "01:00" is not valid, it should be "1:00".
The minute must be consist of two digits and may contain a leading zero, 
for example "10:2" is not valid, it should be "10:02".

*Hint:* each of the numbers on the watch (1,2,4,8,16,32) has one bit 1 in binary:

>>> bin(8); bin(16)
'0b1000'
'0b10000'

::

    # IDEA : GREEDY 
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        ans = []
        for h in range(12):
            for m in range(60):
                if (bin(h)+ bin(m)).count('1') == num:
                    ans.append('%d:%02d' % (h, m))
        return ans

    # The same but using list comprehension
    def readBinaryWatch(num):
        return ['%d:%02d' % (h, m)
                for h in range(12) for m in range(60)
                if (bin(h) + bin(m)).count('1') == num]

# format

>>> m=4
>>> '%02d' % m
'04'

| # count 1s
| We make use of the fact that the given number = the number of lighted numbers = num of 1s.
| And (bin(h) + bin(m)).count('1') == num
| E.g. time 2:12
| h=2 (1 lamp), m=12 (composed from 8 and 4, i.e. 2 lamsp) => 3 lamps overall

>>> bin(2) + bin(12)
'0b100b1100' # 3 1s

25. Bitwise AND of-numbers-range
--------------------------------
Given a range [m, n] where 0 <= m <= n <= 2147483647,
return the bitwise AND of all numbers in this range, inclusive.

For example, given the range [5, 7], you should return 4. ::

    # Solution 1
    def f8(m, n):
        while n > m:
            n &= n-1
        return n
        
    m, n = 5, 7
    print(f8(m,n)) 

>>> 7&6&5
4

# Note, the following won't work.
It will be an infinite loop.
You have to be changing n, the bigger number. & diminishes n.
Changing m, makes m smaller, it will never be greater than n, i.e. infinite loop.
Going 5,6,7. m=5&6=4, m=4&7=4, m=4&8=0 etc. ::

    def f9(m, n):
        while n > m:
            m &= m+1
        return m

::

    # Solution my v
    def f(a):
        ans = a[0]
        for i in range(a[0] + 1, a[1] + 1):
            ans = ans ^ i
        return ans

    a = [5, 7]
    print(f(a)) #4

26. (LC 461) Hamming distance 
---------------------------------------
*(Easy)*
The Hamming distance between two integers is the number of positions at which 
the corresponding bits are different.
Given two integers x and y, return the Hamming distance between them.

| Example
| Input: x = 1, y = 4
| Output: 2
| Explanation:
| 1   (0 0 0 1)
| 4   (0 1 0 0)

::

    # Solution 1 (Bitwise operator)
    def hammingDistance(x, y):
        return bin(x ^ y).count('1')

It uses the properties of Xor operator. Recall, that xor evaluates to 1 if 
the two compared numbers are 0 and 1 (1^1, 0^0 evaluate to 0)

| 0001 ^
| 0100
| 0101

So the resulting number evaluates to 1 only if the two compared numbers are different.
Then we simply count 1s with .count('1') ::

    ### My solutions
    # 1
    def f31(x, y):
        z = x ^ y
        return bin(z).count('1')

    print(f31(1, 4)) #2
    print(f31(29, 5)) #2

    # 2
    def f32(x, y):
        z = x ^ y
        count = 0
        while z:
            if z & 1:
                count += 1
            z >>= 1
        return count

    print(f32(1, 4)) #2
    print(f32(29, 5)) #2

27. (LC 477) Total Hamming Distance
------------------------------------
*(Medium)*
The Hamming distance between two integers is the number of positions at which 
the corresponding bits are different.
Given an integer array nums, return the sum of Hamming distances between all the 
pairs of the integers in nums.

Example 1:
Input: nums = [4,14,2]
Output: 6
Explanation: In binary representation, the 4 is 0100, 14 is 1110, and 2 is 0010 (just
showing the four bits relevant in this case).
The answer will be:
HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.

Example 2:
Input: nums = [4,14,4]
Output: 4

::

    ### Solution
    def hammingDistance(x, y):
        return bin(x ^ y).count('1')

    def total_hd(nums):
        ans = 0
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                dis = hammingDistance(nums[i], nums[j])
                ans += dis
        return ans

    nums = [4,14,2]
    print(total_hd(nums))   #6

    # My solution 2 
    # (Use library to make pairs and built in method to count 1s.)
    import itertools
    def total_ham_dist3(a):
        td = 0  # total distance
        pairs = list(itertools.combinations(a, 2))
        for pair in pairs:
            n1 = pair[0]
            n2 = pair[1]
            td += (n1 ^ n2).bit_count()
        return td

    nums = [4, 14, 2]
    print(total_ham_dist3(nums))

    # My solution 1
    # (Uses XOR and & to count hamming distance. But as for putting whatever way of 
    # counting into a separate function - yes it is better.)
    def f34(a):
        ham_sum = 0
        for i in range(len(a)-1):
            for j in range(i+1, len(a)):
                ham = 0
                ij = a[i]^a[j]
                while ij:
                    ij = ij & (ij - 1)
                    ham += 1
                ham_sum += ham
        return ham_sum

    a = [4,14,2]
    a2 = [4,14,4]
    print(f34(a))   #6
    print(f34(a2))  #4

28. Maximum-product-of-word-lengths
-------------------------------------
Given a string array words, find the maximum value of
length(word[i]) * length(word[j]) where the two words
do not share common letters. You may assume that each
word will contain only lower case letters. If no such
two words exist, return 0.

| Example 1:
| Given ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]
| Return 16
| The two words can be "abcw", "xtfn".

::

    ### Solution
    # Using set() and itertools std lib

    import itertools
    def common(a, b):
        return set(a) & set(b)

    def product_score(w1, w2):
        return 0 if common(w1, w2) else len(w1) * len(w2)

    def maxProduct(words):
        return max(product_score(w1, w2) 
                    for (w1, w2) in itertools.combinations(words, 2))

| # Explained
| 1) & set operator - identifies intersect, items in both a and b.
| 2) ``itertools.combinations(iterable, r)``
| Return r length subsequences of elements from the input iterable.

>>> list(itertools.combinations([1,2,3,4], 2))
[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
>>> list(itertools.combinations([1,2,3, 4], 3))
[(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]

::

    ### Remake
    def f(w1, w2):
        """no common letters"""
        if len(set(w1) & set(w2)) == 0:
            return True
        return False

    def max_prod(a):
        return max(len(i) * len(j) for (i, j) in itertools.combinations(a, 2) if f(i, j))

    ### No magic at all
    def unique(w1, w2):
        if len(set(w1) & set(w2)) == 0:
            return True
        return False

    def max_len(a):
        combos = itertools.combinations(a, 2)
        lens = []
        for x, y in combos:
            if unique(x, y):
                l = len(x) * len(y)
                lens.append(l)
        return max(lens)

    a = ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]
    print(max_len(a))  #16

29. (LC 421) Maximum XOR of Two Numbers in an Array
----------------------------------------------------
| *(Medium), O(32n).*
| Given an integer array nums, return the maximum result of nums[i] XOR nums[j].
| Example 1:
| Input: nums = [3,10,5,25,2,8]
| Output: 28
| Explanation: The maximum result is 5 XOR 25 = 28.

::

    # Solution 
    def findMaximumXOR(nums):
        ans = mask = 0
        for i in range(32)[::-1]:
            mask += 1 << i               #alternatively mask |= 1<<i
            prefixSet = {n & mask for n in nums} 
            temp = ans | 1 << i
            for prefix in prefixSet:
                if temp ^ prefix in prefixSet:
                    ans = temp
                    break
        return ans

    nums = [3,10,5,25,2,8]
    print(findMaximumXOR(nums))  #28 (25^5)

*Explained.*

-We are basing our solution on the fact that the max xor is 1111..
Now, how many 1s to take, we could take the len of max number, or just some
max possible, i.e. 32. 

| -But we rather start looking at the most significant bit on the left.
| 100000..
| 110000..
| -Also consider, max ^ num1 = num2
| Knowing max and num1, we make a lookup in the list for num2.

| 1)
|     ``for i in range(32)[::-1]:``
| I.e. we start with biggest i=32
 
| 2)
|         ``mask += 1 << i``
| We look at the most significant bit
| On iterations our mask is 10000.., 110000, 1110000
| 100000..
 
| 100000..+ (we could also use | OR operator to the same effect)
| x10000..
| 110000..

| 3)
|         ``prefixSet = {n & mask for n in nums}``
| n & mask extracts MSBs of our numbers.
| E.g. for nums 5, 25 (101, 11001), set at one point will be (100, 11000)
| 11100&101=100 (mask & n)
| Before 1s in mask reach the number with longest length, set will be -> all 0s.
| When there are enough ones that they touch the first MSB of our numbers, a set with MSB of our numbers will start to appear.
 
| 4)
|         ``temp = ans | 1 << i``
| With temp we build the biggest XOR starting from the MSB.
| len of temp will be equal to index, when we first start to touch MSBs of our nums.
| Our example 5, 25 (101, 11001). So first temp is going to be 10000.

(Why, because before that tepm = 100000.., is a large number, we are not going to find
temp ^ prefix in prefixes, so we loop fruitlessly until we start finding for
shorter numbers, smaller i-s, like i= 4, for num 25 (10000) first MSB revealed).

| From i=4, mask reveals (5,25), (101, 11001) prefixes = {0, 10000}
| temp = ans | 1<<i, which is 0| 10000 = 10000
| Our condition was if temp ^ prefix in prefixes
| So we have 10000 ^ 10000 = 0
| 0 is in prefixes
| (before i=4, we had 100000 ^ 000 (no MSB revealed yet)= 100000 which is not in prefixes)
| ans = 10000
| we break out of the inner loop
 
| Continues main loop, i=3
| prefixes = {0, 11000}
| temp= 10000 | 1000 = 11000
| 11000 ^ 11000 = 0 which is in prefixes
| ans = 11000
| break from inner
 
| i=2
| prefixes = {100, 11000}  #mask&n = 111100 & 101 = 100 (reveals MSB of 5)
| temp = ans|i<<2 = 11000 | 100 = 11100
| temp^prefix = 11100^100=11000 #now temp pairs with 5, result is MSBs of 25
| ans = 11100
| After that we don't find anything else, so the ans=11100 sticks
| And indeed 5^25 = 11100


