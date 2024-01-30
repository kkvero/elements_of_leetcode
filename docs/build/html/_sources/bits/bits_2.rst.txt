Bit Manipulation Questions Part 2
=================================

10. Binary gap
--------------
Given a positive integer N, find and return the longest distance between two 
consecutive 1' in the binary representation of N. If there are no two 
consecutive 1's, return 0.

Note: I would rather say - longest len of consecutive 1s.

For example: Input: 22 Output: 2 Explanation: 22 in binary is 10110 In the binary 
representation of 22, there are three ones, and two consecutive pairs of 1's. 
The first consecutive pair of 1's have distance 2. The second consecutive pair 
of 1's have distance 1. The answer is the largest of these two distances, which is 2.

**Solution** (v1)::

    def f12(n):
        max_dist = 0
        cur_dist = 0
        while n:
            if n & 1:
                cur_dist += 1
            else:
                max_dist = max(max_dist, cur_dist)
                cur_dist = 0
            n >>= 1
        return max_dist

    print(f12(20))
    print(f12(444))

>>> bin(20)
'0b10100'
>>> bin(444)
'0b110111100'

**Solution** (v2)::

    def f1(N):
        ans=0
        current=0
        while N:
            if N & 1:
                current +=1
                ans = max(ans, current)
            else:
                current = 0
            N >>= 1
        return ans

    print(f1(472))  #returns 3

>>> bin(472)
'0b111011000'

| **Solution** (initial)
| # here:
| last - previous index with_value_1
| ans - longest_size of 1s
| index - current_index

::

    def binary_gap(N):
        last = None
        ans = 0
        index = 0
        while N != 0:
            if N & 1:
                if last is not None:
                    ans = max(ans, index - last)
                last = index
            index += 1
            N >>= 1
        return ans

| ``if N&1`` - i.e. if we encountered 1 (the last value of N is 1).
| ``if last is not None``, i.e. if we have a chain, the previous index had value 1
| ``ans = max(ans, index-last)``, i.e. greedy algorithm, if current size is bigger, we set the ans to that, or leave the old value. 

11. Get, set, clear one bit
---------------------------
Basic functions that get, set, clear one bit at an index.
(Recall that indexing in binary numbers start from LSB, i.e. on the right.)
::

    def get_bit_2(n, i):
        value_at_i = n & (1 << i)
        return value_at_i >> i

    print(get_bit_2(45, 4))  #'0b101101'
    print(get_bit_2(10, 1))  #'0b1010'
    print(get_bit_2(67, 0))  #'0b1000011'

| *Explained*
| n & (1 << i)
| if n=1011 we want to get bit at i=2, then 
| 1<<i is 100
| n&(1<<i) is
| 1011 &
| 0100
| 0000
| To get just one bit at index we shift the result >> i
| If result were 0100 we couldn't return that, we need to shift up to index 1|00

(Also we cannot shift just the original n, then there might be other 1s on 
the left side.)

My version::

    def get(n, i):
        return (n >> i) & 1

    print(get(20, 3)) #1
    print(get(20, 4)) #0

>>> bin(20)
'0b10100'

Clear::

    def clear_bit(num, i):
        mask = ~(1 << i)
        return num & mask

    print(clear_bit(20, 2))  # 16

>>> bin(20)
'0b10100'
>>> bin(16)
'0b10000'

| ~(1 << i), flips 0s to 1s. E.g. i=3, ~1000=111 

What is not obvious at first sight is that in fact it gives us 1110111.
With a 0 just where we want it, at the given index. So it preps with 1s on the left too.
(Otherwise num & mask would give us a shorter number, as is the case with e.g. 10100&11).
(Not operator works at all here because it is in an expression.
Recall when used on its own, ~ gives a weird result in Python. But works correctly 
when its result is passed further on in the function.)
::

    def update_bit(num, i, bit):
        mask = ~(1 << i)
        return (num & mask) | (bit << i)

We proceed by first clearing the bit in question, using exactly the algorithm in
clear_bit. Then apply OR bit << i. This would set the bit, which we turned to 0
to whatever the bit is.

*My version (new)*::

    def set_bit(n, i):
        mask = 1 << i
        n = n | mask
        return n

    print(set_bit(20, 1)) #22

>>> bin(20)
'0b10100'
>>> bin(22)
'0b10110'

*My version (old).*
(Extract right side, extract left side and change its last bit, merge left and right)

::

    def set_bit(n, i, v):
        mask = (1 << i) - 1
        right = n & mask
        left = (n >> i) | v
        ans = (left << (i)) | right
        return bin(ans)

    print(set(20, 1, 1))  # 0b10110

    # The same shorter
    def clear2(n, i):
        right = n & ((1 << i) - 1)
        left = n >> i + 1
        return (left << i+1) | right

    print(clear2(12, 2))

>>> bin(12)
'0b1100'
>>> bin(8)
'0b1000'

12. Count flips to convert
--------------------------
Write a function to determine the number of bits you would need to flip to convert 
integer A to integer B. For example: Input: 29 (or: 11101), 15 (or: 01111) Output: 2

**Solution** (alternative)
(More built-ins.)::

    def flips_to_convert(n1, n2):
        flips = n1 ^ n2
        return flips.bit_count()
        # OR
        # return bin(flips).count("1")

**Solution**::

    def count_flips_to_convert(a, b):
        diff = a ^ b
        # count number of ones in diff
        count = 0
        while diff:
            diff &= (diff - 1)
            count += 1
        return count

| #1
| a^b (a=11101, b=01111)
| 11101^  #xor turns 1and1 to 0, 0and1 to 1
| 01111
| 10010
| #2
| num - 1
| E.g. 0b10100 - 1 = 0b10011  (Or more obvious example, 0b1000 - 1 = 111)
| And when doing & on the new num and the previous:
| 10100 &
| 10011
| 10000 # it takes away the 1s on the right side
| (Another -1 and then & with 10000, will take away the remaining 1 in 10000).

So the loop will work for just as many times as there are 1s.

13. Find difference
-------------------
*Task.* Given two strings s and t which consist of only lowercase letters. 
String t is generated by random shuffling string s and then add one more letter 
at a random position. Find the letter that was added in t.
For example: Input: s = "abcd" t = "abecd" Output: 'e'

*Hint.* We use the characteristic equation of XOR. 
A xor B xor C = A xor C xor B If A == C, then A xor C = 0 and then, B xor 0 = B. [:ref:`4 <ref-label>`]
Meaning, if we add the characters of the two strings in question, xor of the same
characters will be 0. And 0 xor with the only unique character results that character
(char ^ char = 0, 0 ^ char = char).

**Solution 1** Bitwise operator

*Logic.*
This uses the same principle as for the problem: you have an array where each element
appears twice, one element appears twice.
Concatenate s + t and you will get a string where each letter appears twice, except one.
We use the fact that XOR of the same characters is 0 -> x^x=0
So XORing all paired characters will give 0, and 0 ^ unique_char = unique_char.
::

    def find_difference(s, t):
        ret = 0
        for ch in s + t:
            ret = ret ^ ord(ch)
        return chr(ret)

    s1='dfgt'
    s2='rfdtg'
    print(find_difference(s1,s2))

**Solution 2** collections + dict ::

    import collections
    def findTheDifference(s, t):
        cnt_s = collections.Counter(s)
        cnt_t = collections.Counter(t)
        res = [x for x in cnt_t.keys() if x not in cnt_s.keys() or cnt_t[x] != cnt_s[x]]
        return res[0]

**V2**::

    import collections
    def f11(s, t):
        return list((collections.Counter(t) - collections.Counter(s)))[0]

    s = 'sjnd'
    t = 'sjfdn'
    print(f11(s,t))  # f

*Explained* (my conclusions).
We cannot subtract dictionaries. That's a fact.
But it looks like we can when dealing with objects of Counter.
Counter returns the dictionary format. But not exactly, if we print intermediary 
results:

| print(collections.Counter(s))
| Counter({'s': 1, 'j': 1, 'n': 1, 'd': 1})   #a dict inside some Counter()
| print((collections.Counter(t) - collections.Counter(s)))
| Counter({'f': 1})   #Then we can subtract

**Solution 3**::

    def findTheDifference2(s, t):
        t = list(t)
        s = list(s)
        for i in s:
            t.remove(i)
        return t[0]

My version::

    def find_dif(s, t):
        return chr(ord_sum(t) - ord_sum(s))

    def ord_sum(w):
        return sum([ord(x) for x in w])

    s='fghs'
    t='ghsff'
    print(find_dif(s,t))  # 'f'

14. Find missing number
-----------------------
Returns the missing number from a sequence of unique integers in range [0..n] 
in O(n) time and space. The difference between consecutive integers cannot be 
more than 1. 
If the sequence is already complete, the next integer in the sequence will be returned.
*My note, the sequence has to start with 0.*

**Solution 1** ::

    def f3(nums):
        return sum(range(len(nums)+1)) - sum(nums)

    nums2 = [0,1,2,3,4,6,7]
    print(f3(nums2)) #5

# My Versions::

    # Using sum
    def f(a):
        sum1 = sum(a)
        sum_full = sum(range(a[0], a[-1] + 1))
        res = sum_full - sum1
        if res > 0:
            return res
        return a[-1] + 1

    # if next num is not current+1
    def f(a):
        for i in range(len(a) - 1):
            if a[i + 1] != a[i] + 1:
                return a[i] + 1
        return a[-1] + 1

    # use the fact that index=value
    def f(a):
        for i, n in enumerate(a):
            if i != n:
                return i
        return a[-1] + 1

**Solution 2** (fancy (but it is O(n)) and it is bit manipulation) ::

    def find_missing_number(nums):
        missing = 0
        for i, num in enumerate(nums):
            missing ^= num
            missing ^= i + 1
        return missing

    nums = [0,1,3,4]
    # print(find_missing_number(nums)) # 2

*Explained*

``missing ^= num`` At this step we always end up with missing=0

``missing ^= i + 1``
0 + i+1, makes sure that we start with missing=the next expected number.
Because current index + 1 = next expected value.
Because with the given constraints our array is of format [0,1,2,3],
where index=value.

``0 ^ next_number = next_number``  0 coming from the previous step.

| When we encounter the wrong_number, we get, as per the two steps of our algorithm:
| 1) missing ^= num ->   expected ^ wrong = difference
| 2) missing ^= i + 1 -> difference ^ wrong = expected

15. Flip bit longest sequence
-----------------------------
You have an integer and you can flip exactly one bit from a 0 to 1. 
Write code to find the length of the longest sequence of 1s such a flip would create. 
For example: Input: 1775 ( or: 11011101111) Output: 8

**Solution**

| Note:
| 1)We test if it is a gap, i.e. after 0 there will be 1, by testing & with 2, which is 10 in binary.

2)In this algorithm we don't actually flip the 0 of the gap.
In that case we store the previous value of current (prev_len = cur_len).
Then we compare max with prev+cur

3)We don't add the gap itself in the process of counting 1s, we do it only in
return max_len + 1
::

    def flip_bit_longest_seq(num):
        curr_len = 0
        prev_len = 0
        max_len = 0
        while num:
            if num & 1 == 1:  # last digit is 1
                curr_len += 1

            elif num & 1 == 0:  # last digit is 0
                if num & 2 == 0:  # second to last digit is 0, i.e. the next to the left,(bin(2)=10)
                    prev_len = 0
                else:
                    prev_len = curr_len
                curr_len = 0

            max_len = max(max_len, prev_len + curr_len)
            num = num >> 1  # right shift num

        return max_len + 1

**V2** (the same, shorter) ::

    def f2(n):
        prev, cur, maxl = 0,0,0
        while n:
            if n & 1:
                cur += 1
            else:
                if not (n >> 1) & 1:
                    prev = 0
                else:
                    prev = cur
                cur = 0
            maxl = max(maxl, cur + prev)
            n >>= 1
        return maxl + 1

16. Alternating bits
--------------------
Given a positive integer, check whether it has alternating bits: namely, if two 
adjacent bits will always have different values.
(For example: Input: 5 Output: True because the binary representation of 5 is: 101.
Input: 7 Output: False because the binary representation of 7 is: 111.)

**Solutions of my choice**

**My solution** ::

    def f4(n):
        while n:
            bit = n & 1
            if (n >> 1) & 1 == bit:
                return False
            n >>= 1
        return True

**Solution 2** ::

    def hasAlternatingBits(self, n):
        bin_n = bin(n)[2:]
        return all(bin_n[i] != bin_n[i+1] for i in range(len(bin_n) - 1))

| Idea:
| To check if 0 always next/before 1 (1 always next/before 0)
| Return Value from all()
| The all() function returns:
| True - If all elements in an iterable are true
| False - If any element in an iterable is false

| Time Complexity O(n)
| github

::

    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        b = 0b1010101010101010101010101010101010101010101010101010101010101010
        # or b = '0b' + '10'*16
        while b > 0:
            if b == n:
                return True
            b = b >> 1
        return False

**Less so solutions** ::

    def has_alternating_bits(n):
        fb = 0
        sb = 0
        while n:
            fb = n & 1
            if n >> 1:              #if there is a second bit
                sb = (n >> 1) & 1
                if fb ^ sb == 0:    #1^1=0, 1^0=1, 0^0=0. So 0 if sb and fb are the same
                    return False
            else:
                return True
            n = n >> 1
        return True

    print(has_alternating_bits(10))
    print(has_alternating_bits(111))
    # True
    # False

>>> bin(10)
'0b1010'
>>> bin(111)
'0b1101111'

**Solution** Time Complexity - O(1) ::

    def has_alternative_bit_fast(n):
        mask1 = int('aaaaaaaa', 16)  # for bits ending with zero (...1010)
        mask2 = int('55555555', 16)  # for bits ending with one  (...0101)
        return mask1 == (n + (n ^ mask1)) or mask2 == (n + (n ^ mask2))

Note, could use b='0b'+('10'\*32), or '01'\*32...but then it is only a string..
could convert it to int with int('0b0101', 2)

| E.g. if n=101
| --101^
| 10101
| 10000  0 or 0 and leading nums of mask if n is a smaller number.
| Basicaly, n should be equal to mask, then n^mask = 0, and mask=n.
| We don't just say n=mask, to allow for testing for numbers with less digits then mask.

17. Insert bit
---------------
``insert_one_bit(num, bit, i)``: insert exact one bit at specific position 

| For example:
| Input: num = 10101 (21) insert_one_bit(num, 1, 2): 101101 (45) insert_one_bit(num, 0 ,2): 101001 (41) 

``insert_mult_bits(num, bits, len, i)``: insert multiple bits with len at specific position
My note, multiple bits are given as a single number. E.g. we provide 7, when we
want to insert 111.

| For example:
| Input: num = 101 (5) insert_mult_bits(num, 7, 3, 1): 101111 (47) 

**My version** ::

    def insert_bit(num, bit, i):
        mask = (1 << i) - 1
        rs = num & mask
        ls = ((num >> i) << 1) | bit  #** 
        num = (ls << i) | rs
        return num

#**cut off right side, add room for new bit, merge left side with new bit
print(insert_bit(21, 1, 2))  # 45, 0b101100 ::

    def insert_bit(num, bits, _len, i):
        mask = (1 << i) - 1
        rs = num & mask
        ls = ((num >> i) << _len) | bits  # <-the only line that differs
        num = (ls << i) | rs
        return num

    print(insert_bit(21, 4, 3, 2))  # 177, '0b10110001'

**Solution** ::

    def insert_one_bit(num, bit, i):
        mask = num >> i
        mask = (mask << 1) | bit
        mask = mask << i
        right = ((1 << i) - 1) & num
        return right | mask

    def insert_mult_bits(num, bits, len, i):
        mask = num >> i
        # the only line that changes, shift by not 1 but len of bits, | bits not bit 
        mask = (mask << len) | bits   
        mask = mask << i
        right = ((1 << i) - 1) & num
        return right | mask

    # Annotated
    def insert_one_bit(num, bit, i):
        mask = num >> i
        # Make space for extra space and set it to bit
        mask = (mask << 1) | bit
        # Make space for bits on the right of i (adding 0s)
        mask = mask << i
        # Bring back the bits that were on the right side of i
        right = ((1 << i) - 1) & num
        return right | mask

| Algorithm: num=10101, bit=1, i=3
| mask = num >> i
| mask=10101>>3 = 10
| -Create a mask that is the part of our number to the left of index i.
| (bits from i to the most significant bit)
 
| mask = (mask << 1) | bit
| mask=10<<1 = 100, 100|1 = 101 
| -add an extra 0 bit on the right, i.e. we create new space for the new bit
| Turn that 0 bit into the value of bit to insert => (mask << 1) | bit
 
| mask = mask << i
| mask=101<<3 = 101000
| -Make the number again the original size (with inserted bit included). 
| I.e. 0s where the right side should be.
 
| right = ((1 << i) - 1) & num
| (1<<3) - 1 = 1000 - 1 = 111
| This will be the mask to get only the part on the right side of i of the original number.
| 111 & num = 111 & 10101 = 101
| AND with all 1s just returns whatever is in the number it is compared with, BUT
| also the length of 1s.
 
| return right | mask
| - OR operator merges two numbers
| 000101 | 
| 101000 = 
| 101101

18. (LC 231) Power of two
-------------------------
| Given an integer, write a function to determine if it is a power of two.
| (A power of two is a number of the form 2**n=x. 
| Find if x is 2**n.)

| *Logic*
| 1 0b1   #2**0
| 2 0b10
| 4 0b100
| 8 0b1000
| 16 0b10000  #2**4

| If all 2**n numbers are of the form 100..0, then x-1=11..1
| Then x & (x-1) = 100..0 & 11..1 = 0

::

    # Solution 1
    def is_power_of_two(n):
        return n > 0 and not n & (n-1)

    # Solution 2
    class Solution(object):
        def isPowerOfTwo(self, n):
            if n == 0:
                return False
            return n & (n - 1) == 0

    # My V
    def power(n):
        while n:
            if n == 1:
                return True
            if n & 1:       #if last bit is not 0
                return False
            n >>= 1

19. Remove bit
--------------
| Remove_bit(num, i): remove a bit at specific position. 
| For example: Input: num = 10101 (21) remove_bit(num, 2): output = 1001 

::

    # My v
    def remove_bit(n, i):
        rs = n & ((1 << i) - 1)  #prep right side
        n = n >> (i + 1)         #cut off right side including index to remove
        n = (n << i) | rs        #add 0s to left side, merge with right
        return n

    # My v2
    def remove_bit(n, i):
        mask = (1 << i) - 1
        rs = n & mask
        ls = n >> (i + 1)
        n = (ls << i) | rs
        return n

    # v1
    def remove_bit(num, i):
        mask = num >> (i + 1)      
        mask = mask << i
        right = ((1 << i) - 1) & num
        return mask | right

**v1 explained**

| E.g. n=10101, i=2
| ``mask = num >> (i + 1)``
| # 10 dropping 2+1, => 10  => Cut off the right
| ``mask = mask << i``
| # adding i=2 0s  => 1000 => Fill in the right side with 0s
 
| ``right = ((1 << i) - 1) & num``
| # 100 - 1 we get 11 => Prep mask to extract the right side (i.e. 1s len of i,
| i.e. len of the right side)
| 11 & n  => Extract the right side
| 00011 &
| 10101
| 00001
| (n compared with 11 evaluates to n but of size those 1s)
 
| ``return mask \| right``
| 1000 \|  => OR merges "N leftside + 0s" with the original right side
| 0001
| 1001















