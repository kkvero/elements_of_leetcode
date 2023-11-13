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