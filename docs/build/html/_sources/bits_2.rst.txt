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


















