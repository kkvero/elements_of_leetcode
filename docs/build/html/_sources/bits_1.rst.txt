
Bit Manipulation Questions Part 1
=================================

1. Hamming weight
-----------------
Count the number of bits that are set to 1 in a positive integer
(also known as the Hamming weight).

Solution::

    def count_bits(x):
        num_bits = 0
        while x:
            num_bits += x & 1
            x >>= 1
        return num_bits

    x = 12
    print(bin(x))
    print(count_bits(x))
    # OUT
    # 0b1100
    # 2

Other solutions::

    def count_ones_recur(n):
        """Using Brian Kernighan’s Algorithm. (Recursive Approach)"""
        if not n:
            return 0
        return 1 + count_ones_recur(n & (n-1))

    def count_ones_iter(n):
        """Using Brian Kernighan’s Algorithm. (Iterative Approach)"""
        count = 0
        while n:
            n &= (n-1)
            count += 1
        return count

    def hammingWeight(n):
        return bin(n).count('1')

* Explained

| ``number & (number - 1)``
| Takes away one bit with value 1.
| E.g.:

>>> bin(52)
'0b110100'
>>> 52&(52-1)
48
>>> bin(48)
'0b110000'

So the loop will work for just as many times as there are 1s.




