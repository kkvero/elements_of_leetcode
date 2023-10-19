
Bit Manipulation Questions Part 1
=================================

1. Hamming weight
-----------------
Count the number of bits that are set to 1 in a positive integer
(also known as the Hamming weight).

Solution::

    # Python
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

    # Python
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

2. Compute the parity of a word
-------------------------------
The parity of a binary word is 1 if the number of 1s in the word is odd; 
otherwise, it is 0. For example, the parity of 1011 is 1, and the parity of 
10001000 is 0. 

**Solution 1**::

    # Python
    def f9(w):
        bins = ""
        for i in w:
            bins += bin(ord(i))[2:]
        print(bins)
        bins = int(bins, 2)
        count = 0
        while bins:
            count ^= bins & 1
            bins >>= 1
        return count

    print(f9('h'))
    print(f9('ha'))
    # OUT
    # 1101000
    # 1
    # 11010001100001
    # 0

**Solution 2**
*O(n), n is word size*

* Tools that we use here

| :py:func:`format()`
| Converts a character into binary.

>>> format(ord('d'), 'b')   #char 'd'
'1100100'

How to convert string to binary.

>>> st = "hello world"
>>> ' '.join(format(ord(x), 'b') for x in st)
'1101000 1100101 1101100 1101100 1101111 100000 1110111 1101111 1110010 1101100 1100100'

Using :py:class:`bytearray`

>>> ' '.join(format(x, 'b') for x in bytearray(st, 'utf-8'))
'1101000 1100101 1101100 1101100 1101111 100000 1110111 1101111 1110010 1101100 1100100'

Note that type of the above object will still be - <class 'str'>, 
so we won't be able to feed such a value to a function that expects a binary type.
BUT. To convert, just use int().

*Solution.*
Brute-force, iteratively test the value of each bit::

    def parity(x):
        result = 0
        while x:
            result ^= x & 1
            x >>= 1
        return result

    word = 'so'  # 11100111101111  - 11 1s
    word_bin =int(''.join(format(ord(x), 'b') for x in word))
    print(parity(word_bin))  # outputs 1

| Step by step explanation
| 1) x=11100111101111 looking at 1 on the right side
| res=0, x=11100111101111, x&1 = 1, (res 0 ^ 1) = 1
| x >>= 1, x= 1110011110111 
| 2) x=1110011110111 looking at 1 on the right side
| res=1, x=1110011110111, x&1 = 1, (res 1 ^ 1) = 0
| x >>= 1, x= 111001111011 
| 3) x=111001111011 looking at 1 on the right side
| res=0, x=111001111011, x&1 = 1, (res 0 ^ 1) = 1
| x >>= 1, x= 111001111011 

Shorter::

    For x=11100111101111
    res  encountered
    1) 0    1 -> r=1 odd
    2) 1    1 -> r=0 even
    3) 0    1 -> r=1 odd

**Solution 3**, *O(k), k is the number of bits set to 1*

| We are going to use this trick
| x&(x-1) trick
| ==> dropping the rightmost 1 <==
| ("erasing the lowest set bit")
| x&(x-1) = x with its lowest set bit erased
| Example
| if x=(00101100),then x-1= (00101011),
| so x &(x - 1) = (00101100)&(00101011) = (00101000)
| Again note 00101-1-00 becomes 00101-0-00

*Actual solution.*
Using the above trick, we end up counting only 1s::

    def parity(x):
        result = 0
        while x:
            result ^= 1
            x &= x-1    #drops the rightmost 1, loop goes on until we run out of 1s
        return result

**Solution 3**, *O(log n), n is the word size*::

    # Python
    def parity(x):
        x ^= x >> 32
        x ^= x >> 16
        x ^= x >> 8
        x ^= x >> 4
        x ^= x >> 2
        x ^= x >> 1
        return x & 0x1

Recall we have a 64 bit word.
The parity of (b63,b62,. .. ,b3,b2, b1, b0) equals the parity of the XOR of
(b63,b62,. . . ,b32) and (b31, b30,. .., b0).
Note that the leading bits are not meaningful, and we
have to explicitly extract the result from the least-significant bit.

.. _swap-bits-label:

3. Swap bits
------------
A 64-bit integer can be viewed as an array of 64bits, with the bit at index 0 corresponding to the
least significant bit (LSB, see :ref:`lsb-label`), and the bit at index 63 corresponding to the most significant bit (MSB).
Implement code that takes as input a 64-bit integer and swaps the bits at indices i and j. 

*Example*::

    # Visualize
    # Note, index 0 is on the right.
    # E.g. bit swapping for an 8-bit integer.
    # Original:
    # 0 >1< 0 0 1 0 >0< 1
    # MSB               LSB (ind 0)
    # ind 7
    # Swapped:
    # 0 >0< 0 0 1 0 >1< 1

**Solution**

The time complexity O(1) independent of the word size::

    # Python
    def swap_bits(x, i, j):
        if (x >> i) & 1 != (x >> j) & 1:
            bit_mask = (1 << i) | (1 << j)   #**1
            x ^= bit_mask                    #**2
        return x

    number = 997
    print(swap_bits(number, 3, 6))  #941

| #**1 mask gives us 1s at concerning indexes, e.g. 100100 (when j=5, i=2)
| #**2 having such a mask, xoring it with original number, changes bits at indexes we are concerned with. 

Checking:

>>> bin(997)
'0b1111100101'
>>> bin(941)
'0b1110101101'

*Explained*

Because a bit can only have two possible values, 1 or 0. 
It makes sense to first test if the bits differ. If they do not, the swap wouldn't
change the integer.
Again, because only 2 possible values, flipping has the effect of a swap.

| E.g. x=997, i=3, j=6
| if (x >> i) & 1 != (x >> j) & 1:
| Use bit shift operator to check values at corresponding indexes.
| x>>3 is 1111100>>101 looking at 0
| x>>6 is 1111>>100101 looking at 1

| bit_mask = (1 << i) | (1 << j)
| 1<<3 is 1000
| 1<<6 is 1000000
| 1000 | 1000000 is 1001000 #OR operator applies logical OR to each bit
| The bit_mask creates a number that has 1s at the indexes in question.
| Here 1 at index 6 and 3.
| Then using ^ XOR having 1 in mask changes whatever value at the index in x.
| If x=1, x^1 changes x to 0. If x=0, x^1, changes x to 1.

| x ^= bit_mask
| 1111100101 ^  #our x
| ---1001000    #our mask, 1s at index 3 and 6
| 1110101101    #flipped bits at index 3 and 6








