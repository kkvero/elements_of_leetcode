
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
    def count_set_bits(n):
        """Use the built in."""
        return n.bit_count()

    def hammingWeight(n):
        return bin(n).count('1')

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

**My V2** (Simplified, if we are given just one word.)
::

    def parity(s):
        a = [ord(c) for c in s]
        cnt = [n.bit_count() for n in a]
        total = sum(cnt)
        return total & 1

    print(parity("dune")) # 1

    # With print statements
    def parity(s):
        a = [ord(c) for c in s]
        print(a)
        cnt = [n.bit_count() for n in a]
        print(cnt)
        total = sum(cnt)
        print(total)
        return total & 1

    print(parity("dune"))
    #[100, 117, 110, 101]
    #[3, 5, 5, 4]
    #17
    #1

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

**Solution 4**, *O(log n), n is the word size*::

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

4. (LC190) Reverse bits
-----------------------
| Example 1:
| Input: n = 00000010100101000001111010011100
| Output:    964176192 (00111001011110000010100101000000)

**Solution 1**::

    def reverse_bits(n):
        m = 0
        while n:
            m = (m << 1) + (n & 1) 
            n >>= 1
        return m

    print(reverse_bits(600))   # returns 105

    # V2
    def rev_bits2(n):
        m = 0
        while n:
            bit = n & 1
            m <<= 1
            m |= bit
            n >>= 1
        return m

Checking

>>> bin(600)
'0b1001011000'
>>> bin(105)
'0b1101001'

*Explained*

``(m << 1) + (n & 1)`` using bitwise operators, both
values are in the same format, and you can simply concatenate the result of n&1 to m.
FYI the result of n&1 is whatever the last bit of n is. E.g. if n ends with 0,
0&1 returns 0, 1&1 would return 1.

**Solution 2** (When you do not know better.)::

    class Solution:
        def reverseBits(self, n):
            s = bin(n)[2:]
            s = "0"*(32 - len(s)) + s  # we zero pad
            t = s[::-1]
            return int(t,2)

    # Simplified, my v
    def rev_bits(n):
        s = str(bin(n))[2:]
        return "0b" + (s[::-1])

    print(rev_bits(96)) # '0b0000011' 

**Solution 3** (My variant)::

    def reverse_bits(n):
        s = bin(n)[2:]
        L = list(s)
        for i in range(0, len(s)//2):
            L[i], L[len(s)-1-i] = L[len(s)-1-i], L[i]
        s = ''.join(L)
        return s, int(s, 2)

    print(reverse_bits(40))  #OUT ('000101', 5)

>>> bin(40)
'0b101000'

5. Find a closest integer with the same weight
----------------------------------------------
The weight of an integer is the number of bits set to 1 in its binary representation.
E.g. x = 92 which is (1011100), the weight is 4.

(Write a program which takes as input a nonnegative integer x and returns a number y 
which has the same weight as x and their difference \|y-x| is as small as possible.
Assume x is not 0 or all 1s; integer fits in 64 bits.)

**V2** ::

    def same_weight(x):
        for i in range(x.bit_length()):
            if ((x >> i) & 1) != ((x >> (i + 1)) & 1): #if bit at i and bit at i+1 are not the same
                left = x & ((1 << i) - 1)
                right = x >> i
                right_flipped = right ^ ((1 << 2) - 1)  #flipping 2 last bits of right
                merged = (right_flipped << i) ^ left    #prep end with 0s and XOR merge with left
                break
        return merged

    print(same_weight(92)) # 90

**Solution 1** 
(O(n), n is integer width)

*Logic.*
To make sure that x and y differ as little as possible, we have to change LSB (least
significant bits) of x. I.e. we swap the two rightmost consecutive bits that differ.
(Since we must preserve the weight, the bit at index i and at i+1 have to be different)::

    def closest_int_same_bit_count(x):
        NUM_UNSIGNED_BITS = 64
        for i in range(NUM_UNSIGNED_BITS - 1):
            if (x >> i) & 1 != (x >> (i+1)) & 1:   #if bit at i and bit at i+1 are not the same
                x ^= (1 << i) | (1 << (i+1)) #Swaps bit i and bit (i+1)
                return x
        # Raise error if all bits of x are 0 or 1 
        # (we looped through x without finding deffering bits)
        raise ValueError('All bits are 0 or 1')

    print(closest_int_same_bit_count(8))  #4

*My note.* We don't have to check if the number with swapped bits at i and i+1 is
the closest to x, because it is, because we swap LSBs. So as soon as we found 
differing bits at i and i+1, we swap and return. So it comes down to just: 

1. finding differing bits
2. swapping

| *Explained.*
| E.g. x = 10101
| # Check if bit at i and bit at i+1 are not the same
| if (x >> i) & 1 != (x >> (i+1)) & 1:
| i=0
| x >> i = 10101, (10101 & 1) = 1  <---we just get the bit at i of x
| x >> i+1 = 1010, (1010 & 1) = 0  <-- we get the bit at i+1 of x
| We make the (num & 1) comparison to get rid of the bits on the left.
 
| # Swap bit i and bit (i+1)
| x ^= (1 << i) | (1 << (i+1))  
| (1 << i) | (1 << (i+1))  -> i=0 -> 
| 01 | 
| 10
| 11
| x = x^11 -> 
| 10101 ^
| 00011
| 10110  # We swapped bits

6. Compute x**y
---------------
| *Logic*
| The aim is to get more work done with each multiplication,
| i.e. not x**3=x**2 * x,
| but rather each time multiplying by power of 2 of that number.
| Iterated squaring: x, x**2, (x**2)**2 = x**4, (x**4)**2.
| This is when y is the power of 2.
| To find if y is the power of 2, we look at its last bit, 0->even y, 1-> odd y.

Also recall the property of exponentiation: x**(y0+y1) = x**y0 * x**y1

Also FYI 0b1010 = 101 + 101 (14 = 7+7). 
I.e. dropping the LSB of a number (when LSB is 0), we get a smaller number,
twice as small as the original,and hence adding two such numbers gets us the original.
If the original number ends with 1, e.g. 101, then 101 = 100+1

| E.g. y=0b1010 (binary repr)
| x**1010 = x**(101 +101) = x**101 * x**101
| Going further, x**101 = x**(100 + 1) = x**100 * x = x**10 * x**10 * x
| Generalizing, if the least significant bit of y is 0, the result is (x**y/2)**2,
| otherwise, it is (x**y/2)**2 * x  ( ===> plus <* x> )

The only change when y is negative is replacing x by 1/x and y by -y.

**Solution**::

    def power(x, y):
        result, power = 1.0, y
        if y < 0:                         #if y is negative
            power, x = -power, 1.0/x
        while power:
            if power & 1:                 # if LSB of y is 1  (1&1=1, if it were 0&1=0)
                result *=x                # *x in that case #**NOTE
            x, power = x * x, power >> 1  #**adds power to x, drops bit from y
        return result

| #**NOTE
| When we dropped all LSBs from power, we end up with power=1 in the last loop.
| So we will definitely go into result \*=x.
| So we will basically multiply the result by the accumulated x.

| *Example*
| x=3, y=9 (3**9 = 19683), bin(9)=1001
| res=1.0, power=y
| 1)1001 & 1 = 1, so res*=x => 1*3=3
| res=3, x=3*3=9, power=100
| 2)x=9*9=81, power=10 (res unchanged = 3)
| 3)x=81*81=6561, power=1
| 4)1&1=1, so res=res*x= 3 * 6561 = 19683
| x gets calculated also = x*x = 6561*6561, but it won't be used
| power = 0, we return res, which is 19683

*Time complexity O(n).*
The number of multiplications is at most twice the index of y's MSB, implying 
an O(n) time complexity.

7. Reverse digits
-----------------
Note, reverse not bits, but digits.
So when give an integer like 456, return the corresponding integer in reverse order, 
like 654.

The approach when we convert to string and then reverse the string - 
is the brute force approach::

    def f22(n):
        n = str(n)[::-1]
        return int(n)
    print(f22(457))

**Solution.**
(O(n), n is the number of digits in the input number.)::

    def reverse(x):
        result, x_remaining = 0, abs(x)
        while x_remaining:
            result = result * 10 + x_remaining % 10
            x_remaining //= 10
        return -result if x < 0 else result

    print(reverse(5734))

    # Remake
    def reverse_digits2(n):
        new = 0
        while n:
            res = n % 10     #get last (i.e. 6)
            new = (new * 10) + res
            n = n // 10      #get remaining (i.e. 45)
        return new

    print(reverse_digits2(456))  # 654

| *Explained* (main version)
| Here we avoid having to form a string.
| Key points:
| Here we use methods to extract the least significant digit, not bit, but digit of a decimal. (using %, // by 10)

| ``x_remaining % 10``
| n%10 gives us the last digit of n. E.g. 1132%10=2 (the remainder of division by 10)
| ``result * 10``
| We have to place the last digit from x_remaining to the position of the first digit of result.
| Multiplying by 10 we add a 0. 
| E.g. 5734 to 4375
| 1)0*10 + 5734%10 = 0 + 4 (put 4 as first digit, result=4)
| x_remaining = 5734 // 10 = 573 
| 2)4*10=40 + 3=43
| 3) 43*10=430 + 7 = 437 etc.
| ``x_remaining //= 10``
| n//10 takes away the last digit. E.g. 435//10 = 43

The algorithm takes into account that we may be given a negative number,
so we work with its abs() value. 

8. Check if a decimal integer is a palindrome
----------------------------------------------

| *Task.* Return True or False.
| E.g. these are palindromes: 0,1,7,11,121,333,2147447412
| Not palindromes: -1,12,100,2147483647
| Note, single digit numbers are palindromes, all negative numbers are not palindromes.

*Brute force*
Convert to string and compare least and most significant digits in its string form,
working inwards, stopping if there is a mismatch.

*Complexity*
-> When converting to string first. 
The time and space complexity are O(n), n is the number of digits in the input
-> We are going to come up with a solution that has time complexity O(n),
space complexity O(1).
Note, we are going to use the same approach as with a string (comparing the 
leftmost and rightmost digits, working inward), but without converting to string.

*Alternatively*, we could reverse the digits in the number and see if it is unchanged.

**Solution** (*Real Python* version [:ref:`3 <ref-label>`])
::

    def is_palindrome(num):
        # Skip single-digit inputs
        if num // 10 == 0:
            return False
        temp = num
        reversed_num = 0

        while temp != 0:
            reversed_num = (reversed_num * 10) + (temp % 10)
            temp = temp // 10

        if num == reversed_num:
            return True
        else:
            return False


**Solution**::

    import math
    def is_palindrome_number(x):
        if x <= 0:
            return x == 0
        num_digits = math.floor(math.log10(x)) + 1
        msd_mask = 10**(num_digits -1)
        for i in range(num_digits // 2):
            if x // msd_mask != x % 10:
                return False
            x %= msd_mask #Remove the most significant digit of x
            x //= 10      #Remove the least significant digit of x
            msd_mask //= 100
        return True

    print(is_palindrome_number(2))
    print(is_palindrome_number(22))
    print(is_palindrome_number(223432))
    # True
    # True
    # False

    ### My rewrite
    def is_palindrome(n):
        if n < 0:
            return False
        elif n < 10:
            return True
        le = math.floor(math.log10(n)) + 1
        mask = 10 ** (le - 1)
        for _ in range(le // 2):
            msd = n // mask
            lsd = n % 10
            if msd != lsd:
                return False
            n = n % mask
            n = n // 10
            mask = mask // 100
        return True

*Logic*
We come up with expressions that extract the least significant digit and the 
most significant digit of the input integer (without converting it).

| -> E.g. x = 151751
|     ``if x <= 0:``
|         ``return x == 0``
| # Account for negative input, not palindrome if negative, but if 0, then x is a palindrome.
 
|     ``num_digits = math.floor(math.log10(x)) + 1``
| Number of digits in a number x is math.log10(x) + 1
| x = 151751 -> log10(151751)=5.18..+1 = 6.18
| (What is log10, it is our number / 10 that many times. or 10**5.18..)
| math.floor(6.18) = 6 (gives us the lowest whole integer of a float)
 
|     ``msd_mask = 10**(num_digits -1)``
| # Most significant digit mask
| E.g. on first iteration, 10**5=100000
 
|         ``if x // msd_mask != x % 10:``
|             ``return False``
| # if most significant != least significant, then not a palindrome
| most significant digit = 151751 // 100000 = 1 (divisible by whole is just once)
| least significant digit = 151751 % 10 = 1 (remainder is 1)
 
|         ``x %= msd_mask #Remove the most significant digit of x``
|         ``x //= 10      #Remove the least significant digit of x``
| # This is fabulous
| To remove the most significant digit we use the operator % that we used to find 
| the least sign. digit, with the old mask.
| x=151751 % 100000 = 51751  (compare with finding most sign x // msd_mask )
| Remove least sign.
| x=51751 // 10 = 5175   (compare with finding least sign x % 10)
 
|         ``msd_mask //= 100``
| # We decrease our mask for msd by 2 places (because we removed 2 digits, most and
| least significant ones).
| 100000 // 100 = 1000  (which correctly corresponds to msd_mask = 10**(num_digits -1))

.. admonition:: The take away: how to get the least and most significant digit

    | x = 157
    | # Get the last digit (easiest)
    | |:large_blue_diamond:| x % 10 |:large_blue_diamond:|
    | (e.g. 157 % 10 = 7)
    | (150/10 and remainder 7)
    | # Get the first digit
    | |:large_blue_diamond:| x // (10**d) |:large_blue_diamond:| (where d is (num_digits - 1)) 
    | x // 100 = 157 // 100 = 1 
    | (157 divides by 100 completely only 1 time)
    | # How you get the num_digits
    | ``num_digits = math.floor(math.log10(x)) + 1``

9. Add without using +
----------------------

| Or "add bitwise operator".
| The following code adds two positive integers without using the '+' operator. 
| The code uses bitwise operations to add two numbers.
| Input: 2 3 Output: 5

::

    def add_bitwise_operator(x, y):
        while y:
            carry = x & y   #identify which bits are both 1s
            x = x ^ y       #which bits are 0 and 1
            y = carry << 1  #y is now bits to carry towards the MSB
        return x

| # Loop 1
| x=0b10
| y=0b11
| carry=0b10
| x=0b01
| y=0b100
| End of loop.
| # Loop 2
| c= x&y = 01&100=000
| x= x^y = 01^100 = 101
| y= c<<1 = 0<<1 = 00
| y is no more, end of while y
| result is return x, which is 101 (5)
 
| ``carry = x & y``
| We get carry by using & on the numbers.

When adding, we have to carry 1 bit when both bits are 1. So carry is non zero 
when it has 1s. The & operator will identify if x and y have 1s at the same indexes, 
and thus we have to carry:

| 10 &
| 11
| 10 # carry
 
| ``x = x ^ y``
| Gives us the bits that don't require the carry. Plainly adds 0+1 (as xor identifies
| exactly those bits).
| 10 ^
| 11
| 01

| ``y = carry << 1``
| The purpose of the carry is to move the bit that needs to be carried one position closer to the MSB (i.e. to the left).
| So y now carries that bit, moreover at the necessary position.
| (first loop) c=10, y = 10 << 1 = 100
| And hence the loop while y will go on while there are bits to carry.

