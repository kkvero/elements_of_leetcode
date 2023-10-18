Knowledge Base
==============
**bit vs byte**
The smallest unit of information, called an octet or a byte, comprises eight bits 
that can store 256 distinct values.

Bitwise operators
-----------------
.. admonition:: Overview of bitwise operators in Python

    ::

        & 	a & b   Bitwise AND
        | 	a | b   Bitwise OR
        ^ 	a ^ b   Bitwise XOR (exclusive OR)
        ~ 	~a      Bitwise NOT
        << 	a << n  Bitwise left shift
        >> 	a >> n  Bitwise right shift

&, AND 
^^^^^^
a & b, Each bit position in the result is the logical AND of the bits in the 
corresponding position of the operands. (1 if both are 1, otherwise 0.)
Arithmetically, this is equivalent to a product of two bit values.
e.g. 5&3 is 1::

    101 &
     11
    001

# Note, the resulting value is the len of the shorter evaluated numbers
(When operands have unequal bit-lengths, the shorter one is automatically padded 
with zeros to the left, e.g. 101 & 11 -\> 101 & 011).

Another example::

    10101 &
    --111
    --101  # We end up with a shorter value

.. admonition:: Get bits of n to the right of index

    (i is the given index)
    We know how to get bits to the left, just n<<i.
    How to get bits to the right of i?
    We use the combination of the facts that ((1<<i)-1) leaves us with all 1s.
    n compared with 1s evaluates to n but of size those 1s.

    | E.g. n=10101, i=3.
    | right=((1<<i)-1) & n
    | 1<<3=1000, -1 leaves us with 111.
    | 111&10101 = 101 (it is just the side of n to the right of i unchanged)

\|, OR	
^^^^^^
a | b, Each bit position in the result is the logical OR of the 
bits in the corresponding position of the operands. 1 if either is 1, otherwise 0.
E.g.: 1000 \| 10 is 1010.

.. admonition:: Merging numbers

    It can also be said that OR operator merges two numbers (101 \| 101000 = 101101)::

        ---101 | 
        101000 = 
        101101

^, XOR
^^^^^^
Sets each bit to 1 if only one of two bits is 1, x ^ y.

>>> '0b{:04b}'.format(0b1100 ^ 0b1010)
'0b0110'

``a ^= b`` is equivalent to ``a = a ^ b``

~, NOT 	
^^^^^^
Unary, Inverts all the bits, ~x.
0s become 1s and vice versa. 
But, note, in REPL:

>>> ~55
-56

NOT operator works properly only when you use it with a mask of 1s the size of the
original number.

| To get ~55 = 8 
| ~110111  (55)
| 0001000   (8)

>>> ~55 & int('111111', 2)   
8

<<, >>, shift
^^^^^^^^^^^^^
| Arithmetic shift operators.

* In visual terms:

| << - zero fill left shift, add a zero on the right (doubles the value)
| >> - shift right, let the rightmost bits fall off (halves the original value)

It's like we have a horizontal line/wall on the right side::

     1011010|  original number
    10110100|  << 1  Zeros spur up
      101101|  >> 1  Everything disappears on the right side of the wall.

>>> bin(90)
'0b1011010'
>>> bin(90 << 1)
'0b10110100'
>>> bin(90 >> 1)
'0b101101'
>>> bin(90 >> 2)
'0b10110'

- In terms of meaning:

An arithmetic right shift is equivalent to floor division by a power of 2.
(If it results a fraction, the right shift operator automatically floors the result.)

>>> 30 >> 1
15
>>> bin(30); bin(15)
'0b11110'
'0b1111'

Shifting a single bit to the left by one place doubles its value.

>>> 20 << 1
40
>>> 20 << 2
80   # Moving two places, quadruples

I.e. in general: a << n = a * 2**n

Functions, methods for numeric types
------------------------------------
:py:func:`abs` , abs(-34.5), 
:py:func:`math.ceil` , math.ceil(2. 17), 
:py:func:`math.floor` , math.floor(3.14),
:py:func:`min` , min(x, -4), 
:py:func:`max` , max(3.14, y), 
:py:func:`pow` , pow(2.71, 3.14) (or 2.71 ** 3.14), 
:py:func:`math.sqrt` , math.sqrt(225).

| *About:*
| ``math.ceil(x)`` - smallest number greater than x
| E.g.: ``math.ceil(2.17)`` -> 3
| ``math.floor(x)`` - largest integer not greater than x
| E.g.: ``math.floor(3.14)`` -> 3

| **Interconvert integers and strings**
| str(x), int('x'), float('x'), ord(), chr()
| format()

>>> format(ord('d'), 'b')   # convert char 'd' into binary
'1100100'
>>> s = 'fdr'
>>> [ord(x) for x in s]
[102, 100, 114]
>>> chr(102)
'f'

| **Infinity**
| float('inf'), float('-inf'), -float('inf'), -float('infinity')

| **random module**
| random.randrange(28), random.randint(8,1.6), random.random(), random.shuffle(A), random.choice(A)

| **bit_length()**
| Each character corresponds to a decimal, which in its turn corresponds to a binary.

>>> [ord(character) for character in "€uro"]
[8364, 117, 114, 111]
>>> (42).bit_length()
6
# Because:
>>> bin(42)
'0b101010'

| **int.bit_count()**
| :py:meth:`int.bit_count` Return the number of ones.

>>> n=99
>>> bin(n)
'0b1100011'
>>> n.bit_count()
4

Common tasks
------------
Count bits
^^^^^^^^^^
Ways to count turned on bits.

| E.g. n=33

>>> bin(33)
'0b100001

1. Working with a string representation:

>>> bin(n).count('1')
2

2. int method:

>>> n.bit_count()
2

3.1 Bit operators, n & 1::

    def count1s(n):
        count = 0
        while n:
            if n & 1:
                count += 1
            n >>= 1
        return count
    n=33
    print(count1s(n))  # 2

3.2 Bit operators more efficient, number & (number - 1). 
Takes away one bit with value 1.
Useful in loops, the loop will work for just as many times as there are 1s.
E.g.:

>>> bin(52)
'0b110100'
>>> 52&(52-1)
48
>>> bin(48)
'0b110000'

+, - 1
^^^^^^
::

    1000 - 1 = 111   #called turning off the rightmost bit operator
    1000 + 1 = 1001

>>> (1<<3)-1 # 7
>>> bin(7)   # '0b111'
>>> (1<<3)+1 # 9
>>> bin(9)   # '0b1001'

Even and odd numbers
^^^^^^^^^^^^^^^^^^^^
Least-significant bit, determines if the number is even or odd.
That's why we can always use n&1 to check if a number is even or odd.
(n&1 performs AND comparison of 1 AND the LSB of a number.)
&1 is more efficient than n%2 == 0 check.

>>> bin(2); bin(3)
'0b10'
'0b11'
>>> 2 & 1 #0
>>> 3 & 1 #1

In code that checks with &1, we should negate the statement, as 0 means False, but
here it means Yes, LSB of n is 0, thus n is even::

    n = 2
    def is_even(n):
        return not n & 1
    print(is_even(n))  # True

Zero pad 
^^^^^^^^
How to zero pad binaries:

>>> f'{5:06b}'
'000101'

Use binaries verbatim
^^^^^^^^^^^^^^^^^^^^^

>>> age = 0b101010
>>> 0b101010  # instead of the more explicit int('0b101010', 2)
42

But we need int() when binary nums are generated dynamically in code.

Extract LSBs
^^^^^^^^^^^^
I.e. extracting the right side of a number. When it is used: it is one of the steps
when we need to change some bits in the middle of an integer in binary representation. 
The steps would be 1) extract the right side, 2) change LSBs of the remaining left side,
3) stick the right side back in.

To Extract the rightmost LSBs we create a mask of 1s::

    # Here i is the index, which at the same time is the len of mask.
    (1 << i) - 1  
    # e.g. i=2
    # 100
    # 100 - 1 -> 11

Having got 1s, you compare it with the original number using &.
You end up with the right part of a number exactly the size of 1s.
E.g.::

    ---11 &  # mask
    10101    # our original number
    ---01    # extracted LSBs

Merge
^^^^^
Informally speaking, how to stick the right side of a number back in? I.e. merge 
two binary numbers.

After changing a number in some way (del, flip bit), you will need to stick
the right side back in. Use OR operator::

    mask | right

We use the fact that when 0s are compared with a number, 0s turn into that number.
<right> is e.g. the last two digits on the right of the original number.
<mask> is left side + 0s of len that equals len right. E.g.::

    # Original number=1001
    # right=01
    # mask= 1000 (which we got via, if deleting at index, mask=n>>i, mask=mask<<i)
    # Basically mask is a number ending with 0s.
    # Where 0s were, we place the 'right'.
    1000
    --01
    1001

Vocabulary
----------
MSB, LSB
^^^^^^^^
Often the bits in a binary representation of a number are referred to as the MSB or LSB.
It helps to understand which bits will be effected by an operation.

| MSB - most significant bit (the leftmost)
| LSB - least significant bit (the rightmost)

>>> bin(2)
'0b10' # MSB=1, LSB=0

Sign bit
^^^^^^^^
Signed binary integers are encoded negative numbers.
If MSB is 1, then the number is negative (normally in programming languages).
Python has no sign bit.
Integers in Python can have an infinite number of bits.

Masks
^^^^^
Bitmasks allow to isolate particular bits in a binary representation of an integer.
E.g. get the 2 LSBs of a decimal 42.

>>> bin(42) #'0b101010'
>>> mask = (1<<2)-1 #100-1=11, why <<, it makes sure the result will be in binary
>>> mask # 3
>>> bin(mask) #'0b11'
>>> 42 & mask # 2
>>> bin(2) #'0b10' got our 2 LSBs

``hex()``
Hexadecimals are often used to represent masks.

>>> mask = 0b11111111  # Same as 0xff or 255

















