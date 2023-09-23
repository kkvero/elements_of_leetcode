Bit Manipulation Knowledge Base
===============================

**bit vs byte**
The smallest unit of information, called an octet or a byte, comprises eight bits 
that can store 256 distinct values.

Bitwise operators
-----------------
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

^, XOR
^^^^^^
Sets each bit to 1 if only one of two bits is 1, x ^ y

>>> '0b{:04b}'.format(0b1100 ^ 0b1010)
'0b0110'

``a ^= b`` is equivalent to ``a = a ^ b``

\|, OR	
^^^^^^
a | b, Each bit position in the result is the logical OR of the 
bits in the corresponding position of the operands. 1 if either is 1, otherwise 0.
E.g.: 1000 \| 10 is 1010.

.. note::

    | **Merging numbers**

    It can also be said that OR operator merges two numbers (101 \| 101000 = 101101)::

        ---101 | 
        101000 = 
        101101

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

.. note::

    | **Get bits of n to the right of index**

    (i is the given index)
    We know how to get bits to the left, just n<<i.
    How to get bits to the right of i?
    We use the combination of the facts that ((1<<i)-1) leaves us with all 1s.
    n compared with 1s evaluates to n but of size those 1s.

    | E.g. n=10101, i=3.
    | right=((1<<i)-1) & n
    | 1<<3=1000, -1 leaves us with 111.
    | 111&10101 = 101 (it is just the side of n to the right of i unchanged)









