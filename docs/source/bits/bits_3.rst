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




