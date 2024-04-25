Strings Questions Part 1
=========================
106. (LC 125) Test palindromicity (Valid Palindrome)
-------------------------------------------------------
`125. Valid Palindrome <https://leetcode.com/problems/valid-palindrome/>`_
Easy

**Logic**
Use two indices to traverse the string forward and backwards (skipping nonalphanumeric),
which gives us the effect of reversing without using extra space.
(Just reversing the given string and comparing it with the original requires extra space.) ::

    ### Solution 1 (doocs), O(n). Doesn't use extra space.
    class Solution:
        def isPalindrome(self, s: str) -> bool:
            # i moves forward, j backwards
            i, j = 0, len(s) - 1
            while i < j:
                if not s[i].isalnum():    #**
                    i += 1
                elif not s[j].isalnum():  #**
                    j -= 1
                elif s[i].lower() != s[j].lower():  #compare only when both are alnum
                    return False
                else:
                    i, j = i + 1, j - 1
            return True

#** If values are not alphanumeric, we proceed to the next index.

| ### Note that this fails when s='.,' 
| (Expect True, the below code will return False,
| it will end up comparing . != , which it should not compare at all) 

::

    def palindromic(s):
        if len(s) < 2:
            return True
        lp = 0
        rp = len(s) - 1
        while lp < len(s) and rp >= 0:
            while s[lp].isalnum() == False and lp < len(s) - 1:
                lp += 1
            while s[rp].isalnum() == False and rp > 0:
                rp -= 1
            if s[lp].lower() != s[rp].lower():
                return False
            rp -= 1
            lp += 1
        return True

    ### Solution 2 (neetcode) Uses extra space.
    class Solution:
        def isPalindrome(self, s: str) -> bool:
            new = ''
            for a in s:
                if a.isalpha() or a.isdigit():
                    new += a.lower()
            return (new == new[::-1])

    ### My V
    import string
    def is_palindrom(s):
        s2 = ""
        for c in s:
            if c not in string.punctuation and c not in string.whitespace:
                s2 += c.lower()
        return s2 == s2[::-1]

| **Is a string a palindrome** 
| Reads forwards and backwards the same. ::

    def is_palindromic(s):
        return all(s[i] == s[~i] for i in range(len(s) // 2))

    s = "abcba"
    print(is_palindromic(s)) # True

| # Explained
| ~ bitwise not operator.
| Recall that in Python, when NOT is used in expressions, it does what it should - 
| flips bits. But when used not in expressions, ~number = -(number + 1). E.g.:

>>> ~2
-3
>>> ~4
-5

| It is handy for indexing the opposite side of a string/or array.
| s[~i] = s[-(i+1)]
| We use s[i] to traverse a string forwards, s[~i] to traverse backwards.

| //2
| Handles both odd and even length strings.

107. (LC 8) String to Integer
----------------------------------
`8. String to Integer <https://leetcode.com/problems/string-to-integer-atoi/>`_
Medium

| *Interconvert strings and integers*
| E.g. given "123" return 123.
| You should handle negative integers as well.
| Do not use Python int().

::

    ### Solution (EPI book)
    def int_to_str(x):
        is_negative = False
        if x < 0:
            x, is_negative = -x, True
        s = []
        while True:
            s.append(chr(ord("0") + x % 10))  #e.g.x=346, x%10=6
            x //= 10     #346//10=34
            if x == 0:
                break
        return ("-" if is_negative else "") + "".join(reversed(s))

Append to s dif=gits from the end of x.
x=346, s='6', s='64', s='643'. Reverse as the last step.

::

    def string_to_int(s):
        return functools.reduce(
            lambda running_sum, c: running_sum * 10 + string.digits.index(c),
            s[s[0] == "-" :],
            0,
        ) * (-1 if s[0] == "-" else 1)

**int(), str()**

Do consider:

>>> n=12
>>> str(n)
'12'
>>> s='12'
>>> int(n)
12

**functools.reduce(function, iterable[, initializer])**

Apply function of two arguments cumulatively to the items of iterable, from left 
to right, so as to reduce the iterable to a single value. 
For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5). 
The left argument, x, is the accumulated value and the right argument, y, is the update 
value from the iterable. 
If the optional initializer is present, it is placed before the items of the iterable 
in the calculation, and serves as a default when the iterable is empty.

::

    ### Solution to Leetcode, My V (LC accepted T80, S50 %)
    class Solution:
        def myAtoi(self, s: str) -> int:
            num = 0
            reading = False
            sign_set = False
            positive = True
            for c in s:
                if not reading:
                    if c == " " and not sign_set:
                        continue
                    elif c == "+" and not sign_set:
                        sign_set = True
                    elif c == "-" and not sign_set:
                        positive = False
                        sign_set = True
                    elif not c.isdigit():
                        break
                    elif c.isdigit():
                        reading = True
                        num += int(c)
                else:
                    if not c.isdigit():
                        break
                    else:
                        num *= 10
                        num += int(c)
            if not positive:
                num = min(num, 2**31)
                num *= -1
            else:
                num = min(num, 2**31 - 1)
            return num















