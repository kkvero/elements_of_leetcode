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

108. (LC 171) Excel Sheet Column Number
-------------------------------------------
Easy
(Compute the spreadsheet column encoding)

Spreadsheets have column names 'A', 'B'...'Z', 'AA', 'AB'...'AAA', 'AAB'..

- Implement a function that converts a spreadsheet column id to the corresponding integer, with 'A' corresponding to 1.

You would return 4 for 'D', 27 for AA, 702 for ZZ.

- How would you test the function? (Test edge cases and a few random ones.)

| Logic:
| We should convert a string representing a base-26 number to the corresponding integer.
| We take that A corresponds to 1, not 0.
| So we use the 'string to integer' conversion.

**Solution 1** [:ref:`2 <ref-label>`], O(n) ::

    import functools

    def convert_col(col):
        return functools.reduce(
            lambda result, c: result * 26 + ord(c) - ord("A") + 1, col, 0
        )

    ### My V2 (when you do know how ord() works) (LC accepted 97,91%)
    class Solution:
        def titleToNumber(self, s: str) -> int:
            res=0
            for char in s:
                res = (res * 26) + ((ord(char) - ord('A')) +1)
            return res 

| Note:
| -ord(char1) - ord(first char) = char value a=1, b=2 etc.
| -because a=1, not a=0, we add +1

>>> ord('a')-ord('a')
0

-ord('a'), ord('A') are different values! ::

    ### My V1 simple (LC accepted: 13, 95%)
    #(when you don't know how ord works)
    import string
    class Solution:
        def titleToNumber(self, s: str) -> int:
            letters = string.ascii_uppercase
            d={}
            cnt=1
            for c in letters:
                d[c]=cnt
                cnt+=1
            res=0
            for char in s:
                res = res * 26 + d[char]
            return res 

    ### Solution
    import functools, string
    def convert_col2(col):
        return functools.reduce(
            lambda result, c: result * 26 + string.ascii_uppercase.index(c) + 1, col, 0
        )

    col = "Z"
    col2 = "AB"
    col3 = "ZZ"

    print(convert_col2(col))  # 26
    print(convert_col2(col2))  # 28
    print(convert_col2(col3))  # 702

108.2 (LC 168) Excel Sheet Column Title
------------------------------------------
`LC 168. <https://leetcode.com/problems/excel-sheet-column-title/description/>`_
Easy ::

    ### My rewrite (LC accepted)
    class Solution:
        def convertToTitle(self, n: int) -> str:
            ans=''
            while n >0:
                res=(n-1) % 26        #**off by 1, because A=1
                res = chr(ord('A') + res)
                ans += res
                n = (n-1)//26         #**off by 1, because A=1
            return ans[::-1]


    ### Solution 1 (neetcode)
    class Solution:
        def convertToTitle(self, columnNumber: int) -> str:
            # Time: O(logn) - Log base 26 of n
            res = ""
            while columnNumber > 0:
                remainder = (columnNumber - 1) % 26
                res += chr(ord('A') + remainder)
                columnNumber = (columnNumber - 1) // 26

            return res[::-1] # reverse output

| E.g. n=701 (maps to 'ZY')
| (701-1)%26=24
| ord('A')+24=89
| chr(89)='Y'
| n= (n-1)//26  (n-1 takes care of the fact that A=1 in task description)
| Repeat
 
| -Time complexity
| In each loop we divide by 26. So (log base 26 of n).
| The base of the logarithm is not relevant in big O, so O(log N).
| Space O(1).

109. (LC 186) Reverse Words in a String II
--------------------------------------------
(Reverse all the words in a sentence)

| Task 1:
| For example, "Alice likes Bob" transforms to "Bob likes Alice". 
| We do not need to keep the original string.
 
| LC task:
| Solve the problem in-place, i.e. without allocating extra space.
| Input: s = ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
| Output: ["b","l","u","e"," ","i","s"," ","s","k","y"," ","t","h","e"]
 
| Difference: 
| In task 1 we are given a string encoded as bytearray.
 
| Logic:
| Reverse all characters, then reverse again but individual words.
| "ram is costly" -> "yltsoc si mar" -> "costly is ram"

| **Solution** [:ref:`2 <ref-label>`]
| O(n) time, O(1) space 

::

    # s is bytearray
    def reverse_sentence(s):
        # Reverse the whole string
        s.reverse()

        def reverse_word(s, start, end):
            while start < end:
                s[start], s[end] = s[end], s[start]
                start, end = start + 1, end - 1

        start = 0
        while True:
            end = s.find(b" ", start)
            if end < 0:
                break
            reverse_word(s, start, end - 1)
            start = end + 1
        # reverse the last word
        reverse_word(s, start, len(s) - 1)
        return s

    s = "ram is costly"
    s_b = bytearray(s, "UTF-8")
    print(reverse_sentence(s_b))  # bytearray(b'costly is ram')

    ### My V
    def f(s):
        s = s[::-1]
        p1 = p2 = 0
        while p2 < len(s):
            if s[p2] == " ":
                if p1 == 0:
                    s[p1:p2] = s[p2 - 1 : None : -1]  #**otherwise doesn't work when p1=0
                else:
                    s[p1:p2] = s[p2 - 1 : p1 - 1 : -1]
                p1 = p2 + 1
            elif p2 == len(s) - 1:  #reversing last word when p2 is not ' '
                s[p1 : p2 + 1] = s[p2 : p1 - 1 : -1]

            p2 += 1
        return s

    s = ["t", "h", "e", " ", "s", "k", "y", " ", "i", "s", " ", "b", "l", "u", "e"]
    print(f(s)) #['b', 'l', 'u', 'e', ' ', 'i', 's', ' ', 's', 'k', 'y', ' ', 't', 'h', 'e']

110. (LC 17) Letter Combinations of a Phone Number
---------------------------------------------------
`LC 17. Letter Combinations of a Phone Number <https://leetcode.com/problems/letter-combinations-of-a-phone-number/>`_
Medium ::

    # Solution 1
    def letterCombinations0(digits: str) -> List[str]:
        if not digits:
            return []
        d = ["abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
        ans = [""]
        for i in digits:
            s = d[int(i) - 2]
            ans = [a + b for a in ans for b in s]
        return ans

    digits = "23"
    print(letterCombinations0(digits))
    # ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']

    # After 1st pass ans = ['a', 'b', 'c']
    # During 2nd pass ans = ['a'+'g', 'a'+'h', 'a'+'i', 'b'+'g', 'b'+'h' etc. ]

    # Solution 2 Cartesian product
    ### My V (LC accepted 90,40%)
    class Solution:
        def letterCombinations(self, digits: str) -> List[str]:
            if len(digits) == 0:
                return []
            d = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno',
                '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
            letters = [d[s] for s in digits]
            res = itertools.product(*letters)
            return [''.join(item) for item in res]  #*

#* Have to join because product() returns res = [['a', 'd'], ['a', 'e'] .. ] -->
so we join to have the required format ["ad","ae"..]

111. (LC 38) Count and Say (Look-and-say)
-------------------------------------------
`38. Count and Say (Look-and-say) <https://leetcode.com/problems/count-and-say/>`_
Medium

| # More elaboration on the task.
| The sequence starts with 1.
| "How many + Item" - 1 -> one 1 -> 11
| 11 -> two 1 -> 21 NOTE, we do not append to previous, we build from scratch each time.
| 21 -> one 2, one 1 -> 1211 NOTE, we count from MSB
| The first 6 items in such a sequence.
| [1, 11, 21, 1211, 111221, 312211]
 
| You are given integer n, return nth item in sequence.
| Note, return the result as a string.

::

    ### My V
    def f(n):
        if n == 1:
            return 1
        ans = ["1"]
        s = ""
        for _ in range(n - 1):
            cnt = 1
            c = ans[-1]
            for j in range(len(c)):
                if j == len(c) - 1 or c[j + 1] != c[j]:
                    s += str(cnt)
                    s += c[j]
                    cnt = 1
                else:
                    cnt += 1
            ans.append(s)
            s = ""
        return ans[-1]

    print(f(3))
    print(f(4))
    print(f(6))
    #21
    #1211
    #312211

    # Use Stdlib
    import itertools as it
    def look_and_say_pythonic(n):
        s = "1"
        for _ in range(n - 1):
            s = "".join(str(len(list(group))) + key for key, group in it.groupby(s))
        return s

    print(look_and_say_pythonic(4))  # 1211

| ``itertools.groupby(iterable, key=None)``
| How it works.
| import itertools as it
| L = "224555"
| # Output groups
| print([list(g) for k, g in it.groupby(L)])  
| [['2', '2'], ['4'], ['5', '5', '5']]
 
| # Omitting k (for k, g ..) it gets weird
| print([list(g) for g in it.groupby(L)])     
| [['2', <itertools._grouper object at 0x7fc58273b9a0>], ['4',..
 
| # Output (group, group item)
| print([(list(g), k) for k, g in it.groupby(L)])
| [(['2', '2'], '2'), (['4'], '4'), (['5', '5', '5'], '5')]
 
| - what is g
| The actual group. ['2', '2']
| - what is k
| Grouper function. When None it is the value of item (result of identity function). '2'

::

    # Solution 1 (book attr)
    def look_and_say1(n):
        def next_number(s):
            result, i = [], 0
            while i < len(s):
                count = 1
                while i + 1 < len(s) and s[i] == s[i + 1]:
                    i += 1
                    count += 1
                result.append(str(count) + s[i])
                i += 1
            return "".join(result)

        s = "1"
        for _ in range(1, n):
            s = next_number(s)
        return s

    print(look_and_say1(4))  # 1211

    # Solution 2 (doocs attribution)
    def countAndSay2(n: int) -> str:
        s = "1"
        for _ in range(n - 1):
            i = 0
            t = []
            while i < len(s):
                j = i
                while j < len(s) and s[j] == s[i]:
                    j += 1
                t.append(str(j - i))
                t.append(str(s[i]))
                i = j
            s = "".join(t)
        return s

    print(countAndSay2(4))  # 1211

| Time O(n2**n)
| 2**n - because each successive number can have twice as many digits as the previous.
| n - n iteration

112. (LC 13) Roman to Integer
------------------------------------
`LC 13. Roman to Integer <https://leetcode.com/problems/roman-to-integer/>`_
Easy

| **Solution 1** [:ref:`2 <ref-label>`]

- Invalid roman numbers

Note, we do not account for invalid romans (like IC and would return 99 for it,
while I can only come before V and X), because the task guarantees that we are given
only valid roman numbers.

- key points

We iterate from right to left. If current value is smaller than the value to the 
right, then we subtract (right-left), otherwise add. ::

    import functools
    def roman_to_integer(s):
        T = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        return functools.reduce(
            lambda val, i: val + (-T[s[i]] if T[s[i]] < T[s[i + 1]] else T[s[i]]),
            reversed(range(len(s) - 1)),
            T[s[-1]],
        )
    print(roman_to_integer("IX"))  # 9
    print(roman_to_integer("XI"))  # 11

| **Explained**
| - Reminder
| functools.reduce(function, iterable[, initializer])
 
| So 
|         T[s[-1]],
| as last arg to reduce() is the initializer, 
| E.g. 'IX', initializer is 'X'.
 
|         reversed(range(len(s) - 1)),
| We iterate from right to left, start at i=-2
| E.g. 'IX', we start with 'I'
 
| if T[s[i]] < T[s[i + 1]]
| if 'I' < 'X', yes it is, then 
| val + (-T[s[i]]), i.e.('X' - 'I') (because we iterate from right to left).

::

    ### Solution 1 but no magic
    (Iterate left->right, no jumps)
    class Solution:
        def romanToInt(self, s: str) -> int:
            roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
            res = 0
            for i in range(len(s)):
                if i + 1 < len(s) and roman[s[i]] < roman[s[i + 1]]:
                    res -= roman[s[i]]
                else:
                    res += roman[s[i]]
            return res

    ### Solution 1, still no magic, my V (LC accepted)
    (iterate right->left, 2 step jumps if met e.g. IV)
    def f(s):
        d = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        ans = 0
        i = len(s) - 1
        while i >= 0:
            if i > 0 and (d[s[i]] > d[s[i - 1]]):
                ans += d[s[i]] - d[s[i - 1]]
                i -= 2
            else:
                ans += d[s[i]]
                i -= 1
        return ans

    #### My brute force (LC accepted, with middle stats)
    def f(s):
        d = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        ans = 0
        i = 0

        def add_from_dict():
            nonlocal d, i, ans
            ans += d[s[i]]
            i += 1

        while i < len(s):
            if s[i] == "I":
                if i != len(s) - 1 and s[i + 1] == "V":
                    ans += 4
                    i += 2
                elif i != len(s) - 1 and s[i + 1] == "X":
                    ans += 9
                    i += 2
                else:
                    add_from_dict()
            elif s[i] == "X":
                if i != len(s) - 1 and s[i + 1] == "L":
                    ans += 40
                    i += 2
                elif i != len(s) - 1 and s[i + 1] == "C":
                    ans += 90
                    i += 2
                else:
                    add_from_dict()
            elif s[i] == "C":
                if i != len(s) - 1 and s[i + 1] == "D":
                    ans += 400
                    i += 2
                elif i != len(s) - 1 and s[i + 1] == "M":
                    ans += 900
                    i += 2
                else:
                    add_from_dict()
            else:
                add_from_dict()
        return ans

113. (LC 93) Restore IP Addresses
------------------------------------
`LC 93. Restore IP Addresses <https://leetcode.com/problems/restore-ip-addresses/>`_
Medium
(Or compute all valid IP addresses.)

The simpler version of a backtrack process is 22. Generate Parentheses 
(in stacks, here N 177.)

=> When you see: create all possible combinations: think Backtracking ::

    ### Solution 2 (neetcode, LC accepted 95, 18%)
    # The order of s stays the same, but we place the "." in different places
    class Solution:
        def restoreIpAddresses(self, s: str) -> List[str]: 
            res = []
            if len(s) > 12:   #no valid response possible
                return res
            def backtrack(i, dots, curIP):
                if dots == 4 and i == len(s):  #we're beyond limits
                    res.append(curIP[:-1])
                    return
                if dots > 4:
                    return
                for j in range(i, min(i + 3, len(s))):
                    if int(s[i:j+1]) <= 255 and (i == j or s[i] != "0"):
                        backtrack(j + 1, dots + 1, curIP + s[i:j+1] + ".")
            backtrack(0,0, "")
            return res

**Explained**

|     if len(s) > 12:   
| s='123454245653356'
| Our given initial string <s> is too long, not possible to construct a single valid IP
| from such a string.
 
|     def backtrack(i, dots, curIP):
| =>Give to backtrack func variables that will change during the backtracking.
| i - index we are at in the given string
| dots - we keep count of total num of dots inserted so far.
| curIP - current IP we are constructing.
 
|     if dots == 4 and i == len(s):
|         res.append(curIP[:-1])
| The "good" base case. When we are done with constructing the current IP and add it to result.
| When the i index in the current loop would be out of bounds, and we've recorded the 4th dot: '0.1.2.3.'
| We also slice off the last dot:
| 0.1.2.3.
 
|     if dots > 4:
|         return
| We got 4 dots but index didn't yet reach the end of the string.
| E.g. 1.2.3.4.5667
| So no valid IP from this combination of dots.
 
|     for j in range(i, min(i + 3, len(s))):
| The max of our looping is the size of 1 part, which is index+3 (or len of initial string).
 
|     if int(s[i:j+1]) <= 255 and (i == j or s[i] != "0"):
| If slice results in a num <=255 AND (that num is EITHER a single digit OR the first digit of several digits is not zero).
| i == j -> if len of char is just 1. We are slicing, so when i==j, len of slice is 1.
| s[i] != "0" -> that is the first char is not zero.

| **Solution 2** [:ref:`2 <ref-label>`]
| Tips:
| -Brute force with nested loops.
| -Separate func that checks part validity.

::

    ### Solution 1 (LC accepted 80, 90%)
    class Solution:
        def restoreIpAddresses(self, s: str) -> List[str]:
            def is_valid_part(s):
                # '00', '000', '01', etc. are not valid, but '0' is valid.
                return len(s) == 1 or (s[0] != "0" and int(s) <= 255)

            result, parts = [], [None] * 4
            for i in range(1, min(4, len(s))):
                parts[0] = s[:i]
                if is_valid_part(parts[0]):
                    for j in range(1, min(len(s) - i, 4)):
                        parts[1] = s[i : i + j]
                        if is_valid_part(parts[1]):
                            for k in range(1, min(len(s) - i - j, 4)):
                                parts[2], parts[3] = s[i + j : i + j + k], s[i + j + k :]
                                if is_valid_part(parts[2]) and is_valid_part(parts[3]):
                                    result.append(".".join(parts))
            return result

    s = "25525511135"
    s2 = "101023"
    print(get_valid_ip_address(s))  # ['255.255.11.135', '255.255.111.35']
    print(get_valid_ip_address(s2)) # ['1.0.10.23', '1.0.102.3', '10.1.0.23', '10.10.2.3', '101.0.2.3']

| Time complexity (don't quite get it).
| The total number of IP addresses is a constant (2**32), implying an O(1) time complexity 
| for the above algorithm.
 
| Explained:
| s[0] != '0'
| Meaning first digit of a part is not 0. E.g. 01.. with leading 0 is not valid.
| (While 0 alone is valid.)
 
| in range(1, min(4, len(s))
| till 4 (exclusive 4, i.e. really 3 digits) or len(s) whichever we reach first.
 
|         parts[0] = s[:i]
|         ..
|                 parts[1] = s[i : i + j]
| E.g. s='255..' i=1
| parts[0] = s[:1] (=[2])
| parts[1] = s[1:1+1] (=[5])

114. Write a string sinusoidally
-------------------------------------
**Task**
Given string s like "Hello_World". Its sinusoid (snakestring) is::

    # e   _   l
    #H l o W r d
    #   l   o   !

Written in 1 dimension it is: "e_lHloWrdlo!".

Given string, output its sinusoid (snakestring).

| **Logic:**
| Look for pattern. 
| I.e. write down concrete examples.
| Any sinusoid has 3 rows.
| The above concrete example shows this pattern:
| Row 1: s[1],s[5],s[9],... ->step=4 starting with 1
| Row 2: s[0],s[2],s[4],..  ->step=2 starting with 0
| Row 3: s[3],s[7],s[11],.. ->step=4 starting with 3

::

    ### Solution V1 (full book attr)
    # Three iterations through s
    def snake_string(s):
        result = []
        # outputs the first row, step=4, starting with 1
        for i in range(1, len(s), 4):
            result.append(s[i])
        for i in range(0, len(s), 2):
            result.append(s[i])
        for i in range(3, len(s), 4):
            result.append(s[i])
        return "".join(result)

    s = "Hello_World"
    print(snake_string(s)) # e_lHloWrdlo

    ### Solution V2 (full book attr)
    # Using Python slicing.

    def snake_string_pythonic(s):
        return s[1::4] + s[::2] + s[3::4]

Time:
3 iterations, O(n) each, so overall O(n). n is len of string.

::

    ### My V
    def sin_string(s):
        s1, s2, s3 = "", "", ""
        for i in range(0, len(s), 2):
            s2 += s[i]
        for j in range(1, len(s), 4):
            s1 += s[j]
        for k in range(3, len(s), 4):
            s3 += s[k]
        return s1 + s2 + s3

    s = "Hello_World"
    print(sin_string(s)) #e_lHloWrdlo

115. Implement run-length encoding
-------------------------------------
| **Task**
| Run-length encoding (RLE) compression.
| Encoding. 
| Encodes string "aaaabcccaa" into "4a1b3c2a". 
| Decoding of "3e4f2e" refurns "eeeffffee".
| Implement both encoding and decoding functions.

::

    ### Solution (O(n))
    def decoding(s):
        cnt, res = 0, []
        for c in s:
            if c.isdigit():
                cnt = int(c)
            else:
                res.append(c * cnt)
                cnt = 0
        return "".join(res)

[:ref:`2 <ref-label>`] ::

    def encoding(s):
        result, count = [], 1
        for i in range(1, len(s) + 1):
            if i == len(s) or s[i] != s[i - 1]:  #if 1st part of or is True, he is not going to eval the 2nd, so no out of bounds error
                # Found new character so write the count of previous character
                result.append(str(count) + s[i - 1])
                count = 1
            else:  # s[i] == s[i - 1].
                count += 1
        return "".join(result)

    s = "ffyyu"
    print(encoding(s)) # 2f2y1u

    ### My V
    def encode2(s):
        index = 0
        s += str(0)  # to make 'aaabb..c0' extra char to compare with the previous char
        ans = ""
        for i in range(1, len(s)):
            if s[i] != s[index]:
                cnt = i - index
                ans += str(cnt)
                ans += s[index]
                index = i
        return ans

    s = "aaaabcccaa"
    print(encode2(s))  # 4a1b3c2a

    from string import ascii_lowercase
    def decode1(s):
        ans = ""
        for i in range(len(s) - 1):
            if s[i] not in ascii_lowercase:
                ans += s[i + 1] * int(s[i])
        return ans

    s = "2a5f1t"
    print(decode1(s))  # aaffffft

















