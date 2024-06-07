Strings Questions Part 3
=========================
126. (LC 67) Add Binary
-------------------------
`67. Add Binary <https://leetcode.com/problems/add-binary/>`_
Easy ::

    # My V
    def f(a, b):
        res = int(a, 2) + int(b, 2)
        # return bin(res)
        return bin(res)[2:]

    # Solution
    class Solution:
        def addBinary(self, a: str, b: str) -> str:
            return bin(int(a, 2) + int(b, 2))[2:]

>>> int('1010', 2)
10
>>> int('1011', 2)
11
>>> bin(21)
'0b10101'

::

    # V2
    def f(a, b):
        a = int(a, 2)
        b = int(b, 2)
        return bin(a + b)[2:]

    a = "11"
    b = "1"
    print(f(a, b))  # 100

    a = "1010"
    b = "1011"
    print(f(a, b))  # 10101

127. (LC 415) Add Strings
-----------------------------
`415. Add Strings <https://leetcode.com/problems/add-strings/>`_
Easy

| Keys:
| -String slicing

::

    ### Solution 1 (LC 85, 30)
    class Solution(object):
        def addStrings(self, num1, num2):
            result = []
            carry = 0
            while num1 or num2 or carry:  # takes care of nums being of different len
                digit = carry
                if num1:
                    digit += int(num1[-1])
                    num1 = num1[:-1]
                if num2:
                    digit += int(num2[-1])
                    num2 = num2[:-1]
                carry = digit > 9  # True also means 1, and its not like carry can be >1, but alt. carry=digit//10
                result.append(str(digit % 10))
            return "".join(result[::-1])

    ### My V (LC accepted 6-10, 80)
    class Solution:
        def addStrings(self, num1: str, num2: str) -> str:
            ans = ''
            carry = 0
            while num1 or num2 or carry:
                n1 = int(num1[-1]) if num1 else 0
                n2 = int(num2[-1]) if num2 else 0
                res = n1 + n2 + carry
                carry = res//10
                ans += str(res % 10)
                num1 = num1[:-1]
                num2 = num2[:-1]
            return ans[::-1]

    ### My attempt (LC accepted 30,60%) (a lot of off by 1 gotchas)
    def add_strings(s1, s2):
        if len(s2) > len(s1):
            s1, s2 = s2, s1
        ans = [0] * (len(s1) + 1)
        for i in range(1, len(s1) + 1):
            res = ans[i - 1] + int(s1[-i])
            if i <= len(s2):
                res += int(s2[-i])
            carry = res // 10
            ans[i] = carry
            ans[i - 1] = res % 10
        if ans[-1] == 0:
            ans = ans[:-1]
        ans = ans[::-1]
        ans = [str(x) for x in ans]
        return "".join(ans)

128. (LC 58) Length of Last Word
------------------------------------
`58. Length of Last Word <https://leetcode.com/problems/length-of-last-word/>`_
Easy ::

    ### Solution 1 (neetcode)
    def lengthOfLastWord(s: str) -> int:
        c = 0
        for i in s[::-1]:
            if i == " ":
                if c >= 1:
                    return c
            else:
                c += 1
        return c

    ### Solution 1 v2
    def f1(s):
        cnt = 0
        for i in range(len(s) - 1, -1, -1):  #OR for i in s[::-1]:
            if s[i] == " ":
                if cnt >= 1:
                    return cnt
            else:
                cnt += 1
        return cnt

    ### My V
    class Solution:
        def lengthOfLastWord(self, s: str) -> int:
            c = 0
            for char in s[::-1]:
                if char != " ":
                    c += 1
                else:  # char is a space
                    if c > 0:
                        return c
            return c

    # A shortcut (LC 55,85%)
    def f2(s):
        return len(s.split()[-1])

    # LC accepted (33,84%)
    class Solution:
        def lengthOfLastWord(self, s: str) -> int:
            a = s.split()
            return len(a[-1])

129. (LC 14) Longest Common Prefix
-------------------------------------
`14. Longest Common Prefix <https://leetcode.com/problems/longest-common-prefix/>`_
Easy ::

    ### My V2 (LC accepted 17, 45%)
    class Solution:
        def longestCommonPrefix(self, strs: List[str]) -> str:
            pref = strs[0]
            for w in strs:
                if w == '':
                    return w
                pointer = 0
                while pointer < len(w) and pointer < len(pref):
                    if pref[pointer] != w[pointer]:
                        pref = pref[:pointer]
                        if pref == '':
                            return pref
                        break
                    pointer += 1
                pref = pref[:pointer]

            return pref

    ### My V1 (LC accepted 13, 99%)
    def f(a):
        common = a[0]
        for s in a[1:]:
            for i in range(len(common) + 1):
                if common[:i] != s[:i]:
                    break
                else:
                    cur_common = common[:i]
            common = cur_common
        return common

130. (LC 43) Multiply Strings
----------------------------------
`43. Multiply Strings <https://leetcode.com/problems/multiply-strings/>`_
Medium

| **Key things**
| Edge case.
| Initiate the response as list, [0]*
| Reverse input from the start.
| Multiply and add in one go.
| Dealing with shift to the left in multiplication with res[i1+i2].
| Filter out extra 0s at the front (check if value==0, from the front)

::

    class Solution:
        def multiply(self, num1: str, num2: str) -> str:
            if "0" in [num1, num2]:
                return "0"

            res = [0] * (len(num1) + len(num2))
            num1, num2 = num1[::-1], num2[::-1]
            for i1 in range(len(num1)):
                for i2 in range(len(num2)):
                    digit = int(num1[i1]) * int(num2[i2])
                    res[i1 + i2] += digit
                    res[i1 + i2 + 1] += res[i1 + i2] // 10
                    res[i1 + i2] = res[i1 + i2] % 10
            
            # Filter out prepended 0s
            res, beg = res[::-1], 0
            while beg < len(res) and res[beg] == 0:
                beg += 1
            res = map(str, res[beg:])
            return "".join(res)

    ### My V
    def f(s1, s2):
        if "0" in [s1, s2]:
            return "0"
        ans = [0] * (len(s1) + len(s2))
        for i in range(len(s2) - 1, -1, -1):
            for j in range(len(s1) - 1, -1, -1):
                index = i + j + 1              #place into ans dynamically
                res = int(s2[i]) * int(s1[j])
                ans[index] += res
                carry = ans[index] // 10
                ans[index - 1] += carry
                ans[index] %= 10
        if ans[0] == 0:    #del prepped 0 
            ans = ans[1:]
        ans = [str(x) for x in ans]
        return "".join(ans)

    print(f("14", "15"))  # '210'
    print(f("556", "30"))  # '16680'
    print(f("556", "0"))  # '0'







