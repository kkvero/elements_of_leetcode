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

131. (LC 496) Next Greater Element I
-----------------------------------------
`496. Next Greater Element I <https://leetcode.com/problems/next-greater-element-i/>`_
Easy

| Note, in output, you return values in array2, not indexes.
| Attribution [:ref:`10 <ref-label>`].

| **Two approaches.**
| **O(n*m)**
| n and m our two arrays a1, a2.
| ~O(N**2), space O(m)
| O(n*m) because we do have a nested loop iterating a2.

| **Key things**
| 1)Hash map a1.
| 2)Iterating over a2, we use hash of a1 in 2 ways:
| - To see if n in a2 is at all in a1. 
| E,g, a1=[4,1,2], a2=[1,3,4,2], 3 is not in a1, so we can skeep it.
| - To know where to put n on a2 in res.
| E.g. 1 in a2, hash of a1 says that 1 is at index 1 in a1.
| So having 3 from a2, we know to put it at index 1 in res.
| res=[_,3,_]
| 3)Initialize res=[-1]*len(a1)

::

    #Solution
    class Solution:
        def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
            # O (n * m)
            nums1Idx = { n:i for i, n in enumerate(nums1) }
            res = [-1] * len(nums1)
            
            for i in range(len(nums2)):
                if nums2[i] not in nums1Idx:
                    continue
                for j in range(i + 1, len(nums2)):  #check values after a2[i+1]
                    if nums2[j] > nums2[i]:
                        idx = nums1Idx[nums2[i]] #get index of that N in a1
                        res[idx] = nums2[j]
                        break
            return res

    ### My V (LC accepted 28, 85)
    class Solution:
        def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
            d = {}
            for i in range(len(nums2)):
                d[nums2[i]] = -1
                for j in range(i, len(nums2)):
                    if nums2[j] > nums2[i]:
                        d[nums2[i]] = nums2[j]
                        break
            ans = []
            for n in nums1:
                ans.append(d[n])
            return ans

| **O(n+m)**
| **Keys:**
| -Monotonic stack
| -ans, initiate as [-1, -1, -1..]
| -hash of nums1, {value: index}
| -iterate nums2, if value < stack[-1], add to stack, if value is greater, then 
| that is the next greater value for all values in stack.

| Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
| Output: [-1,3,-1]

::

    class Solution:
        def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
            nums1Idx = { n:i for i, n in enumerate(nums1) }  # {4:0, 1:1, 2:2}
            res = [-1] * len(nums1)
            stack = []
            for i in range(len(nums2)):
                cur = nums2[i]

                # Mono decreasing stack. If we found in nums2 N that is greater than stack[-1] 
                # than it is the next greater value for ALL values in stack
                # (and in stack we have all values that are in nums1)
                while stack and cur > stack[-1]:
                    val = stack.pop() # take top val
                    idx = nums1Idx[val]
                    res[idx] = cur

                if cur in nums1Idx:  #because we can have values in nums2 that are not in nums1
                    stack.append(cur)
            return res

| **Explained**
| Example.
| a1=[4,1,2], a2=[2,1,3,4]
| In general:
| Notice that when we find answer 3, it is the answer for all nums before that, i.e. for 1,2.
| 3>1, 3>2.
 
| Walkthrough:
| 1)stack=[]
| a2=[2..]
| If 2 in a1, we add 2 to stack.
| stack=[2]
| 2)a2=[2,1..]
| Next up is 1.
| Is 1 greater than any num in stack. No. Add 1 to stack. Move on.
| 3)
| stack=[2,1]  <--Note, stack is always in decreasing order
| a2=[2,1,3..]
| Looking at 3.
| 3 > stack[-1]
| So 3 is the answer for all nums in stack.
| Start popping from stack.
| stack.pop()=1
| We find index of 1 in a1, put value 3 at that index
| stack.pop()=2 ..
| So: res = [_,3,3]
| ->As last thing we check if 3 is in a1, only if it is, put 3 to the stack, otherwise no.

132. (LC 344) Reverse String
---------------------------------
`344. Reverse String <https://leetcode.com/problems/reverse-string/>`_
Easy

| Note:
| Input: s = ["h","e","l","l","o"]
| Input format is important, would need a different solution for string, not array input.
 
| **Solution 1** (most efficient)
| Key points:
| -change in place
| -pointers
| -do not return anything

::

    def reverse_string(s):
        l = 0
        r = len(s) - 1
        while l < r:
            s[l],s[r] = s[r],s[l]
            l += 1
            r -= 1

    ### My V (LC accepted 98, 70)
    class Solution:
        def reverseString(self, s: List[str]) -> None:
            """
            Do not return anything, modify s in-place instead.
            """
            if len(s) <=1:
                return
            rp = len(s)-1
            for lp in range(len(s)//2):
                if s[lp] != s[rp]:
                    s[lp], s[rp] = s[rp], s[lp]
                rp -= 1


| **Solution 2**
| Somewhat less efficient. But good to know if you will need to discuss alternatives.
| Uses Stack. O(N). Extra space.
| -We put all chars to stack.
| -Pop from stack, each time replacing chars in array.
| -Again, do not return, we're modifying in place.

::

    def f(a):
        stack = []
        for c in a:
            stack.append(c)
        i = 0
        while stack:
            a[i] = stack.pop()
            i += 1

    s = ["h", "e", "l", "l", "o"]
    f(s)
    print(s) # ['o', 'l', 'l', 'e', 'h']

| **Solution 3. Recursion.**
| Even less efficient. Time O(N), space O(N).

::

    class Solution:
        def reverseString(self, s: List[str]) -> None:
            def rev_str(l, r):
                if l < r:
                    s[l], s[r] = s[r], s[l]
                    rev_str(l + 1, r - 1)
            rev_str(0, len(s) - 1)

    sol = Solution()
    sol.reverseString(s)
    print(s) # ['o', 'l', 'l', 'e', 'h']

    # Recursion my V
    def f(a, l=0, r=len(a) - 1):
        if l >= r:
            return
        a[l], a[r] = a[r], a[l]
        f(a, l + 1, r - 1)

    f(s)
    print(s) # ['o', 'l', 'l', 'e', 'h']



