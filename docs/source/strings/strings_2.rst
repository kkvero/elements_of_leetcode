Strings Questions Part 2
=========================
116. (LC 443) String Compression
--------------------------------------
`443. String Compression <https://leetcode.com/problems/string-compression/>`_
Medium

| **Key:**
| -The task describes exactly what alg they want you to use:
| "Begin with an empty string s. For each group of consecutive repeating characters in chars."
| So don't use something entirely different.

-Using 2 pointers that build a string s of consecutive characters, e.g. 'aaaa'.

| **Hooks:**
| #1 Stop when it is the last index in chars.
| ``while p2 < (len(chars)) and (not s or s[0] == chars[p2]):``

::

    ### My V2 (LC accepted 25, 98)
    class Solution:
        def compress(self, chars: List[str]) -> int:
            p1 = p2 = 0
            while p2 < len(chars):
                s = ""
                while p2 < (len(chars)) and (not s or s[0] == chars[p2]): #1
                    s += chars[p2]
                    p2 += 1
                chars[p1] = s[0]
                p1 += 1
                if len(s) == 1:
                    continue
                else:
                    for n in str(len(s)):  #working with string repr of num
                        chars[p1] = n
                        p1 += 1
            return p1

    ### My V1 (LC accepted 97, 30)
    class Solution:
        def compress(self, chars: List[str]) -> int:
            p1 = p2 = 0
            while p2 < len(chars):
                s = ""
                while p2 < (len(chars)) and (not s or s[0] == chars[p2]):
                    s += chars[p2]
                    p2 += 1
                chars[p1] = s[0]
                p1 += 1
                if len(s) == 1:
                    continue
                else:
                    length = len(s)            #**
                    zeros = len(str(length))
                    num = 10 ** (zeros-1)

                    while num >= 1:           
                        n = length // num
                        length %= num
                        num /= 10
                        chars[p1] = str(int(n))
                        p1 += 1

            return p1

| #**
| This decomposes num working with integers.
| Simpler to work with str repr of int.
|     for c in cnt:
|         chars[k] = c
|         k += 1

But makes a difference for runtime.


117. (LC 28) Find the Index of the First Occurrence in a String
---------------------------------------------------------------------
`28. Find the Index of the First Occurrence in a String <https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/>`_
Easy

AKA Find the first occurrence of a substring, OR, Find needle in a haystack

| **Keys to easy solution**
| -Slicing
| (Slice haystack the size of string and compare to needle.)

| **How easy can become hard**
| There is an easy solution.
| The brute force solution, O(n*m).

(But in best case O(n). E.g. hay='abcde', needle='xyz'. We check 'abc' and 'xyz',
but we check first chars, see they don't match and don't compare the rest. Move on to 
'bcd'.)

| We need a better solution for the case like:
| haystack = "mississippi"
| needle = "issip"
| Where in several cases we would need to compare to every/many chars of the string.

| So testing string equality is expensive.
| There are three linear time string matching algorithms: KMP, Boyer-Moore, and Rabin-Karp. 

::

    ### Solution 1 Brute force (doocs attribution [7])
    class Solution:
        def strStr(self, haystack: str, needle: str) -> int:
            n, m = len(haystack), len(needle)
            for i in range(n - m + 1):
                if haystack[i : i + m] == needle:
                    return i
            return -1

    ### My V (LC accepted 5, 68%)
    class Solution:
        def strStr(self, haystack: str, needle: str) -> int:
            for i in range(len(haystack) - len(needle) + 1):
                if haystack[i: i + len(needle)] == needle:
                    return i
            return -1

**Rabin-Karp algorithm** [:ref:`13 <ref-label>`]

O(m+n), m, n are string lengths.
Of the three algs mentioned above, Rabin-Karp is by far the simplest to understand and implement.
The algorithm uses a hash function. We hash strings as numbers, compare those,
only if we think we found a match, to make sure we don't stumble upon a collision,
we check string representations.

| - What exactly
| t = 'GACGCCA' (text/haystack)
| s = 'CGC'  (string/needle)

If containing letters = [A,C,G,T] are hashed to their indices, then A=0, C=1, G=2, T=3.
Then the hash code for s='CGC'=121.
How we move through the text string:
'GAC' is 201, not our s=121, then
'ACG', we compute its code from the previous string of 'GAC'. I.e. ((201-200)*10)+2=12
etc.

::

    import functools
    def rabin_karp(t, s):
        if len(s) > len(t):
            return -1  # s is not a substring of t
        BASE = 26
        # Hash codes for the substring of t and s.
        t_hash = functools.reduce(lambda h, c: h * BASE + ord(c), t[: len(s)], 0)
        s_hash = functools.reduce(lambda h, c: h * BASE + ord(c), s, 0)
        power_s = BASE ** max(len(s) - 1, 0)  # BASE^|s-1|.

        for i in range(len(s), len(t)):
            # Checks the two substrings are actually equal or not
            # against hash col-lision.
            if t_hash == s_hash and t[i - len(s) : i] == s:
                return i - len(s)  # Found a match.

            # Uses rolling hash to compute the hash code
            t_hash -= ord(t[i - len(s)]) * power_s
            t_hash = t_hash * BASE + ord(t[i])
            # Tries to match s and t[-len(s):].
            if t_hash == s_hash and t[-len(s) :] == s:
                return len(t) - len(s)
        return -1  # s is not a substring of t.

    haystack = "sadbutsad"
    needle = "sad"
    print(rabin_karp(haystack, needle)) # 0

| **KMP (Knuth-Morris-Pratt algorithm)** [:ref:`10 <ref-label>`] 
| O(N+M)
| See the neetcode for that.

118. (LC 424) Longest Repeating Character Replacement
---------------------------------------------------------
`LC 424. Longest Repeating Character Replacement <https://leetcode.com/problems/longest-repeating-character-replacement/>`_
Medium
Topics: String, Sliding window

| **Keys**
| In addition to sliding window:
| - hash table
| - while (r - l + 1) - max(count.values()) > k:
| (window total size - the letter which predominates in cur window, its number of occurrences)

::

    # O(26n)
    def characterReplacement(s: str, k: int) -> int:
        count = {}  #count letters in window{'A': 2}
        res = 0
        l = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            while (r - l + 1) - max(count.values()) > k:
                count[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)

        return res

| **Explained**
|         count[s[r]] = 1 + count.get(s[r], 0)
| We make our first entry into the hash table count={'A': 1}
| I.e. it is:

>>> D = {}
>>> D['A'] = 1+D.get('A', 0)  
>>> D
{'A': 1}

# 0 is the default value

|         while (r - l + 1) - max(count.values()) > k:
| This checks if our window is a valid window.
| (r - l + 1) - this is the OVERALL current window, i.e. (right pointer - left + 1) 
| (+1 is because indexing starts at 0, ind1 - ind0 is actually len 2, not 1, so 1+1)
 
| max(count.values()) - is the max number of the SAME LETTERS in our window, 
| e.g. in window 'ABA' max number of the same letters is A=2.
| So it is the number of letters we will not have to change to make it a valid window.

.. note::

    (Current window - letters we won't have to touch) = letters we will have to change

And that number, "letters we will have to change" should not be greater than k, 
(k is num of letters that we are allowed to change.)

|             count[s[l]] -= 1
|             l += 1

When in current window the number of letters we will have to change exceeds k,
we have to decrease the size of the current window, i.e. move i +1.
And also of course remove that letter in our count hash table.
(e.g. window was 'ABA' count={A:2, B:1}, l=+1, window becomes 'BA' count={A:1, B:1})

|        res = max(res, r - l + 1)
| Record max value of a valid window.

::

    ### Solution with optimization (a tiny one) making O(n)
    def characterReplacement(s: str, k: int) -> int:
        count = {}
        l = 0
        maxf = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            maxf = max(maxf, count[s[r]])

            if (r - l + 1) - maxf > k:
                count[s[l]] -= 1
                l += 1

        return r - l + 1

119. (LC 76) Minimum Window Substring
------------------------------------------
`76. Minimum Window Substring <https://leetcode.com/problems/minimum-window-substring/>`_
Hard
::

    # O(n)
    def minWindow(s: str, t: str) -> str:
        if t == "":
            return ""

        countT, window = {}, {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, need = 0, len(countT)
        res, resLen = [-1, -1], float("infinity")
        l = 0
        for r in range(len(s)):
            c = s[r]
            window[c] = 1 + window.get(c, 0)

            if c in countT and window[c] == countT[c]:
                have += 1

            while have == need:
                # update our result
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = r - l + 1
                # pop from the left of our window
                window[s[l]] -= 1         #we decrease the counter for that letter
                if s[l] in countT and window[s[l]] < countT[s[l]]:
                    have -= 1
                l += 1
        l, r = res
        return s[l : r + 1] if resLen != float("infinity") else ""

    s = "ADOBECODEBANC"
    t = "ABC"
    print(minWindow(s, t))  # BANC


|     have, need = 0, len(countT)
| have and need are single integer counts.
| E.g. in the given example we need overall 3 letters. 
| {'A': 1, 'B': 1, 'C': 1} (need=3)
| if it were
| {'A': 1, 'B': 1, 'C': 2} (Then need=4)
 
| For the have we count in a similar way, but for the current window.
| We start with have=0.
| Note that when window={'A': 1, 'B': 1, 'C': 2, 'D':4}, 
| -> have=3 still (if what we need is {'A': 1, 'B': 1, 'C': 1})
 
| Then when we achieve the state need==have, we can start moving the left pointer 
| (to get a shorter window), when popping letters that are not part of "t" (like R, G etc.),
| "have" remains the same. Also popping characters that we have more than necessary (e.g. A:2),
| we also don't decrease the "have".

The point of "need" and "have" is to avoid checking the actual hash tables (of the t and window),
and only do the have == need comparisons.

|     res, resLen = [-1, -1], float("infinity")
| res=[-1,-1] because it just initializes for res=[lpointer, rpointer] 

::

        while have == need:
            # update our result
            # pop from the left of our window
            if s[l] in countT and window[s[l]] < countT[s[l]]:
                have -= 1
            l += 1

Only while counts for have and need are the same, we try to get a shorter window,
i.e. we will pop from the left by moving the left pointer.
We will not touch the "have" count if the letter we pop is a letter we don't care
about, i.e. letter not in t.
Only when the letter that we pop is in t (i.e. countT hash table),
and this situation: window[s[l]] < countT[s[l]], 
E.g. t="B" and our window has {'B':2}, then we do window['B']-=1, but we do not touch the "have".

120. (LC 242) Valid Anagram
-----------------------------
`242. Valid Anagram <https://leetcode.com/problems/valid-anagram/>`_
Easy
::

    ### My V (LC accepted 60, 30)
    class Solution:
        def isAnagram(self, s: str, t: str) -> bool:
            scnt = collections.Counter(s)
            tcnt = collections.Counter(t)
            return scnt == tcnt

    ### Solution V1 (neetcode attr)
    class Solution:
        def isAnagram(self, s: str, t: str) -> bool:
            if len(s) != len(t):
                return False

            countS, countT = {}, {}

            for i in range(len(s)):
                countS[s[i]] = 1 + countS.get(s[i], 0)
                countT[t[i]] = 1 + countT.get(t[i], 0)
            return countS == countT

    ### Solution V2 (ddocs attr)
    class Solution:
        def isAnagram(self, s: str, t: str) -> bool:
            if len(s) != len(t):
                return False
            cnt = Counter(s)
            for c in t:
                cnt[c] -= 1
                if cnt[c] < 0:
                    return False
            return True

121. (LC 49) Group Anagrams
-----------------------------------
`49. Group Anagrams <https://leetcode.com/problems/group-anagrams/>`_
Medium

| Hints:
| 1) Sorting
| 2) Tuples are a hashable type.
| We can't use another dictionary as a key in another dict.
| But we can use tuple as a key.
| Also recall that `ord(char)-ord('a')` gives a number 0-25.

::

    ### Solution doocs attr
    class Solution:
        def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
            d = defaultdict(list)
            for s in strs:
                k = ''.join(sorted(s))
                d[k].append(s)
            return list(d.values())

    ### Solution doocs attr
    class Solution:
        def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
            d = defaultdict(list)
            for s in strs:
                cnt = [0] * 26
                for c in s:
                    cnt[ord(c) - ord('a')] += 1
                d[tuple(cnt)].append(s)
            return list(d.values())

    ### My V (LC accepted 90, 80)
    class Solution:
        def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
            d = collections.defaultdict(list)
            for w in strs:
                d[''.join(sorted(w))].append(w)
            ans = [v for v in d.values()]
            return ans

122. (LC 20) Valid Parentheses
------------------------------------
`20. Valid Parentheses <https://leetcode.com/problems/valid-parentheses/>`_
Easy TOPICS: STACK

| Steps:
| -use stack of opened characters, use hash for closed chars
| -is char opened parentheses/braces
| if yes, append to stack
| if char closed parenthese/braces:
| if no stack or stack[-1] and cur char are not the same, return false

::

    ### Solution 0 (EPI, LC accepted 45,55%)
    def is_well_formed(s):
        stack_opened = []
        lookup = {"(": ")", "[": "]", "{": "}"}
        for c in s:
            if c in lookup:  # it is opened char
                stack_opened.append(c)
            elif not stack_opened or lookup[stack_opened.pop()] != c:
                return False
        return not stack_opened

    ### Solution 1 (doocs attr)
    class Solution:
        def isValid(self, s: str) -> bool:
            stk = []
            d = {'()', '[]', '{}'}
            for c in s:
                if c in '({[':
                    stk.append(c)
                elif not stk or stk.pop() + c not in d:
                    return False
            return not stk

    ### Solution 2 (neetcode attr)
    class Solution:
        def isValid(self, s: str) -> bool:
            Map = {")": "(", "]": "[", "}": "{"}
            stack = []
            for c in s:
                if c not in Map:
                    stack.append(c)
                    continue
                if not stack or stack[-1] != Map[c]:
                    return False
                stack.pop()
            return not stack

    ### My V2
    def f(s):
        stack = []
        d = {")": "(", "]": "[", "}": "{"}
        for c in s:
            if c == ")" or c == "]" or c == "}":
                if len(s) == 0:
                    return False
                match = stack.pop()
                if d[c] != match:
                    return False
            else:
                stack.append(c)
        return len(stack) == 0

My V1
(This is less memory efficient, because we store in 'a' where we collect all opened parentheses,
twice as many parentheses. Like a=['()', '{}'..]. When using a reference set or mapping,
we store just that one map and a=['(', '{'...]) ::

    def valid_pair(s):
        a = []
        for c in s:
            if c == "(":
                a.append("()")
            elif c == "[":
                a.append("[]")
            elif c == "{":
                a.append("{}")
            else:
                if len(a) == 0:
                    return False
                p = a.pop()
                if c != p[1]:
                    return False
        return len(a) == 0

123. (LC 5) Longest Palindromic Substring
----------------------------------------------
`5. Longest Palindromic Substring <https://leetcode.com/problems/longest-palindromic-substring/>`_
Medium

| FYI, time complexity of the brute force approach would be:
| n * n**2 = n**3 
| n (linear scan to check if a string is a palindrome) * n**2 (this many substrings)
| 
| **Keys:**
| -build palindrome from center
| -odd/even length strings
| 
| **Key aspects:**
| - We will check for palindromicity from the middle, outwards.

So the usual way is having 'bab', we start on the left, right, move inwards.
Here we will start form 'a' in 'bab'.
So looping through string s, we consider each character the center of a palindrome.
Thus we achieve n*n=O(n**2).

| - Edge case, handle the case when len(s)=even. 
| Relevant because we consider each character the center of a palindrome.
| 'babad', len=5
| 'baba', len=4

-This problem has as hint Dynamic programming. Bacause there is the Manachester
alg that uses Dynamic programming. I.e. we keep track of the largest palindrom 
radius we found so far. So we won't check chars within that radius.
We use an additional array where we record the radius of a found palindrome.
We are not using it here. 

**Solution 1** ::

    ### Solution 1 (attr neetcode) (LC accepted 50, 85%)
    class Solution:
        def longestPalindrome(self, s: str) -> str:
            ans=''
            for i in range(len(s)):
                for L, R in ((i,i), (i, i+1)):  # odd and even lengths
                    while L >= 0 and R < len(s) and s[L] == s[R]:
                        if (R-L+1) > len(ans):
                            ans = s[L:R+1]
                        L-=1
                        R+=1
            return ans

| **Explained**
|         ans = ""
|         # ansLen=0
| We could separately initiate resLen=0 to keep track of the max len, instead we just 
| refer to len(res) for that info.
 
|         for i in range(len(s)):
| Remember, i is the center of a potential palindrome.
 
|             for l, r in ((i,i), (i,i+1)): # odd and even lengths
| This is an optimization. It could have been written as two separate loops:
| # odd length substrings
| l, r = i, i  ---> substring has one character as center
| while l >= 0...
 
| # even length substrings
| l, r = i, i+1  ---> substring has two chars as center
| while l >= 0...
 
| So we repeat the while loop twice in case the len of string is odd, and in case len=even.
 
| l=i, r=i+1
| Means for a string like 'cbbd' we initialize e.g. l='c', r='b'; l='b', r='b', i.e. two centers.
 
|                 while l >= 0 and r < len(s) and s[l] == s[r]:
| While left and right pointers are inbound (within the string) AND while the substring 
| made with pointers is a palindrome.
 
| if (r - l + 1)
| Len of our substring palindrome.
 
**Solution 2**
Optimization to avoid double pass for odd and even len substrings (with one center
and two centers: 'aba', 'baab'). That is inserting a guard character like 'b!a!a!b'
making all odd substring. 
Actually performs a bit worse on Leetcode: 47,85% against non such optimization 50,85%)
(LC accepted)
And yes the solution becomes giagantic, and yes we will have trouble with substrings lengths
which we have to solve with mor code. ::

    class Solution:
        def longestPalindrome(self, s: str) -> str:
            ans=''
            s = self.pad(s)
            for i in range(len(s)):  #i is center
                L=R=i
                while L >=0 and R<len(s) and s[L] == s[R]:
                    L-=1
                    R+=1
                # Undo last move that broke palindromic rule
                L+=1
                R-=1
                palindrome = s[L:R+1]
                if self.unpadded_len(palindrome) > self.unpadded_len(ans):
                    ans = palindrome
            ans = self.unpad(ans)
            return ans

        def pad(self, s):
            assert '!' not in s
            s = '!'.join(s)
            return s
        
        def unpad(self, s):
            s = s.replace('!', '')
            return s

        def unpadded_len(self, pal):
            if not pal:
                return 0
            if pal[0] == '!':
                return len(pal)//2
            return (len(pal)+1)//2


**ALL POSSIBLE SOLUTIONS** [:ref:`13 <ref-label>`]

**Solution 1: brute force, O(n**3)**

- Checking if a string is a palindrome

# Leveraging Python negative indexing::

    def is_palindrome(s):
        def mirror(i):
            return ­(i + 1)
        for i in range(len(s) / 2):
            if s[mirror(i)] != s[i]:
                return False
        return True

# A faster way::

    check if it is equal to its reverse.
    def is_palindrome(s):
        return s == s[::­1]

# Complete solution::

    def longest_palindrome(s):
        best = ''
        for left in range(len(s)):
            for right in range(left, len(s)):
                substring = s[left:right+1]
                if is_palindrome(substring) and len(substring) > len(best):
                    best = substring
        return best

Time: Iterating over substrings is O(n**2 ) + is_palindrome helper has O(n) = O(n**3 ).

**Solution 2: dynamic programming, O(n**2 )**

| Avoid redundant comparisons, reuse previously computed results.
| - Compare starting not from the edges, but from the center.
| 'cdeedc'
| Then we also have completed the check for all its smaller substrings that have the same center.

We take each char to be the center. Expand outwards till string edges or found chars that make
current substring not a palindrome::

    # Illustration
    # ycabxbaddd
    #     ^

start e.g. at x. Checking bxb, abxba, stop at cabxbad.

# Complete alg (for odd-length palindromic substring)::

    def longest_palindrome(s):
        best = ''
        for center in range(len(s)):
            # Expand symetrically as long as the palindrome property holds
            left = center
            right = center
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            # The last move broke the palindrome property, undo it
            left += 1
            right = 1
            palindrome = s[left:right+1]
            # Record the palindrome if longest
            if len(palindrome) > len(best):
                best = palindrome
        return best

Yes, it calculates only for odd-length substrs. To add calculating for even-length,
we have options:

- change current func (quite a bit)
- add another func
- add a guard character to transform the palindrome so that it has odd length.

And minimally change the original function. ::

    def palindrome_pad(s):
        # 'aa' > 'a!a', 'aba' > 'a!b!a'
        assert '!' not in s
        return '!'.join(s)

The reverse transformation::

    def palindrome_unpad(padded):
        return padded.replace('!', '')

    def unpadded_length(padded):    #**
        if not padded:
            return 0
        if padded[0] == '!':
            return (len(padded) ­ 1) / 2
        return (len(padded) + 1) / 2

| #** Why unpadded_length func
| E.g. s = 'd!b!b!a', palindrome = '!b!b!', See, first char is !.
| Then len can be calculated normally. 5//2=2
| But if s = 'a!b!b!a', palindrome = 'a!b!b!a'
| Then if we do normally 7//2=3, So we have to adapt: (7+1)//2=4

# Final solution::

    def longest_palindrome(s):
        s = palindrome_pad(s)
        best = ''
        for center in range(len(s)):
            # Expand symetrically as long as the palindrome property holds
            left = center
            right = center
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            # The last move broke the palindrome property, undo it
            left += 1
            right = 1
            palindrome = s[left:right+1]
            # Record the palindrome if longest
            if unpadded_size(palindrome) > unpadded_size(best): 
                best = palindrome
        return palindrome_unpad(best)

**Solution 3: dynamic programming, O(n)**

Manacher's algorithm. Eliminate redundant expansion steps when the current 
center is inside a bigger palindrome identified previously.

abcbaxyz

If we already identified 'abcba', going a->b->c, then going b->a we can skip because they are inside
a pigger palidromic substring.

(Manacher's optimization is not likely to come up during an interview due to its complexity.
Still, it is good to be aware of its existence.)

| -The alg includes again using guard char to make odd-length. 
| -Also using additional array where we record longest palindrom substring radius found so far.
| radius = [0] * len(s)
| For abcbaxyz
| radius = [0,0,2,0,0..]
| -Uses at most the O(n**2) alg, with some more condition stnts.

124. (LC 647) Palindromic Substrings
---------------------------------------
`647. Palindromic Substrings <https://leetcode.com/problems/palindromic-substrings/>`_
Medium

| Key:
| -When iterating take each character to be the center of a possible palindrome. 

::

    ### Solution 1
    def palindrome_substrings(s):
        ans = 0
        for i in range(len(s)):
            l, r = i, i
            for l, r in ((i, i), (i, i + 1)):
                while l >= 0 and r < len(s) and s[l] == s[r]:
                    ans += 1
                    l -= 1
                    r += 1
        return ans

    ### Solution 1 rewrite (LC accepted 65, 30)
    class Solution:
        def countSubstrings(self, s: str) -> int:
            res = 0
            for center in range(len(s)):
                # L=center
                # R=center and R = center+1
                for L, R in ((center, center), (center, center+1)):
                    while L >= 0 and R < len(s) and s[L] == s[R]:
                        res +=1
                        L-=1
                        R+=1
            return res

    ### My V 1 (LC accepted 5, 85%)
    # Separate loops for strings with odd, even lengths. 
    # I.e. Center 1 char ('bab'), 
    # another loop for strings with center 2 chars ('baab').

    class Solution:
        def countSubstrings(self, s: str) -> int:
            def is_palindrome(s):
                if s == s[::-1]:
                    return True
                return False
            
            res = 0
            for center in range(0, len(s)):
                L=center
                R=center
                while L >= 0 and R < len(s):
                    if is_palindrome(s[L : R+1]):
                        res +=1
                    L-=1
                    R+=1

            for center in range(1, len(s)):
                L=center - 1
                R=center
                while L >= 0 and R < len(s):
                    if is_palindrome(s[L : R+1]):
                        res +=1
                    L-=1
                    R+=1
            return res

    ### The same, in one loop (efficiency though is a bit worse according to LC).
    class Solution:
        def countSubstrings(self, s: str) -> int:
            def is_palindrome(s):
                if s == s[::-1]:
                    return True
                return False
            
            res = 0
            for center in range(0, len(s)):
                L=center
                R=center
                while L >= 0 and R < len(s):
                    # when center is 1 character 'bab'
                    if is_palindrome(s[L : R+1]):
                        res +=1
                    # when center is 2 characters 'baab'
                    if L > 0:
                        if is_palindrome(s[L-1 : R+1]):
                            res +=1
                    L-=1
                    R+=1

            return res

125. (LC 271) Encode and Decode Strings
-------------------------------------------
`271. Encode and Decode Strings <https://www.lintcode.com/problem/659/>`_
Medium

Design an algorithm to encode a list of strings to a string. The encoded string is 
then sent over the network and is decoded back to the original list of strings.

| Example
| Input: dummy_input = ["Hello","World"]
| My Note: Encode: Conversion to single string - e.g."Hello%World"
| Decode result: Output: ["Hello","World"]
 
| Logic
| We will use a delimiter to separate strings.
| 1) We could pick a delimiter outside the 256 range, e.g. 257. Then that is all we have to do.
| 2.1) The follow up of the task is to make our alg be able to encode\decode any range of chars. 
| The we pick any char as a delimiter but also record the length of each string. 
| Example. Input = ["Hello","World"]
| Encoded = "5#Hello5#World"   '#' as delimiter
| 2.2) Use 4 spaces + len(string) as delimiter

::
    
    ### Solution (logic 1, delimiter char 257)
    class Codec:
        def encode(self, strs: List[str]) -> str:
            """Encodes a list of strings to a single string."""
            return chr(257).join(strs)

        def decode(self, s: str) -> List[str]:
            """Decodes a single string to a list of strings."""
            return s.split(chr(257))


    # Usage
    # codec = Codec()
    # codec.decode(codec.encode(strs))

| ### Solution (Logic 2.1, delimiter # and len )
| O(n), encode + decode
| Explained
| i is the pointer where we are in the string.
| j is the pointer that points to delimiters
|             length = int(s[i:j])
| for str like '5#World'
| We get the 5 (we read the integer before the delimiter).

::

    # V1
    class Codec:
        def encode(self, strs):
            res = ""
            for s in strs:
                res += str(len(s)) + "#" + s
            return res

        def decode(self, s):
            res, i = [], 0
            while i < len(s):
                j = i
                while s[j] != "#":
                    j += 1
                length = int(s[i:j])
                res.append(s[j + 1 : j + 1 + length])
                i = j + 1 + length
            return res

    # V2
    class Codec:
        def encode(self, strs):
            return ''.join(map(lambda s: f"{len(s)}#{s}", strs))

        def decode(self, s):
            res = []
            i = 0
            while i < len(s):
                j = i
                while s[j] != '#':
                    j += 1
                length = int(s[i:j])
                i = j + 1
                j = i + length
                res.append(s[i:j])
                i = j
            return res
    c = Codec()
    encoded = c.encode(strs)
    print(encoded) # 5#Hello5#World
    decoded = c.decode(encoded)
    print(decoded) # ['Hello', 'World']

**Solution 2.2** (4 spaces as delimiter)

>>> s="world"
>>> ans=[]
>>> ans.append('{:4}'.format(len(s))+s)
['   5world']

::

    class Codec:
        def encode(self, strs: List[str]) -> str:
            """Encodes a list of strings to a single string."""
            ans = []
            for s in strs:
                ans.append('{:4}'.format(len(s)) + s)
            return ''.join(ans)

        def decode(self, s: str) -> List[str]:
            """Decodes a single string to a list of strings."""
            ans = []
            i, n = 0, len(s)
            while i < n:
                size = int(s[i : i + 4])
                i += 4
                ans.append(s[i : i + size])
                i += size
            return ans





