Sliding Window Questions Part 1
================================
146. (LC 567) Permutation in String
----------------------------------------
`567. Permutation in String <https://leetcode.com/problems/permutation-in-string/description/>`_
Medium

| **Solution 1** [:ref:`7 <ref-label>`]
| Key points:
| 1)Sliding window
| 2)Hash tables, just compare hash tables.
 
| -we make hash tables using collections.Counter()
| Of the full s1.
| Of the part of s2, window size, which is len of s1.
| -we can compare dicts!
| if cnt1 == cnt2:
| -Move the window. Iterate starting from len(s1).
| (Modify hash tables. Simply +=1 to the value at rp, -= to the key in hash at lp)

::

    from collections import Counter
    class Solution:
        def checkInclusion(self, s1: str, s2: str) -> bool:
            n = len(s1)
            cnt1 = Counter(s1)
            cnt2 = Counter(s2[:n])
            if cnt1 == cnt2:
                return True
            for i in range(n, len(s2)):
                cnt2[s2[i]] += 1
                cnt2[s2[i - n]] -= 1
                if cnt1 == cnt2:
                    return True
            return False

    ### My remake
    def f(s1, s2):
        if len(s1) > len(s2):
            return False
        cnts1 = collections.Counter(s1)
        cnts2 = collections.Counter(s2[: len(s1)])
        if cnts1 == cnts2:
            return True
        lp = 0
        for rp in range(len(s1), len(s2)):
            cnts2[s2[rp]] += 1
            cnts2[s2[lp]] -= 1
            lp += 1
            if cnts1 == cnts2:
                return True
        return False

| **Note**
| we use special properties of dicts, created by collections.Counter(iterable)
| 1)
|     cnts2[s2[rp]] += 1
| Counter makes a dict with properties like defaultdict. So we can add to non-existent keys,
| which doesn't work in a regular dict.

>>> import collections
>>> s='abc'
>>> cnt=collections.Counter(s)
>>> cnt['f']+=1   #'f' is not in cnt, but we add to it OK
>>> cnt
Counter({'a': 1, 'b': 1, 'c': 1, 'f': 1})
>>> d={'b':1}
>>> d['c']+=1     #Nope in reg dict
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'c'

| 2)
|     ``if cnts1 == cnts2:``
| cnts like cnt1={'a':1, 'd':0}, cnt2={'a':1} will be equal, 
| again which is not the case for normal dict.



| **Solution 2** (If you don't want to use Python tricks.) [:ref:`10 <ref-label>`]
| -What is your sliding window going to be?
| The size of your window in s2 (longer string) is the len of s1 (shorter string).
 
| We will have 2 hash maps: for chars in s1 and those in s2.
| Arrays are used here: [0]*26
| In each hash/array we will be tracking the counts for all 26 chars. 
| No char in string will have count 0, char present, count=1 
 
| Variable matches=0
 
| E.g.
| s1='abc', s2='baxyzabc'
| window 1 = 'bax'
| matches=24
| window 2 = 'axy'
| matches=22
| ..Stop when we have matches=26

::

    class Solution:
        def checkInclusion(self, s1: str, s2: str) -> bool:
            if len(s1) > len(s2):
                return False

            s1Count, s2Count = [0] * 26, [0] * 26
            for i in range(len(s1)):
                s1Count[ord(s1[i]) - ord("a")] += 1  #fills in hash/array for s1
                s2Count[ord(s2[i]) - ord("a")] += 1  #fills in for window1 of s2

            matches = 0
            for i in range(26):  #matches for window1,yes iterate through all 26 values
                matches += 1 if s1Count[i] == s2Count[i] else 0

            l = 0
            for r in range(len(s1), len(s2)):  #<==Move RP
                if matches == 26:
                    return True

                index = ord(s2[r]) - ord("a")    #BLOCK 1 for rp
                s2Count[index] += 1
                if s1Count[index] == s2Count[index]:
                    matches += 1
                elif s1Count[index] + 1 == s2Count[index]: #if in s1 'a'=0, in s2 'a'=1
                    matches -= 1

                index = ord(s2[l]) - ord("a")    #BLOCK 2 for lp
                s2Count[index] -= 1
                if s1Count[index] == s2Count[index]:
                    matches += 1
                elif s1Count[index] - 1 == s2Count[index]:  #if in s1 we had 'a'=1, now in s2 'a'=0
                    matches -= 1
                l += 1                      #<==Move LP
            return matches == 26

    ### My V (LC accepted 78,84%)
    def f(s1, s2):
        d = {}
        for c in s1:
            d[c] = 1 + d.get(c, 0)
        total_cnt = sum([v for v in d.values()])
        lp = rp = 0
        # for rp in range(len(s2)):  NOPE
        while rp < len(s2):
            if s2[rp] in d and d[s2[rp]] > 0:
                d[s2[rp]] -= 1
                total_cnt -= 1
                if total_cnt == 0:
                    return True
            else:
                if s2[rp] not in d:   #s2[rp] not in D, build from scratch
                    while lp != rp:
                        d[s2[lp]] += 1
                        lp += 1
                        total_cnt += 1
                else:               #s2[rp] in D, don't build from scratch, just try moving lp
                    d[s2[lp]] += 1
                    total_cnt += 1
                    rp -= 1       #Move rp BACK and try again

                lp += 1
            rp += 1  #DONT FORGET
        return False

| Seems to be a classic sliding window.
| One tricky edge case:
| s1 = "adc"
| s2 = "dcda" #True has s1 substring

::

    # d c d a
    # L   R

| Case 1) value at R in hash and D[s2[R]]> 0, so L stays, R +=1
| Case 2) value at R not in hash, just move L all the way to R (L=R), start looking from scratch.
| -> spec. case 3) value at R in hash, but D[s2[R]]=0.
| Then L+1, R-1 and try again.

147 (LC 239) Sliding Window Maximum
-------------------------------------
`239. Sliding Window Maximum <https://leetcode.com/problems/sliding-window-maximum/description/>`_
Hard

| The example given in the task is misleading (makes you think that mono increasing stack is enough).
| A better one that covers more cases:
| nums = [0, 5, 3, 2, 1, 2, 4, 4]
 
| **Keys**
| -use dequeue, mono decreasing, store (value, index)
| E.g.: q = [(5,1), (3,2), (2,3)] (so the greatest value is always at q[0])
| -pop from dequeue if current value greater than previous values, q[-1]
| cur=2, q=[(2,3), (1,4)], pop val=1 => q=[(2,3), [2,5]]
| -popleft if index of the value on the left of q (q[0] i.e. greatest) is beyond our k window
| cur i = 4, q = [(5,1), (3,2), (2,3)], pop (5,1)
| -append to ans value at q[0], first value in q is always the greatest.

QUEUE (my V)::

    ### My V (LC accepted 30, 50)
    class Solution:
        def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
            ans = []
            q = collections.deque()
            for i, n in enumerate(nums):
                while q and q[-1][0] < n:
                    q.pop()
                while q and q[0][1] <= i-k:
                    q.popleft()
                q.append((n, i))
                if i >= k-1:
                    ans.append(q[0][0])
            return ans



| HEAP
| -Using min heap, store (value, index) pairs.
| Because we may have a situation like this:
| nums = [9,10,9,-7,-4,-8,2,-6]
| k = 5
| Storing just values in heap, would give output: [10,10,9,9] instead of [10,10,9,2].
| (We never deleted 2nd to greatest (9) that we've passed.)
| To delete all values that we've passed, store also indices in heap.

::

    ### My V (LC accepted 10, 5)
    class Solution:
        def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
            q = [(-n, i) for i, n in enumerate(nums[:k-1])]
            heapq.heapify(q)
            ans = []
            for i in range(k-1, len(nums)):
                heapq.heappush(q, (-nums[i], i))
                lp = i+1-k
                while q[0][1] < lp:
                    heapq.heappop(q)
                ans.append(-q[0][0])
            return ans

Solution 2 [:ref:`7 <ref-label>`]::

    class Solution:
        def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
            q = [(-v, i) for i, v in enumerate(nums[: k - 1])]
            heapify(q)
            ans = []
            for i in range(k - 1, len(nums)):
                heappush(q, (-nums[i], i))
                while q[0][1] <= i - k:
                    heappop(q)
                ans.append(-q[0][0])
            return ans

















