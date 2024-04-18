Array Questions Extra
======================
137. (LC 347) Top K Frequent Elements
----------------------------------------
`347. Top K Frequent Elements <https://leetcode.com/problems/top-k-frequent-elements/>`_
Medium

| O(n) time and space
| **Bucket sort algorithm** [:ref:`10 <ref-label>`]
| *Approach 1*
| You make a 'bucket' array, where:
| index - initial nums array values
| bucket value - frequency
| Problem: in our nums array we could have values [1,1,2, 100]
| Then our 'bucket array' would need to have 100 indexes.

| *Approach 2*
| 'Bucket' array:
| index - frequency  <===
| value - list of nums values
| Then len(bucket)+1 =len(nums)
| E.g. 
| nums = [1,1,1,2,2,3]
| bucket = [[], [3], [2], [1], [], [], []]  (i.e. e.g. value 1, which we record at index 1, 
| appears 3 times)

::

    ### Solution
    class Solution:
        def topKFrequent(self, nums: List[int], k: int) -> List[int]:
            count = {}
            freq = [[] for i in range(len(nums) + 1)]

            for n in nums:
                count[n] = 1 + count.get(n, 0)
            for n, c in count.items():
                freq[c].append(n)

            res = []
            for i in range(len(freq) - 1, 0, -1):  #(start, end, step in reverse)
                for n in freq[i]:           #for n in [1], so we get 1, not [1]
                    res.append(n)
                    if len(res) == k:
                        return res

    ### My V3 (bucket sort)
    import collections, itertools
    def f(a, k):
        b = [[] for i in range(len(a) + 1)]  # bucket where i=freq, val=val
        cnt = collections.Counter(a)
        ans = []
        for key, v in cnt.items():
            b[v].append(key)
        for i in reversed(range(len(b))):
            if b[i] != [] and k > 0:
                ans.append(b[i])
                k -= len(b[i])
        return list(itertools.chain.from_iterable(ans))

    ### My V2 (bucket sort)
    import collections
    def f(a, k):
        cnt = collections.Counter(a)
        b = [[] for i in range(len(a) + 1)]  # bucket
        for k, f in cnt.items():  # k=key=value in nums, f for frequency
            b[f].append(k)
        ans = []
        for i in range(len(b) - 1, 0, -1):
            for n in b[i]:
                if len(ans) < k - 1:
                    ans.append(n)

            # for j in range(len(b[i])):
            # if len(ans) < k - 1:
            #     ans.append(b[i][j])
        return ans

    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    print(f(nums, k)) #[1,2]

    ### My V1
    def f(a, k):
        cnt = collections.Counter(a)
        freq = [(v, k) for k, v in cnt.items()]
        ans = []
        for _ in range(k):
            m = max(freq)
            ans.append(m[1])
            freq.remove(m)
        return ans

    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    print(f(nums, k)) #[1, 2]




















