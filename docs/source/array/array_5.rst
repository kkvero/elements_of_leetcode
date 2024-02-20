Array Questions Part 5
======================
76. (LC 896) Monotonic Array
-------------------------------
`896. Monotonic Array <https://leetcode.com/problems/monotonic-array/>`_
Easy
::

    class Solution:
        def isMonotonic(self, nums: List[int]) -> bool:
            isIncr = isDecr = False
            for i, v in enumerate(nums[1:]):
                if v < nums[i]:
                    isIncr = True
                elif v > nums[i]:
                    isDecr = True
                if isIncr and isDecr:
                    return False
            return True

    class Solution:
        def isMonotonic(self, nums: List[int]) -> bool:
            incr = all(a <= b for a, b in pairwise(nums))
            decr = all(a >= b for a, b in pairwise(nums))
            return incr or decr

    ### My V1 (LC accepted, pretty efficient, beats T 90%, S 67%)
    def f(nums):
        if nums[0] > nums[len(nums) - 1]:
            up, down = False, True
        else:
            up, down = True, False
        for i in range(1, len(nums)):
            if up:
                if nums[i] < nums[i - 1]:
                    return False
            elif down:
                if nums[i] > nums[i - 1]:
                    return False
        return True

    ### My V2
    import itertools
    def monotonic2(a):
        L2 = [a <= b for a, b in itertools.pairwise(a)]
        L3 = [a >= b for a, b in itertools.pairwise(a)]
        if all(L2) or all(L3):
            return True
        return False



















