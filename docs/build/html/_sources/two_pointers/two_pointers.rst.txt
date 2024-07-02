Two Pointers Questions Part 1
===============================
138. (LC 167) Two Sum II - Input Array Is Sorted
----------------------------------------------------
`167. Two Sum II - Input Array Is Sorted <https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/>`_
Medium

**Key:**
If you are given a sorted array, consider moving from both ends toward the center.

**Solution 1** [:ref:`10 <ref-label>`]
O(n) ::

    class Solution:
        def twoSum(self, numbers: List[int], target: int) -> List[int]:
            l, r = 0, len(numbers) - 1

            while l < r:
                curSum = numbers[l] + numbers[r]

                if curSum > target:
                    r -= 1
                elif curSum < target:
                    l += 1
                else:
                    return [l + 1, r + 1]

**Explained:** ::

            if curSum > target:
                r -= 1

| How do we know to decrease the right pointer.
| Consider our sorted a=[2,7,11,15], t=9
| 2+15=17 >9
| So we know we need a smaller sum.
| If we moved lp, 7+15 we would only be making the sum greater.
| So we must move the right pointer.

::

    ### My V (LC accepted 70, 80)
    class Solution:
        def twoSum(self, numbers: List[int], target: int) -> List[int]:
            p1 = 0
            p2 = len(numbers)-1
            while p1 < p2:
                sum_ = numbers[p1] + numbers[p2]
                if sum_ == target:
                    return [p1+1, p2+1]
                elif sum_ > target:
                    p2 -= 1
                else:
                    p1 +=1

139. (LC 42) Trapping Rain Water
------------------------------------
`42. Trapping Rain Water <https://leetcode.com/problems/trapping-rain-water/>`_
Hard

| 2 approaches.
| Both are O(n) time.
| One is O(n) space, optimization with pointers is O(1) space.
| 
| Keys (O(n) space)
| -find running max for right side
| -find running max for left side (while calculating water trapped)
| Water trapped = min(current_left_max, current_right_max) - current_height
| -If water trapped > 0, then add to ans.
| 
| # The one with O(n) space, linear traversal.
| We traverse the given array height=[0,1,0,2,1,0,1,3,2,1,2,1]

.. admonition:: The formula for trapped water is:

    min(maxLeft, maxRight) - height[i]

| So we calculate for each index, including here at index 1, h=[1,2,3]
| min(mL,mR) - h[i] = min(1,3)-h[1] = 1-2=-1 (no water trapped)
| 
| Calculating maxL, maxR for a current index gives us the closest max left and max right walls.
| E.g.

::

    # Illustration
    # height:    0,1,0,2,1,0,1,3,2,1,2,1
    # maxLeft:   0 0 1 1 2 2 2 2 3 3 3 3  -->
    # maxRight:  3 3 3 3 3 3 3 2 2 2 1 0  <--
    # min(mL,mR) 0 0 1 1 2 2 2 2 2 2 1 0  ^V
    # trapped w: 0 0 1 0 1 2 1 0 0 1 0 0 (some zeros are actually -N, but we make sure we make it 0)

::

    ### My V O(N) (LC accepted 25, 70%)
    class Solution:
        def trap(self, height: List[int]) -> int:
            # Calculating running max for values at the right side
            max_right = []
            max_v = height[-1]
            for i in reversed(range(len(height))):
                max_v = max(max_v, height[i])
                max_right.append(max_v)
            max_right = max_right[::-1]

            #Calculating running max for left side. At the same time as calculating water trapped.
            max_left_v = height[0]
            ans=0
            for j in range(len(height)):
                max_left_v = max(max_left_v, height[j])
                min_v = min(max_left_v, max_right[j])
                water = min_v - height[j]
                if water > 0:
                    ans += water
            
            return ans

| **Logic when using pointers**
| -Initiate pointers.
| -Initiate leftMax, rightMax
|     Iteration:
| -Compare leftMax and rightMax, we will be moving that pointer, which max is smaller.
| (If they have the same value, we move whichever.)
| height=[0,1,0,2,1,0,1,3,2,1,2,1]
|         L                     R
| leftMax=0, rightMax=1
| mL<mR, so we would move L+=1
| -Check if we found a new leftMax or rightMax.
|     leftMax = max(leftMax, height[l])
| -Calculate water we can trap
| leftMax - height[l]
| Note that because we first assigned a new leftMax and if current h[i]= new leftMax
| then water trapped = h[i]-h[i]=0. This is how we avoid negative amounts of water trapped.

::

    ### Two pointers
    class Solution:
        def trap(self, height: List[int]) -> int:
            if not height:
                return 0

            l, r = 0, len(height) - 1
            leftMax, rightMax = height[l], height[r]
            res = 0
            while l < r:
                if leftMax < rightMax:
                    l += 1
                    leftMax = max(leftMax, height[l])
                    res += leftMax - height[l]
                else:
                    r -= 1
                    rightMax = max(rightMax, height[r])
                    res += rightMax - height[r]
            return res

    ### My V3 
    def trap_water(a):
        lp = 0
        rp = len(a) - 1
        maxL = a[lp]
        maxR = a[rp]
        water = 0
        while lp < rp:
            if maxL < maxR:  #From this we already know we're dealing with the result of minH=min(maxL, MaxR)
                lp += 1
                maxL = max(a[lp], maxL) 
                water += maxL - a[lp]
            else:
                rp -= 1
                maxR = max(a[rp], maxR)
                water += maxR - a[rp]
        return water

140. (LC 259) 3Sum Smaller
-----------------------------
`259. 3Sum Smaller <https://leetcode.com/problems/3sum-smaller>`_
(Locked content)

Given an array of n integers nums and an integer target, find the number of 
index triplets i, j, k with 0 <= i < j < k < n that satisfy the condition 
nums[i] + nums[j] + nums[k] < target.

| Example 1:
| Input: nums = [-2,0,1,3], target = 2
| Output: 2
| Explanation: Because there are two triplets which sums are less than 2:
| [-2,0,1]
| [-2,0,3]

**Keys:**
If we need 3 pointers, make the third P the index in array.

| -Sort the input array.
| -initiate pointers within the loop len(a)
| leftP=i+1
| So e.g.:

::

    # Illustration
    # [-2,0,1,3]
    #   i L   R

| So L is actually our middle pointer, i index is our leftmost pointer.
| -while L<R: (i is constant in this loop)
| move R-=1 if sum>=t
| otherwise, ATTENTION add to answer rp-lp, and lp+=1 AFTER that
| (by rp-lp we add all combinations between rp and lp)
| -move on to the next i, assign L, R anew:

::

    # [-2,0,1,3]
    #     i L R

::

    ### Solution
    def f(a, t) -> int:
        a.sort()
        ans = 0
        for i in range(len(a)):
            lp = i + 1
            rp = len(a) - 1
            while lp < rp:
                s = a[i] + a[lp] + a[rp]
                if s >= target:
                    rp -= 1
                else:
                    ans += rp - lp   #<===
                    lp += 1
        return ans

    nums = [-2, 0, 1, 3]
    target = 2
    print(f(nums, target)) #2
    nums2 = [0, 1]
    target2 = 0
    print(f(nums2, target2)) #0

141. (LC 844) Backspace String Compare
-----------------------------------------
`844. Backspace String Compare <https://leetcode.com/problems/backspace-string-compare/>`_
Easy

| Solution ideas:
| -Initiate 2 pointers, both at the end of each string.
| -Initiate vars skip1, skip2 to keep track of the encountered backspace #.
| -Main while loop while i >= 0 or j >= 0
| -inner 2 while loops for each pointer 'while pointer >=0', check for #, use break
| -at the end of the main while loop check if one of the pointers reached the end of string.

::

    class Solution:
        def backspaceCompare(self, s: str, t: str) -> bool:
            i, j, skip1, skip2 = len(s) - 1, len(t) - 1, 0, 0
            while i >= 0 or j >= 0:
                while i >= 0:
                    if s[i] == '#':
                        skip1 += 1
                        i -= 1       #<==1
                    elif skip1:
                        skip1 -= 1
                        i -= 1       #<==2
                    else:
                        break
                while j >= 0:
                    if t[j] == '#':
                        skip2 += 1
                        j -= 1
                    elif skip2:
                        skip2 -= 1
                        j -= 1
                    else:
                        break
                if i >= 0 and j >= 0:  #AND
                    if s[i] != t[j]:
                        return False
                elif i >= 0 or j >= 0:  #OR
                    return False
                i, j = i - 1, j - 1
            return True

142. (LC 977) Squares of a Sorted Array
---------------------------------------------
`977. Squares of a Sorted Array <https://leetcode.com/problems/squares-of-a-sorted-array/>`_
Easy

| [:ref:`10 <ref-label>`]
| # Time: O(n), one pass using two pointers.
| # Space: O(1), output array is not considered for space complexity.
 
| Idea:
| Recognize that sorted array like [-4, -1, 0, 3, 10], squares are [16,1,0,9,100].
| An array like that will have greatest values at its far ends.
| ===> We can iterate with two pointers from the far ends of the squared array.
| Check which value at the pointer is greater, put that value into a new 'ans' array
| in reverse order, e.g. ans=[0,0,0,0,100]. Or put from front and reverse when giving the final answer.

::

    ### Solution V1 + my V
    def f(a):
        lp, rp = 0, len(a) - 1
        ans = []
        a = [x**2 for x in a]
        while lp <= rp:
            if a[rp] > a[lp]:
                ans.append(a[rp])
                rp -= 1
            else:
                ans.append(a[lp])
                lp += 1
        return ans[::-1]

    ### My V
    def f(a):
        lp, rp = 0, len(a) - 1
        ans = [0] * len(a)
        a = [x**2 for x in a]
        i = 1
        while lp <= rp:
            if a[rp] > a[lp]:
                ans[-i] = a[rp]
                rp -= 1
            else:
                ans[-i] = a[lp]
                lp += 1
            i += 1
        return ans

    a = [-4, -1, 0, 3, 10]
    print(f(a)) #[0, 1, 9, 16, 100]

    ### Solution V2
    class Solution:
        def sortedSquares(self, nums: List[int]) -> List[int]:
            n = len(nums)
            res = [0] * n
            l, r = 0, n - 1
            
            while l <= r:
                left, right = abs(nums[l]), abs(nums[r])
                if left > right:
                    res[r - l] = left * left
                    l += 1
                else:
                    res[r - l] = right * right
                    r -= 1
            return res

**My solution.** If you forgot to traverse from opposite ends. (LC accepted: 72, 78%)
(Traversing from left if all positives, from right if all negatives, from somewhere
in the middle if both pos and neg). ::

    class Solution:
        def sortedSquares(self, nums: List[int]) -> List[int]:
            if len(nums) == 1:
                return [nums[0] ** 2]
            if nums[0] >= 0 and nums[len(nums) - 1] >= 0:
                lp, rp = -1, 0
            elif nums[0] < 0 and nums[len(nums) - 1] < 0:
                lp, rp = len(nums) - 1, len(nums)
            else:
                for i in range(1, len(nums)):
                    if nums[i] >= 0:
                        lp = i - 1
                        rp = i
                        break
            ans = []
            while lp >= 0 or rp < len(nums):
                if rp >= len(nums):
                    res = nums[lp] ** 2
                    lp -= 1
                elif lp < 0:
                    res = nums[rp] ** 2
                    rp += 1
                elif abs(nums[lp]) < nums[rp]:
                    res = nums[lp] ** 2
                    lp -= 1
                else:
                    res = nums[rp] ** 2
                    rp += 1
                ans.append(res)
            return ans













