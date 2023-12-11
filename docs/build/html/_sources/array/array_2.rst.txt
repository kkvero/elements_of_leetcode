Array Questions Part 2
======================
46. (LC 280)  Computing an alternation
---------------------------------------
LC 280. Wiggle Sort ::

    ### Solution to Leetcode
    class Solution:
        def wiggleSort(self, nums: List[int]) -> None:
            """
            Do not return anything, modify nums in-place instead.
            """
            for i in range(1, len(nums)):
                if (i % 2 == 1 and nums[i] < nums[i - 1]) or (
                    i % 2 == 0 and nums[i] > nums[i - 1]
                ):
                    nums[i], nums[i - 1] = nums[i - 1], nums[i]

*Task* [:ref:`2 <ref-label>`].
Write a program that takes an array A of n numbers, and rearranges A's elements to get a new array
B having the property that B[0] <= B[1] >= B[2] <= B[3] >= B[4] < B[5] >=....

| (I.e. There is an interleaving: 
| n less than neighbors, n greater than neighbors, n less than neighbors etc)
| Example.
| A = [2,1,5,7,8]
| B = [1,5,2,8,7] #5 greater than adjacents, 2 smaller, 8 greater
 
| # Key to solution
| Alternate sorting in direct and reverse order (based on whether index is odd ot even).
 
| # The straightforward solutions, O(N log N)
| 1) to sort A and interleave the bottom and top halves of the sorted array. 
| 2) sort A and then swap the elements at the pairs (A[1],A[2]),(A[3],A[4]),....
| Both these approaches have the same time complexity as sorting, namely O(n log n).

3) But it is not necessary to sort A to achieve the desired configuration -
you could simply rearrange the elements around the median, and then perform the 
interleaving.
Median finding can be performed in time O(n), which is the overall time complexity of this approach.

*Logic*, O (N).
You may notice that the desired ordering is very local, and it is not necessary
to find the median. Iterating through the array and swapping A[i] and A[i + 1] 
when i is even and A[i] > A[i+1] or i is odd and A[i] < A[i + 1] achieves the desired configuration.

| This approach has the same time complexity as the median finding (O(N)), but it is [:ref:`2 <ref-label>`]:
| - easier to implement, 
| - never needs to store more than two elements in memory or read a previous element. 
| - illustrates algorithm design by iterative refinement of a brute-force solution.

::

    ### Solution
    def rearrange(A):
        for i in range (len(A)):
            A[i:i+2] = sorted(A[i:i + 2], reverse=i% 2)

    ### Solution my V (less magic)
    def alternation(a):
        a.sort()
        for i in range(len(a) - 1):
            if not (i & 1): 
                a[i], a[i + 1] = a[i + 1], a[i]
        return a


    A = [2, 1, 5, 7, 8]   # [2, 1, 7, 5, 8]
    print(alternation(A))

    A = [2,1,5,7,8]
    rearrange(A)
    print(A)  # [1, 5, 2, 8, 7]
 
| **Explained** (main version)
| *Main aspects*
| 1)A[i:i+2] - is a slice of 2 numbers in A
| A = [2,1,5,7,8] => 2,1; 1,5; 5,7 etc
| 2)sorted(A[i:i+2]) - is sorting this slice of 2 numbers
| 3)reverse=i% 2 - sets reverse to True or False 

(when i is even, the remainder = 0, setting reverse to False. When i is odd,
any value of the remainder will set reverse to True)

| The main principle - we make sure when 
| i=even A[i] > A[i+1] (sorted)
| i=odd, A[i] < A[i+1] (reverse sorted)
| Hence achieve the interleaving.


















