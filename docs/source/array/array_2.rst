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

47. (LC 204) Enumerate all primes to n
----------------------------------------

| 204. Count Primes 
| (Medium)
| Here the task is to return a list of the primes [:ref:`2 <ref-label>`].

A natural number is called a prime if it is bigger than 1 and has no divisors 
other than 1 and itself.
Write a program that takes an integer argument and returns all the primes 
between 1 and that integer. 
For example, if the input is 18, you should return <2,3,5,7,11,13,17>.

| **Square root approach**
| An improved brute force is to iterate through all numbers and do 'trial division',
| dividing each integer from 2 to the square root of i+1.

For the rule is - since if i has a divisor other than 1 and itself, it must also 
have a divisor that is no greater than its square root.

| *Complexity*
| Each test has time O(sqroot n), 
| the entire computation time is O(n * sqroot n), i.e. O (n ** 3/2) 

::

    def is_prime(num):
        for i in range(2, int(math.sqrt(num)) + 1):  #Don't forget +1
            if num % i == 0:
                return False
        return True

    def primes(n):
        return [x for x in range(2, n) if is_prime(x)]

    print(primes(10)) #[2, 3, 5, 7]


| **Sieving approach**
| *Complexity*
| O(n log log n)
| Sieving is superior to trial-division.

| *Basic Logic*
| We exclude the multiples of primes. (Because multiples of primes cannot be primes themselves.)

When a number is identified as prime, we sieve it, i.e. remove all its multiples 
from future consideration.
We use a Boolean array to encode the candidates. E.g. [F,F,T,T,T,T]. We can set 
the first two to false because 0 and 1 are not primes. ::

    ### Solution (sieving)
    # Given n, return all primes up to and including n.
    def generate_primes(n):
        primes = []
        # is_prime[p] represents if p is prime or not. Initially, set each to
        # true, expecting 0 and 1. Then use sieving to eliminate nonprimes.
        is_prime = [False, False] + [True] * (n - 1)
        for p in range(2, n + 1):
            if is_prime[p]:        #enter only if value at index p is True
                primes.append(p)   #append num 2
                # Sieve p's multiples.
                for i in range(p, n + 1, p):  #use step, range 3rd param
                    is_prime[i] = False
        return primes

    # no comments  
    def generate_primes(n):
        primes = []
        is_prime = [False, False] + [True] * (n - 1)
        for p in range(2, n + 1):
            if is_prime[p]:        
                primes.append(p)   
                for i in range(p, n + 1, p):
                    is_prime[i] = False
        return primes

    print(generate_primes(11))  #[2, 3, 5, 7, 11]

48. Permute the elements of an array
--------------------------------------
A permutation is a rearrangement of members of a sequence into a new sequence.
A permutation can be specified by an array P, where P[i] represents the location 
of the element. 

| E.g. A = ['a','b','c','d']
| perm = [2,0,1,3]
| This maps:
| a,b,c,d
| 2,0,1,3 (map 'a' to i=2)
| b,c,a,d

| *Task*
| Given an array A of n elements and a permutation P, apply P to A.

| Hint: Any permutation can be viewed as a set of cyclic permutations. 
| For an element in a cycle, how would you identify if it has been permuted?

| *time and space O(n)*
| When using an additional array to write the resulting array.

::

    ### My V2
    def perm(a, p):
        for i in range(len(a)):
            if type(a[i]) != tuple:
                a[i] = tuple(
                    a[i],
                )
                a[i], a[p[i]] = a[p[i]], a[i]
        return [t[0] for t in a]

    A = ["a", "b", "c", "d"]
    p = [2, 0, 1, 3]
    print(perm(A, p))  #['b', 'c', 'a', 'd']

    ### My version, using dict to mark permuted elements
    def permute(a, p):
        index = 0
        d = {}
        while index < len(p):
            for i in range(len(a)):
                if a[i] not in d:
                    d[a[i]] = 1
                    a[i], a[p[index]] = a[p[index]], a[i]
                index += 1
        return a

    A = ["a", "b", "c", "d"]
    perm = [2, 0, 1, 3]
    print(permute(A, perm)) #['b', 'c', 'a', 'd']

| *O(n) time, O(1) space*
| We don't use an additional array, we change A in place.

Independent cyclic permutations: We keep going forward from i to P[i] till we get 
back to i.
After we are done with that cycle, we need to find another cycle that has not yet
been applied. It can be done by storing a Boolean for each array element.
But another way (that will give us O(n1) space) is to use the sign bit
in the entries in the permutation-array. Specifically, we subtract n from P[i] after applying it. This
means that if an entry in P[i] is negative, we have performed the corresponding move. 

| **Solution**
| Key: subtract len of p from p[i] to mark permuted items.

::

    def apply_permutation(perm, A):
        for i in range(len(A)):
            next = i
            # Check if perm[i] is nonnegative (i.e. check if the element at index i 
            # has not been moved
            while perm[next] >= 0:
                A[i], A[perm[next]] = A[perm[next]], A[i]
                temp = perm[next]
                # Subtracts len(perm) from an entry in perm to make it negative,
                # which indicates the corresponding move has been performed.
                perm[next] -= len(perm)
                next = temp
        # Restore perm.
        perm[:] = [a + len(perm) for a in perm]

    A = ['a','b','c','d']
    perm = [2,0,1,3]
    apply_permutation(perm, A)
    print(A)  #['b', 'c', 'a', 'd']

| # In the loop
| 1)
| i=0
| next=0
| A = [c,b,a,d]
| perm = [-2,0,1,3]
| next = 2
| 2)
| i=0 still
| next=2
| A = [b,c,a,d]
| perm = [-2,0,-3,3]
| next=1

(Though we already got our aimed at permuted array, we will go on for some time
with the loop. Then return to the main i loop, but that will finish quickly as 
our values in perm get negative, and while loop works only for positives.)

49. (LC 31) Compute the next permutation
-------------------------------------------
| 31. `Next Permutation <https://leetcode.com/problems/next-permutation/>`_
| Medium

(dictionary ordering)

| :py:func:`itertools.permutations()`
| itertools.permutations(iterable, r=None)
| Return successive r length permutations of elements in the iterable.

If r is not specified or is None, then r defaults to the length of the iterable 
and all possible full-length permutations are generated.
 
If the input iterable is sorted, the output tuples will be produced in sorted order.

Elements are treated as unique based on their position, not on their value. 
So if the input elements are unique, there will be no repeated values within a permutation.

Roughly it means:
# permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
# permutations(range(3)) --> 012 021 102 120 201 210

Exactly it means:

>>> import itertools
>>> itertools.permutations('ABCD', 2)
<itertools.permutations object at 0x7fa8b147a1b0>
>>> list(_)
[('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'A'), ('B', 'C'), ('B', 'D'), ('C', 'A'), 
('C', 'B'), ('C', 'D'), ('D', 'A'), ('D', 'B'), ('D', 'C')]
>>> it = itertools.permutations('ABCD', 2)
>>> it.__next__()
('A', 'B')
>>> it.__next__()
('A', 'C')

50. Sample offline data
-------------------------
Implement an algorithm that takes as input an array of elements and returns
a subset of the given size of the array elements. All subsets should be equally 
likely. Return the result in input array itself.

# Practical applications -
To select a random subset of customers to test a new UI (see if it increases
the visit duration) before rolling out the change.

| # Implementation 
| A = [3,7,5,11], k=3 where k is the size of a new array, i.e. the limit.
| We use the random number generator to pick a random integer in the interval [0, len(A)-1].
| It is going to be a random index in our original array.
| E.g. randint(0,3) returns 2. It means we now swap A[0] with A[2].
| (Note, 3 is not k, it is len(A)-1 -> 4-1. Also 2 is the index in A.)
| Now A = [5,7,3,11]
| Then we do randint(1,3). Etc. 
| (Left bound moves +1 towards the len of A. We make k number of such random picks.)

::

    ### Solution
    import random
    def random_sampling(k, A):   #k is size of the new array
        for i in range(k):
            # Generate a random index in [i, len(A) - 1].
            r = random.randint(i, len(A) - 1)
            A[i], A[r] = A[r], A[i]
        return A[:k]

    # OR
        # index = random.randrange(i, len(a))

    A = [3,7,5,11]
    print(random_sampling(3, A)) #example output [7, 11, 5]

# My note - 
Since we use the stdlib module random anyway, 
why not use the dedicated .sample(list, sample_size)
``print(random.sample(A, 3)) #[5, 3, 11]``











