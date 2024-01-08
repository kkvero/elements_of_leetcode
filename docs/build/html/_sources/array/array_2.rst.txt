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

51. Sample online data
-------------------------
(In practice it can be to provide a uniform sample of packets for a network session.)

*Task* [:ref:`2 <ref-label>`] -
Given input A size n, write a program that reads input A maintaining a uniform random
subset of size k.

| # Brute force
| Read all packets
| [...], [...], [...], [...]   
| choose randomly using solution 5.12 
| k=2
| [..1.], [...], [...2], [...]   
| Space O(n), time O(nk)
 
| # Space O(k), time proportional to the number of elements in the stream.
| Example. 
| k=2, packets p,q,r,t,u,v
| For our first subset we just take the first two packets {p, q}.
| Selecting the next packet.
| 1)r will be selected with probability k/(n+1), here 2/3 (where n starts at 0).
| Suppose it is not selected.
| 2) t suppose it is selected (probability 2/4)
| Then we replace one of the previously chosen packets with t.
| E.g. we end up with {p,t}.
| 3) etc.

::

    ### V0 my
    import random
    def random_subset1(a, k):
        random.shuffle(a)
        return a[:k]

    ### V2 my
    import random
    def random_subset2(a, k):
        for i in range(k):
            ri = random.randrange(i, len(a))  #generate random index
            a[i], a[ri] = a[ri], a[i]
        return a[:k]

    ### V1 My version (don't know if we can use it here)
    import random
    def random_subset(a, k):
        for i in range(k):
            swap_i = random.choice(range(i, len(a)))
            a[i], a[swap_i] = a[swap_i], a[i]
        return a[:k]

    a = [1, 2, 4, 5, 3]
    print(random_subset(a, 3))  #2 runs, OUT:[1, 3, 2], [4, 2, 5]

.. admonition:: itertools.islice(iterable, stop)

    >>> a = itertools.islice('ABCD', 2)
    >>> a
    <itertools.islice object at 0x7fa8b0b5c540>
    >>> list(a)
    ['A', 'B']

**Solution** ::

    import itertools, random
    def online_random_sample(it, k):
        sampling_results = list(itertools.islice(it, k))
        num_seen_so_far = k
        for x in it:
            num_seen_so_far += 1
            idx_to_replace = random.randrange(num_seen_so_far)
            if idx_to_replace < k:
                sampling_results[idx_to_replace] = x
        return sampling_results

    it = list('pqrtuv')
    print(online_random_sample(it, 2))
    # 3 calls produce random results: ['v', 'q'], ['v', 'p'], ['p', 'q']

    # With comments
    def online_random_sample(it, k):
        # Gets us first result [p, q] (slice input data[0:2])
        sampling_results = list(itertools.islice(it, k))

        # Start sampling starting at index 2, before that is our initial sample
        num_seen_so_far = k
        for x in it:
            num_seen_so_far += 1  #2+1=3
            # In the first loop, choose random index out of 0,1,2.
            # As range() randrange() stops before the given number.
            idx_to_replace = random.randrange(num_seen_so_far)
            if idx_to_replace < k:                     #**
                sampling_results[idx_to_replace] = x
        return sampling_results

| #** This is tricky
| if idx_to_replace < k:
| In the first loop index can be 0,1,2. Our k=2
| If index=2, it means we do not choose element, i.e. here 'r'.
| If index is 0 or 1, we replace the so far [p,q] at index 0 or 1,
| e.g. if index 0, we will have [r,q]

52. (LC 384) Compute a random permutation
------------------------------------------
384. `Shuffle an Array <https://leetcode.com/problems/shuffle-an-array/>`_ - 
Medium

Design an algorithm that creates uniformly random permutations of {0, 1,...,n - 1}.

**Notes**:
Generating random permutations with equal probability is not as straightforward 
as it seems.
Iterating and swapping each element with another randomly does not generate all 
permutations with equal probability.
E.g. when there are n=3 elements. The number of permutations is 3! = 6.
Ways to choose elements to swap is 3**3 = 27.
Since 27 is not divisible by 6, some permutations correspond to more ways than others.

| **Key points**:
| -Store the result in original array A that we make out of list(range(n))
| -Our algorithm should make sure that we don't pick an element that has already been picked, i.e. pick only from the remaining.
| E.g. n=4, thus A = [0,1,2,3].

1)First random number is chosen between index 0 and index 3. We get e.g. index to 
update with=1, we swap current index 0 with index 1. Get A=[1,0,2,3].

2)Next choose from indices 1-3, e.g. random choice gives us index 3, swap i=1 with
index=3. A=[1,3,0,2] etc.

We iterate through A and swap current index with a randomly generated index from
only the remaining indices.
This reminds of the function we used earlier. We are going to use it as a helper 
function for the current solution. ::

    ### Solution
    import random
    def random_sampling(k, A):   #k is size of the new array
        for i in range(k):
            r = random.randint(i, len(A) - 1)  #r is random index 
            A[i], A[r] = A[r], A[i]
        return A[:k]

    def compute_random_permutation(n):
        permutation = list(range(n))
        random_sampling(n, permutation)
        return permutation

    print(compute_random_permutation(5))

    # OUT
    # [2, 3, 0, 4, 1]
    # [0, 3, 2, 4, 1]

    ### V1 my
    # The main point - choose a random index from len of array. 
    # Swap current index with random index.
    # Diminish indices to randomly choose from by 1 (or len - current index)

    import random
    def random_permut(n):
        a = list(range(n + 1))
        le = len(a) - 1
        for i in range(le):
            index = random.randint(i, le)
            a[i], a[index] = a[index], a[i]
        return a

    print(random_permut(5))  # [1, 0, 2, 3, 5, 4], run2 [2, 5, 3, 1, 4, 0]

    ### V2 my
    import random
    def shuffle_array2(a):
        for i in range(len(a) - 1):
            ri = random.randrange(i, len(a))
            a[i], a[ri] = a[ri], a[i]
        return a

    ### FYI. Python stdlib tools
    #1
    import random
    def shuffle_array(a):
        random.shuffle(a)
        return a

#2

>>> import itertools
>>> it = itertools.permutations(range(0,4))
>>> it.__next__()
(0, 1, 2, 3)
>>> it.__next__()
(0, 1, 3, 2)

53. Generate nonuniform random numbers
---------------------------------------
| [:ref:`2 <ref-label>`]
| # In essence
| We are given probabilities of occurrence of some numbers.
| Create a nonuniform random number generator.
| (Before we randomly generated numbers that could occur with equal probability.
| Well here each number can occur with a different probability.
| E.g. 7 with P=0.5, 9 with P=0.2 etc.)

# Practical application -
Load test for a server. You have the inter-arrival time of requests to the server 
over a period of one year. You have a histogram of the distribution of the arrival
times of requests. In the load test you would like to simulate data arriving at the 
same time as the distribution observed historically.

| # Example
| You are given n numbers and probabilities p0,p1,...,Pn-1, which sum up to 1.
| E.g. we have:
| values = [3,5,7,11] # i.e. 3 packets arrived etc. I suppose
| probabilities = [9/18, 6/18, 2/18, 1/18]
| (Then in 1000 000 calls to your program, 3 should appear about 500 000 times, 5 - 333 333 times.)

| # Python stdlib tools we will use here
| 1) itertools.accumulate([1,2,3,4,5]) --> 1 3 6 10 15
| 2) >>> random.random()
| 0.5787888523695183 # generates a random number between 0 and 1
| 3) bisect.bisect(A, x), returns an insertion point
| >>> bisect.bisect([1,3,6], 4)
| 2  #4 would be at index 2

# Solution logic

1) with itertools.accumulate(probabilities) we make a list that accumulates
probabilities -> P0, P0+P1, P0+P1+P2 etc.
The interval between two such accumulated values will correspond to the probability 
of each element (intervals like this (0.0, 0.5), (0.5, 0.833) etc., our
accumulated list [0.0, 0.5, 0.833 ..]

2) We use random.random() to generate a value between 0 and 1.
E.g. 0.6 is generated.

3) We use bisect.bisect() to find the index where that value would be in our
accumulated list. 0.6 in [0.0, 0.5, 0.833 ..] would be at index 2, after 0.5,
so we would return values[2] which is ([3,5,7,11]) 7. 

The take away - we choose randomly from <probabilities of the numbers>.

That way we generate one of the n numbers according to the specified probabilities.
(Meaning if the probability of a number ([3,5,7,11]) is high, then the chances
that the randomly generated value (between 0,1) will fall in that range are higher.) ::

    print(bisect.bisect([0.0, 0.5, 0.83, 0.9, 1.0 ], 0.4))  #1
    print(bisect.bisect([0.0, 0.5, 0.83, 0.9, 1.0 ], 0.5))  #2

::

    ### Solution
    import itertools, bisect, random
    def nonuniform_random_number_generation(values, probabilities):
        prefix_sum_of_probabilities = list(itertools.accumulate(probabilities))
        interval_idx = bisect.bisect(prefix_sum_of_probabilities, random.random())
        return values[interval_idx]

    # my rewrite
    def non_uniform_random_num(a, p):
        probabilities = list(itertools.accumulate(p))
        r = random.random()
        index = bisect.bisect_left(probabilities, r)
        return a[index]

    a = [2, 4, 6, 7]
    p = [0.3, 0.2, 0.4, 0.1]
    print(non_uniform_random_num(a, p))

| # My note (off by 1 topic)
| for accumulated sum of probabilities, e.g. [0.5, 0.6, 0.8, 1.0], 
| it seems we will never get the last value in array a, because random will not generate
| num above 1.0, but its OK. Last value is at index 3, random()= e.g. 0.9
| gives bisect_left() index 3. So we will get the last num.

::

    # Let's see what's going on. Adding print calls.
    def nonuniform_random_number_generation(values, probabilities):
        prefix_sum_of_probabilities = list(itertools.accumulate(probabilities))
        print(prefix_sum_of_probabilities)
        interval_idx = bisect.bisect(prefix_sum_of_probabilities, random.random())
        print(interval_idx)
        return values[interval_idx]

    values = [3,5,7,11] 
    probabilities = [9/18, 6/18, 2/18, 1/18]
    print(nonuniform_random_number_generation(values, probabilities))
    # OUT
    # [0.5, 0.8333333333333333, 0.9444444444444444, 1.0]  #yeah, starts from non-zero
    # 1 #index
    # 5

Time O(n) which is time to create the array of intervals.
Space O(n).

54. (LC 36) Valid Sudoku
-------------------------
`36. Valid Sudoku <https://leetcode.com/problems/valid-sudoku/>`_ (Medium)

**Version 1.** If you are given a solved, completely filled, Sudoku.

# Logic.
We check that no row, column, or 3x3 2D subarray contains duplicates. ::

    ### Solution
    import itertools

    def sudoku_ok(line):
        return (len(line) == 9 and sum(line) == sum(set(line)))

    def check_sudoku(grid):
        bad_rows = [row for row in grid if not sudoku_ok(row)]
        grid = list(zip(*grid))
        bad_cols = [col for col in grid if not sudoku_ok(col)]
        squares = []
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
            square = list(itertools.chain(row[j:j+3] for row in grid[i:i+3]))
            square = [n for i in square for n in i]
            squares.append(square)
        bad_squares = [square for square in squares if not sudoku_ok(square)]
        return not (bad_rows or bad_cols or bad_squares)

    sudoku = [[5,3,4,6,7,8,9,1,2],
            [6,7,2,1,9,5,3,4,8],
            [1,9,8,3,4,2,5,6,7],
            [8,5,9,7,6,1,4,2,3],
            [4,2,6,8,5,3,7,9,1], 
            [7,1,3,9,2,4,8,5,6],
            [9,6,1,5,3,7,2,8,4],
            [2,8,7,4,1,9,6,3,5],
            [3,4,5,2,8,6,1,7,8]  # <-- not valid sudoku, two 8s
            ]

    board = [[7, 9, 2, 1, 5, 4, 3, 8, 6], 
                [6, 4, 3, 8, 2, 7, 1, 5, 9],
                [8, 5, 1, 3, 9, 6, 7, 2, 4],
                [2, 6, 5, 9, 7, 3, 8, 4, 1],
                [4, 8, 9, 5, 6, 1, 2, 7, 3],
                [3, 1, 7, 4, 8, 2, 9, 6, 5],
                [1, 3, 6, 7, 4, 8, 5, 9, 2],
                [9, 7, 4, 2, 1, 5, 6, 3, 8],
                [5, 2, 8, 6, 3, 9, 4, 1, 7]]

    print(check_sudoku(sudoku)) #False
    print(check_sudoku(board))  #True

| # *Explained.*
| ``grid = list(zip(*grid))``
| Matrix transpose. To make columns. E.g.:

>>> Z = list(zip((1, 2, 3), (10, 20, 30), (5,7,6)))
>>> Z
[(1, 10, 5), (2, 20, 7), (3, 30, 6)]

::

    for i in range(0, 9, 3):
        for j in range(0, 9, 3):

| Making 3x3 sub-grids.
| Iterate indices with step 3, i.e. 0,3,6. For both rows and columns.

| square = list(itertools.chain(row[j:j+3] for row in grid[i:i+3]))
| square = [n for i in square for n in i]
| chain('ABC', 'DEF') --> A B C D E F
| It really produces tuples [(1,3,5), (5,3,8), (5,9,7)].
| So with the second line we flatten into a list.
| [n for i in square for n in i] - i.e. for tuple in square, for number in tuple, 
| i.e. for each number in tuple in square.

**Version 2.** Check for validity a partially filled board (0 value for blank entries).
[:ref:`10 <ref-label>`]

# *Explained.*
First of all - do not over complicate.

All that the task asks is to check that each row, column and 3x3 cube is valid,
i.e. all values in each of these are numbers 1-9 without duplicates.
So for a row=[1,2,3,4,5,6,'.','.',9] - all we need to check is if all the VISIBLE
filled in values satisfy the rules. NOT if the blank spaces can "potentially" cause 
duplicates (when rows, columns, cubes meet).
So you evaluate the board as of its state "right now".

- How we will identify the 3x3 cubes

| We will give each cube an index. The coordinates would then be: 
|  0 1 2
| 0
| 1
| 2
| Leftmost cube is at [0, 0], last most is at [2,2].
| How do we get these indices, we take the index of a cell, and //3.
| E.g. cell at [8, 8] is in [8//3, 8//3], i.e. in cube [2,2].

- Big O

O(9**2) both time and space, because we iterate through each col, row and also store
the values in hash sets.

::

    ### Solution
    class Solution:
        def isValidSudoku(self, board: List[List[str]]) -> bool:
            cols = collections.defaultdict(set)
            rows = collections.defaultdict(set)
            squares = collections.defaultdict(set)  # key is a tuple= (r /3, c /3)

            for r in range(9):
                for c in range(9):
                    if board[r][c] == ".":
                        continue
                    if (
                        board[r][c] in rows[r]
                        or board[r][c] in cols[c]
                        or board[r][c] in squares[(r // 3, c // 3)]
                    ):
                        return False
                    cols[c].add(board[r][c])
                    rows[r].add(board[r][c])
                    squares[(r // 3, c // 3)].add(board[r][c])

            return True

- collections.defaultdict(set)

We will use hash sets.
So the lines like this of our code:
``cols[c].add(board[r][c])`` Will do stuff like this:

>>> D = collections.defaultdict(set)
>>> D[5].add(30)
>>> D
defaultdict(<class 'set'>, {5: {30}})
>>> D[5].add(31)
>>> D
defaultdict(<class 'set'>, {5: {30, 31}})  #Yes, sets have {}

- ``squares[(r // 3, c // 3)].add(board[r][c])``

What this means is that the key for the 'squares' hash set is a tuple.
So it does this:

>>> D[(1,1)].add(8)
>>> D
defaultdict(<class 'set'>, {5: {30, 31}, (1, 1): {8}})  <=== adds (1, 1): {8}

55. (LC 118) Compute rows in Pascal's triangle
------------------------------------------------
`118. Pascal's Triangle <https://leetcode.com/problems/pascals-triangle/>`_ (Easy)

Example of the first 5 rows in Pascal's triangle::

    # Visualization
    #       [1]
    #      [1, 1]
    #     [1, 2, 1]
    #   [1, 3, 3, 1]
    # [1, 4, 6, 4, 1]

Each row has one more entry than the previous one.
Each entry is the sum of adjacent numbers above.

Write a program which takes as input a nonnegative integer n and returns the first 
n rows of Pascal's triangle.
Hint: Write the given fact as an equation. ::

    ### Solution
    Space and time O(n**2)

    def generate_pascal_triangle(n):
        result = [[1] * (i+1) for i in range(n)]
        for i in range(n):
            for j in range(1, i):
                result[i][j] = result[i-1][j-1] + result[i-1][j]
        return result

    print(generate_pascal_triangle(5))

    ### Less magic version, V1
    def pascal_triangle(n):
        pt = []
        for k in range(1, n + 1):
            pt.append([1] * k)
        for i in range(2, n):
            for j in range(1, i):  # i because 3 items in 3rd row, 4 items in 4th row etc
                pt[i][j] = pt[i - 1][j - 1] + pt[i - 1][j]
                # Don't be tempted to do [j+1] !!
                # item above j to the right is not j+1, it is j, because 1 less item above
        return pt[n - 1]  #here just returning the row n

    ### V2
    def pasca(n):
        tri = []
        for i in range(n):
            row = [1] * (i + 1)
            for j in range(1, i):
                row[j] = tri[i - 1][j - 1] + tri[i - 1][j]
            tri.append(row)
        return tri

### Explained (the magic version)
``result = [[1] * (i+1) for i in range(n)]``
Initializes triangle:
[[1], [1, 1], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1]]

::

        for i in range(n):
            for j in range(1, i):

| i is row, j is item in row.
| Notably, when i=0, i=1, for j in range(1,0), (1,1)
| the loop won't even go anywhere. 
| So we will start assignments to the result when row=i=2, j=1,
| so we are looking at 2 in [1,2,1].

|             ``result[i][j] = result[i-1][j-1] + result[i-1][j]``
| Sets the entry at row i, index j to the sum of the two entries above.
| 1,1
| 1,2,1 
| result[i-1][j-1] + result[i-1][j]
| row above, index j-1 (+) row above, index j
