Array Questions Part 1
======================
37. Flatten a nested list
--------------------------
| (convert 2D list to 1D)
| Using list comprehension with multiple levels of looping.

>>> vec = [[1,2,3], [4,5,6], [7,8,9]]
>>> [num for elem in vec for num in elem]
[1, 2, 3, 4, 5, 6, 7, 8, 9]

::

    # My version 2
    def flatten(L):
        return [y for x in L for y in x]

    # My version 1
    def flatten(a):
        return [n[i] for n in a for i in range(len(n))]

>>> L1=[[1,2],[4,5]]
>>> L2=[x for y in L1 for x in y]
>>> L2
[1, 2, 4, 5]

| **Key:**
| Don't read it in one go: x for y...
| Rather:
| part1) ``x`` - what we to be returned
| part2) how to calculate it - ``for y in L1 for x in y``

**Stdlib**

>>> import itertools
>>> L3=itertools.chain(*L1)
>>> list(L3)
[1, 2, 4, 5]
>>> L4=itertools.chain.from_iterable(L1)
>>> list(L4)
[1, 2, 4, 5]

38. Reorder, evens first
--------------------------
Your input is an array of integers, and you have to reorder its entries so that 
the even entries appear first.
You are required not to allocate extra memory. i.e. space O(1).

**Solution**

When working with arrays, take advantage of the fact that you can operate efficiently
on both ends. 

[Even, Unclassified, Odd]

For this problem, we can partition the array into three subarrays: Even, Unclassified,
and Odd, appearing in that order. Initially Even and Odd are empty, and Unclassified is the entire
array. We iterate through Unclassified, moving its elements to the boundaries of the Even and Odd
subarrays via swaps, thereby expanding Even and Odd, and shrinking Unclassified. ::

    def even_odd(A) :
        next_even, next_odd = 0, len(A) - 1  #starting indices for Even, Odd parts
        while next_even < next_odd:
            if A[next_even] % 2 == 0:
                next_even += 1
            else :
                A[next_even], A[next_odd] = A[next_odd], A[next_even]
                next_odd -= 1

39. (LC 75) Dutch national flag 
--------------------------------

| `75. Sort Colors <https://leetcode.com/problems/sort-colors/>`_
| Medium

::

    ### Solution my version
    def sort_flag(a):
        p = 0
        r = 0
        b = len(a) - 1
        while p < b:           #**
            if a[p] == 0:
                a[p], a[r] = a[r], a[p]
                r += 1
                p += 1
            elif a[p] == 1:
                p += 1
            elif a[p] == 2:
                a[p], a[b] = a[b], a[p]
                b -= 1
        return a

    #** had to make p <= b for array [1, 1, 0, 0, 2, 0] otherwise returns [0, 0, 1, 1, 0, 2]

    a = [0, 0, 1, 1, 0, 2, 0, 2, 2, 1, 0]
    print(sort_flag(a)) #[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


| # Explained
| p - is the moving pivot
| r - red pivot, b - black pivot.
| Note that this is not much different than sorting array in two halves.
| When we meet red (0), we swap it with r pivot (leftmost pivot), move +1 r and p.
| When we meet black (2), swap p with b, p+1, b-1.
| The trick here is when we meet white (1), >we leave it there<, just p+1.
| It's also OK when we have [0,1,1,0..]: a[p=3]=0 we swap as usual [0,>1<,1,>0<..]
| and get [0,0,1,1..] (i.e. 1s in te middle are preserved.)

::

    # Solution V2
    class Solution:
        def sortColors(self, nums: List[int]) -> None:
            i, j, k = -1, len(nums), 0
            while k < j:
                if nums[k] == 0:
                    i += 1
                    nums[i], nums[k] = nums[k], nums[i]
                    k += 1
                elif nums[k] == 2:
                    j -= 1
                    nums[j], nums[k] = nums[k], nums[j]
                else:
                    k += 1

| **Dutch national flag** [:ref:`2 <ref-label>`]
| ==> Practicing working with <subarrays>

The quicksort algorithm uses partitioning and recursion.
The first partition, given a pivot reorders the array in such a way that elements
less than pivot appear first, then elements less than pivot.
Quicksort has large run times when arrays have many duplicates.

The dutch flag partitioning is supposed to make this more effective by ordering
in this way: elements less than pivot, elements equal to pivot, elem greater than pivot.

So you see, 3 parts, like in a three-colors flag, like a Dutch flag.

| --not completely sorted
| Note, that such a partitioning does not order the array completely!
| The point is to achieve this state: [elems < pivot, elems=pivot, elems>pivot] in O(n).
| # So given::

    A = [1,0,3,2,1,2]
    dutch_flag_partition(2, A)  #pivot = 3
    print(A)

    A2 = [1,0,3,2,1,2,5,1]
    dutch_flag_partition(3, A2) #pivot = 2
    print(A2)

    A3 = [1,0,3,2,1,2]
    dutch_flag_partition(0, A3) #pivot = 1
    print(A3)

    # Results valid partitions:
    [1, 0, 2, 1, 2, 3]        #pivot = 3
    [1, 0, 1, 1, 2, 2, 5, 3]  #pivot = 2 #is everything less than pivot on the left?->Yes
    [0, 1, 1, 2, 2, 3]

**Task**

Write a program that takes an array A and an index i, and rearranges the elems
such that: [first come elems < pivot, followed by elems=pivot, and last go elems>pivot]

So it is the partition step in quicksort.

| **Solution**
| Time O(n), space O(1) 

::

    # pi - pivot index
    # s, e, l = smaller, equal, larger
    def dutch_flag_partition(pi, A):
        pivot = A[pi]
        s, e, l = 0, 0, len(A)
        while e < l:
            if A[e] < pivot:
                A[s], A[e] = A[e], A[s]
                s, e = s+1, e+1
            elif A[e] == pivot:
                e += 1
            else: # A[e] > pivot
                l -= 1
                A[e], A[l] = A[l], A[e]

| We maintain 4 subarrays:
| # bottom group: A[:smallerJ.
| # middle gtoup: A[smaller:equal].
| # unclassified group: A[equal: larger] .
| # top group: A[larger:] .
|     while e < l:
| # Keep iterating as long as there is an unclassified element
|         if A[e] < pivot:
| # A[equal] is the inconing unclassjfied element


40. (LC 66) Increment an arbitrary-precision integer
--------------------------------------------------------
| `66. Plus One <https://leetcode.com/problems/plus-one/>`_
| Easy 

::

    ### Solution 1
    class Solution:
        def plusOne(self, digits: List[int]) -> List[int]:
            n = len(digits)
            for i in range(n - 1, -1, -1):
                digits[i] += 1
                digits[i] %= 10
                if digits[i] != 0:
                    return digits
            return [1] + digits

- Increment an arbitrary-precision integer [:ref:`2 <ref-label>`]

Your program takes as input an array of digitsinteger that represents (it encodes)
a nonnegative decimal integer D. Your program updates the array to represent the 
integer D+1.
Example: Input [1,2,9] Output [1,3,0]

- Brute-force solution

Convert array to the corresponding integer (i.e. 129). Add 1 (129+1=130), convert
back to an array of digits.
((However when implemented in a language that has finite-precision arithmetic 
(imposes a limit on the range of values an integer type can take), this approach will 
fail on inputs that encode integers outside of that range.)) ::

    ### my version 2
    def plus_one(a):
        a[-1] += 1
        if (a[-1] + 1) // 10 == 0:
            return a
        a = [0] + a
        carry = 0
        for i in reversed(range(len(a))):
            a[i] += carry
            carry = a[i] // 10
            a[i] = a[i] % 10
        if a[0] == 0:
            return a[1:]
        return a

    a = [1, 2, 9]
    a2 = [9, 9, 9]
    print(plus_one(a))
    print(plus_one(a2))
    # [1, 3, 0]
    # [1, 0, 0, 0]

    ### my version 1
    def plus_one(a):
        a = [0] + a
        for i, n in reversed(list(enumerate(a))):
            if (n + 1) > 9:
                a[i] = 0
            else:
                a[i] = n + 1
                break
        if a[0] == 0:
            return a[1:]
        else:
            return a

| **Solution 2**
| O(n), n is the length of A.
| Operate directly on the array.
| Grade school, add starting from the end (least significant digit), and propagate carries. 

::

    def plus_one(A):
        A[-1] += 1
        for i in reversed(range(1, len(A))):
            if A[i] != 10:
                break
            A[i] = 0
            A[i - 1] += 1
        if A[0] == 10:
            # There is a carry-out, we need one more digit.
            # Update first item in A to 1, and append a 0 at the end
            A[0] = 1
            A.append(0)
        return A

    A22 = [1,2,9]
    print(plus_one(A22))  #[1, 3, 0]

41. Multiply two arbitrary-precision integers
----------------------------------------------
Certain applications require arbitrary precision arithmetic. 
(Arbitrary precision arithmetic - algorithms which allow to process much greater
numbers than can be fit in standard data types)

One way to achieve this is to use arrays to represent integers.
E.g. [3,4,5,4,6], [-7,5,3,2] (Note negative integers too.)

Write a program that takes two arrays representing integers, and returns an 
integer (in form of array) representing their product.

| *Logic* [:ref:`2 <ref-label>`]
| Hint: Use arrays to simulate the grade-school multiplication algorithm.
| The possibility of overflow precludes us from converting to the integer type.

Using grade school multiplication logic, we multiply the first number by each 
digit of the second, and then adding all the resulting terms.

From a space perspective, it is better to incrementally add the terms rather than 
compute all of them individually and then add them up.

**Solution**
O(nm).
(There are m partial products, each with at most n + 1 digits. We perform O(1) operations on each
digit in each partial product, so the time complexity is O(nm).) ::

    def multiply(num1, num2):
        sign = -1 if (num1[0] < 0) ^ (num2[0] < 0) else 1     #**1
        num1[0], num2[0] = abs(num1[0]), abs(num2[0])

        result = [0] * (len(num1) + len(num2))              #**2
        for i in reversed(range(len(num1))):
            for j in reversed(range(len(num2))):
                result[i + j + 1] += num1[i] * num2[j]       #**3
                result[i + j] += result[i + j + 1] // 10
                result[i + j + 1] %= 10
        # Remove the leading zeros.
        result = result[next((i for i, x in enumerate(result)      #**4
                            if x != 0), len(result)):] or [0]
        return [sign * result[0]] + result[1:]

    a1 = [5, 6]
    a2 = [-7, 5]
    print(multiply(a1, a2))  # [-4, 2, 0, 0]

#**1 If one of the comparisons evaluate to True, i.e. 1, then their xor is 1,
then we apply if (i.e. -1), else 1.

#**2 The number of digits required for the product is at most n + m for n and m 
digit operands, so we use an array of size n + m for lhe result.
(Note, 'at most' that size, so we might end up with a leading 0, i.e. [0, x, y, z])

#**3 We loop through indexes in reversed order.
NOTE, its array indexing, so the indexes are [0,1,2,3,etc]

| E.g. if our two numbers are [5,6], [7,5], 
| first loop, i=1, j=1 (reversed(range(len(2))) is 1)
| result = [0,0,0,0]
| result[i + j + 1] += num1[i] * num2[j]      # res[1+1+1] += 6*5=30 ([0,0,0,30])
| result[i + j] += result[i + j + 1] // 10   
| # res[1+1] += res[1+1+1] //10 which is 3, this is the carry-in, [0,0,3,30]
| result[i + j + 1] %= 10   
| #res[1+1+1] = res[1+1+1] % 10 which is 0, remainder [0,0,3,0]

#**4 ::

    result = result[next((i for i, x in enumerate(result)
                          if x != 0), len(result)):] or [0]

| 1)With next manual iterator through a generator we get:
| the first index which value is not 0.
| 2)We also include the default of len(result). 
| next syntax - next(iterator, default)
| default is the value that will be returned if the iterator is exhausted,
| this is for the case when all values in result are 0,
| e.g. [0,0,0,0] len(result)=4, result[4:] = []
| Otherwise iterator will throw an error.

| -> When testing:
| without len(result): - he does only one __next__ call
| with len(result): extracts all

::

    ### My simplified V
    # (Not accounting for sign, without removing leading zeros)
    def f(a1, a2):
        ans = [0] * (len(a1) + len(a2))
        loop = len(ans)
        for i in range(len(a2) - 1, -1, -1):
            loop -= 1
            index = loop
            for j in range(len(a1) - 1, -1, -1):
                prod = a2[i] * a1[j]
                prod = ans[index] + prod
                carry = prod // 10
                ans[index - 1] += carry
                ans[index] = prod % 10  # not +=
                index -= 1
        return ans

    n1 = [3, 4, 5, 4, 6]
    n2 = [7, 5, 3, 2]
    print(f(n1, n2)) #[2, 6, 0, 2, 0, 0, 4, 7, 2]

42. (LC 55) Jump Game (Advancing through an array)
----------------------------------------------------
| `55. Jump Game <https://leetcode.com/problems/jump-game/>`_
| Medium

| **Task note**
| Each element in the array represents your MAXIMUM JUMP length.
| So in [3,0,8,2,0,0,1] you don't have to jump from i=0 to i=3, you can jump to i=2 (with 8).

| **Key**
| - Start looking at nums[i] values in reverse.
| [2,3,1,1,<4>]  #<We imagine index 4 to be our end goal.>
|  0,1,2,3,4     #i
| End goal = index 4
| Can we get from i=3 (1 jump) to goal i=4, yes. 
| Then we can move our goal post to this position. Goal = index 3.
| [2,3,1,<1>,4]
| 
| Can we get from i=2 (1 jump) to goal i=3, yes. Move goal.
| [2,3,<1>,1,4]
| ...

Greedy, O(n). ::

    ### Solution 1 (neetcode)
    class Solution:
        def canJump(self, nums: List[int]) -> bool:
            goal = len(nums) - 1
            for i in range(len(nums) - 2, -1, -1):
                if i + nums[i] >= goal:
                    goal = i
            return goal == 0

    ### My V (LC accepted 60, 60)
    class Solution:
        def canJump(self, nums: List[int]) -> bool:
            goal = len(nums) - 1
            for i in reversed(list(range(len(nums)))):
                if i + nums[i] >= goal:
                    goal = i
            return goal == 0

43. (LC 26) Delete duplicates from a sorted array
----------------------------------------------------
| `26. Remove Duplicates from Sorted Array <https://leetcode.com/problems/remove-duplicates-from-sorted-array/>`_
| Easy

| **Keys:**
| -One write index + for num in nums iteration.
| -Backwards lookup at write index-1

::

    #Illustration
    #[0, 0, 0, 1]
    #    wi    n

| As soon as n != nums[wi-1]: we can write to wi, shift wi+=1
| Else, continue for num in nums iteration.

::

    ### Solution 1
    class Solution:
        def removeDuplicates(self, nums: List[int]) -> int:
            k = 0
            for x in nums:
                if k == 0 or x != nums[k - 1]:
                    nums[k] = x
                    k += 1
            return k

| **V2** (My V)
| 2 pointers + write pointer
 
| We start with a not valid write pointer (wp=-1).
| The code below is a refactoring of the full logic of:
| 4 cases:
| -values at p1, p2 are different
|  -wp not yet valid (then keep moving p1,p2)
|  -wp valid, write to it (move p1, p2)
| -values at p1, p2 are the same
|  -wp not valid, make wp valid (wp=p2), move p1,p2
|  -wp valid, only move p1,p2

::

    ### My V (LC accepted 70, 98)
    class Solution:
        def removeDuplicates(self, nums: List[int]) -> int:
            if len(nums) < 2:
                return len(nums)
            p1,p2,wp = 0,1,-1
            while p2 < len(nums):
                if nums[p1] != nums[p2]:
                    if wp > 0:
                        nums[wp] = nums[p2]
                        wp += 1
                else:
                    if wp < 0:
                        wp = p2
                p1 += 1
                p2 += 1
            return wp if wp > 0 else p2

44. (LC 121) Buy and sell a stock once
----------------------------------------
| `121. Best Time to Buy and Sell Stock <https://leetcode.com/problems/best-time-to-buy-and-sell-stock/>`_
| Easy

Write a program that takes an array denoting the daily stock price, and returns the maximum profit
that could be made by buying and then selling one share of that stock. There is no need to buy if
no profit is possible.

*Logic*.
E.g. consider a sequence of stock prices [310, 315, 275, 295, 260, 270, 290, 230, 255, 250].
(Here buy at 260, sell at 290, max profit is 30.) Note that we cannot just buy at lowest 
price and sell at highest. 

The maximum profit that can be made by selling on each specific day is 
the difference of the current price and the minimum seen so far. 

| *Algorithm*
| 0)So we keep track of the minimum_price and of the maximum profit.
| 1)We check what profit we could make if we sell today (i.e. price_today - minimum_price)
| 2)See if that profit is gratest, by greadily comparing it with the previous max_profit
| 3)Finally we check if the price_today can be our new minimum_price.

::

    def buy_and_sell_stock_once(prices):
        min_price, max_profit = float('inf'), 0.0
        for price in prices:
            max_profit_today = price - min_price
            max_profit = max(max_profit, max_profit_today)
            min_price = min(min_price, price)
        return max_profit

    stock = [310, 315, 275, 295, 260, 270, 290, 230, 255, 250]
    print(buy_and_sell_stock_once(stock))  #30

    # OR
    def max_profit(a):
        price_min = a[0]
        profit = 0
        for price in a:
            profit = max(profit, price - price_min)
            price_min = min(price, price_min)
        return profit

45. (LC 123)  Buy and sell a stock twice
------------------------------------------
Write a program that computes the maximum profit that can be made by buying and selling a share
at most twice. The second buy must be made on another date after the first sale.

We will use O(n) twice, ending up with O(n**2)

`123. Best Time to Buy and Sell Stock III <https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/>`_

| **Logic**
| A = [12,11,13,9,12,8,14,13,15]
| 1# We record the best solution (highest profit) for A[0,j] (just as we did with 
| buying and selling once).
| F = [0,0,2,2,3,3,6,6,7] 
| (calculating profit: 12-12=0, 12-11=0, 13-11=2 etc )
| We look at the current price and the minimum price so far.
| 2# Then we do a reverse iteration, compute highest profit for a single buy and sell
| for A[j, n-1] (j between 1 and (len A - 1), inclusive).
| We then look at the current price and the highest price so far.
| S = [7,7,7,7,7,7,2,2,0]
| (15-15=0, 13-15=2, 13-14=1 but we stick with the previous highest which is 2, 8-15=7)

I suppose: with iteration forward we try to find the <earliest> highest profit.
With reverse iteration we find the <latest> biggest profit. Because the problem 
states that the second buy-sell can happen only <after> the first sell.

3# At last we combine the results for possible profits from the forward and reverse
iterations.
But again, because the condition is that "the second buy must happen on another 
date after the first sell", we combine S[i] with F[i-1] 
(current day profit + previous day profit). 

F = [0,0,2,2,3,3,6,6,7] 
S = [7,7,7,7,7,7,2,2,0]

M = S[i] + F[i-1], NOTE where F[-1] is taken to be 0. So we sort of then have
F = [0,0,0,2, etc]

M = [7,7,7,9,9,10,5,8,6]
(7+0=7, 7+0=7, 7+0=7, 7+2=9 etc)
We just look for the max in this list, i.e. 10. ::

    ### Solution
    def buy_and_sell_stock_twice(prices):
        max_profit, min_price = 0.0, float('inf')
        # Because we want to make an actual list of profits, initiate
        first_profits = [0] * len(prices) 
        # Forward phase, for each day max profit if we sell on that day.
        for i, price in enumerate(prices):
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
            first_profits[i] = max_profit
        
        # Backward phase. For each day calc max profit if we sell on that day
        max_price = float('-inf')
        for i, price in reversed(list(enumerate(prices[1:], 1))):
            max_price = max(max_price, price)
            # We combined 2nd and 3rd steps (didn't make a list of profits in reverse)
            max_profit = max(
                max_profit,
                max_price - price + first_profits[i-1]
            )
        return max_profit

    A = [12,11,13,9,12,8,14,13,15]
    print(buy_and_sell_stock_twice(A)) # 10

    ### Solution 2 (github 2nd source)
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            f1, f2, f3, f4 = -prices[0], 0, -prices[0], 0
            for price in prices[1:]:
                f1 = max(f1, -price)
                f2 = max(f2, f1 + price)
                f3 = max(f3, f2 - price)
                f4 = max(f4, f3 + price)
            return f4

- enumerate(iterable, start=0)

What do we achieve with reversed(list(enumerate(A[1:], 1)))
For the reverse iteration we will not be calculating profit for the first day,
so we take A[1:].
Also we want to start our index count at 1, not at 0. ::

    A = [12,11,13,9,12,8,14,13,15]
    for i, n in reversed(list(enumerate(A))):
        print(i, n)

    for i, n in reversed(list(enumerate(A[1:], 1))):
        print(i, n)
    # 8 15
    # 7 13
    # 6 14
    # 5 8
    # 4 12
    # 3 9
    # 2 13
    # 1 11
    # 0 12
    
    # 8 15
    # 7 13
    # 6 14
    # 5 8
    # 4 12
    # 3 9
    # 2 13
    # 1 11


