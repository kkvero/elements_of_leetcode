Knowledge Base Array
====================
Data types, data structures
----------------------------
- What is the Difference Between Data Type and Data Structure?

Data type is one of the forms of a variable to which the value can be assigned 
of a given type only. Data structure is a collection of data of different data types.

Data types don't involve time complexity while data structures involve the concept 
of time complexity.

- Python data types

Numeric data types: int, float, complex
String data types: str
Sequence types: list, tuple, range
Binary types: bytes, bytearray, memoryview
Mapping data type: dict
Boolean type: bool
Set data types: set, frozenset

Array data structure
--------------------
The simplest data structure, which is a contiguous block of memory.

- Time complexities of working with arrays

Retrieving and updating A[i] takes O(1) time.
To delete the element at index i from an array of length n is O(n - i) time compl.

Tips for Arrays 
------------------

|:black_small_square:| Array problems often have simple brute-force solutions that use O(n) space, 
but there are subtler solutions that <use the array itself> to <reduce space> 
complexity to O(1).

|:black_small_square:| Filling an array from the front is slow, so see if it's possible to 
<write values from the back> [:ref:`2 <ref-label>`].

|:black_small_square:| Instead of deleting an entry (which requires moving all entries to its left), 
consider <overwriting> it.

Array types in Python
----------------------

| list type, and tuple type (like list but immutable).
| The key property of a list is that it is dynamically-resized.

Instantiating a list
---------------------

| ``L = [1] + [0] * 10    # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
| ``list(range(100))``
| List comprehension.

Basic operations
------------------

| ``len(A), A.append(x), A.remove(x), A.insert(i, x)`` where i is index
| # check if value is present (O(n))
| a in A
| # difference between 
| B = A, B = list(A)
| # deep and shallow copy
| ``copy.copy(A)`` and ``copy.deepcopy(A)``

Array methods
----------------

| min(A), max(A)
| ``A.reverse()`` (in-place), ``reversed(A)`` (returns an iterator)
| ``A.sort()`` (in-place), ``sorted(A)`` (returns a copy)
| ``del A[i]``, ``del A[i:j]``
| # binary search for sorted lists
| ``bisect.bisect(A, 6)``, ``bisect.bisect_left(A, 6)``, ``bisect.bisect_right(A, 6)``

extend()
-------------

| ``s.extend(t)`` appends t to s (items in t to the end of list s)
| s[:0] = t  prepends (adds items in t to the beginning of s)
| s = [1, 2]
| t = [3, 4]
| s.extend(t)
| print(s) #[1, 2, 3, 4]

Slicing 
--------

| A[i:j:k]
| # last three items [:ref:`2 <ref-label>`]
| A[-3:], or some items from the end A[-3:-1]
| # step
| A[1:5:2] from 1 to 5, every 2nd item
| A[5:1:-2] the same in reverse order (from item 5 to 1, every 2nd but counting from the end)
| B = A[:] shallow copy of A into B
| # rotate a list (like for the Caesar's cipher we rotate the alphabet to make a mask)
| A[k:] + A[:k] 

# Exclude an item

>>> L= [1,2,3,4]   #e.g. to exclude item with i=1
>>> L[:1]+L[1+1:]
[1, 3, 4]

An integer overflow
--------------------
Occurs when an arithmetic operation attempts to create a numeric value that is 
outside of the range that can be represented with a given number of digits
(either higher than the maximum or lower than the minimum representable value).

Because of a possibility of integer overflow, we might sometimes use an array
to represent an integer (i.e. [1,2,3,4] instead of 1234) when doing arithmetic.

Arbitrary-Precision arithmetic
--------------------------------
Also known as "bignum" or simply "long arithmetic" is a set of data structures 
and algorithms which allows to process much greater numbers than can be fit in 
standard data types.




