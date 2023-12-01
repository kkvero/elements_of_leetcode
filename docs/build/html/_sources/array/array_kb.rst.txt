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

Array
------
The simplest data structure, which is a contiguous block of memory.

- Time complexities of working with arrays

Retrieving and updating A[i] takes O(1) time.
To delete the element at index i from an array of length n is O(n - i) time compl.

- Tips for Arrays

|:black_small_square:| Array problems often have simple brute-force solutions that use O(n) space, 
but there are subtler solutions that <use the array itself> to <reduce space> 
complexity to O(1).

|:black_small_square:| Filling an array from the front is slow, so see if it's possible to 
<write values from the back>.

|:black_small_square:| Instead of deleting an entry (which requires moving all entries to its left), 
consider <overwriting> it.
