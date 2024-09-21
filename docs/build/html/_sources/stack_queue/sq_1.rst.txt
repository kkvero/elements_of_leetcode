
Stack and Queue Questions Part 1
================================
173. Print elements of a linked list
---------------------------------------
| Using a stack to print the entries of a singly linked list.
| T O(N), S O(N). 
| (We could also use the technique of reversing a linked list, then T O(N), S O(1).)

::

    def print_linked_list_in_reverse(head):
        nodes = []
        while head:
            nodes.append(head.data)
            head = head.next
        while nodes:
            print(nodes.pop())

174. (LC 155) Min Stack
----------------------------
`155. Min Stack <https://leetcode.com/problems/min-stack/description/>`_
Medium

| We can improve on space: 
| -if a new element pushed to stack is greater then current max, then we don't have to
| add it to minStack (it will never be min).
| -record min and min_count. So if elem=5 and we encounter a second 5, we record count+=1
| for elem 5. E.g.:
| stack = [2,2,1,4]
| minStack = [(2,1), (2,2), (4,1)] 
| #can be tuples, named tuples, or class within our Stack class.

**Solution 1** [:ref:`10 <ref-label>`] [:ref:`7 <ref-label>`] T O(1), S O(N) ::

    class MinStack:
        def __init__(self):
            self.stack = []
            self.minStack = []

        def push(self, val: int) -> None:
            self.stack.append(val)
            val = min(val, self.minStack[-1] if self.minStack else val)
            self.minStack.append(val)

        def pop(self) -> None:
            self.stack.pop()
            self.minStack.pop()

        def top(self) -> int:
            return self.stack[-1]

        def getMin(self) -> int:
            return self.minStack[-1]




















