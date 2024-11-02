Stack and Queue Questions Part 2
================================
181. (LC 456) 132 Pattern
----------------------------
`456. 132 Pattern <https://leetcode.com/problems/132-pattern/description/>`_
Medium

The intuition tells that the problem could be solved either greedy remembering
min, max, mid values or using a stack.
The fact is, it uses both.

| **Logic:**
| E.g.
| nums = [3,1,4,2]
| To the obvious logic of using a decreasing stack:
| stack = [3], stack=[3,1], stack=[4], stack=[4,2]
| We need to record the curMinValue:
| stack = [[3,3]], stack=[[3,1]], stack=[[1,4]], stack=[[1,2]]
| (more precisely: min value before current, i.e. excluding current)
| If we know what was the min value, and the top of the stack is always the greater value,
| then as soon as we meet value greater than min, but less than top of the stack, we got 132 pattern.
 
| **Variables:**
| -stack
| Has values in decreasing order.
| If we meet value greater, we pop stack.
| -greedy var for curMinValue
| Min value we met so far in a variable AND in stack: [curValue, curMinValue]

**Solution 1** [:ref:`10 <ref-label>`] ::

    class Solution:
        def find132pattern(self, nums: List[int]) -> bool:
            stack = [] # pair [num, curLeftMin], mono-decreasing stack
            curMin = nums[0]
            for n in nums:
                while stack and n >= stack[-1][0]:
                    stack.pop()
                if stack and n < stack[-1][0] and n > stack[-1][1]:
                    return True
                stack.append([n, curMin]) 
                curMin = min(n, curMin)
            return False

182. (LC 735) Asteroid Collision
----------------------------------
`735. Asteroid Collision <https://leetcode.com/problems/asteroid-collision/description/>`_
Medium

| **Key:**
| 3 cases when no collision: [-1, -2], [1, 2] AND [-1, 5] (same direction AND opposite directions.)
| So opposite signs don't mean there has to be a collision.

**Solution 1** [:ref:`10 <ref-label>`] ::

    class Solution:
        def asteroidCollision(self, asteroids: List[int]) -> List[int]:
            stack = []
            for a in asteroids:
                while stack and a < 0 and stack[-1] > 0:   #collision only when a is negative
                    diff = a + stack[-1]
                    if diff > 0:     #a loses, it gets destroyed
                        a = 0
                    elif diff < 0:   #a wins
                        stack.pop()
                    else:            #if a==stack[-1]
                        a = 0
                        stack.pop()
                if a:                #if a was not destroyed in prev steps, add it to stack
                    stack.append(a)
            return stack

**My V** (LC accepted 70, 90) ::

    class Solution:
        def asteroidCollision(self, asteroids: List[int]) -> List[int]:
            stk=[]
            for a in asteroids:
                if stk and stk[-1] > 0 and a < 0:     #only if case [+, -] [->,<-]
                    while stk and a and stk[-1] > 0:  #again check [+,-], because stk could be [-2,+1] a=-2 stop after popping +1
                        if a + stk[-1] == 0:
                            stk.pop()
                            a = None
                        elif abs(a) > abs(stk[-1]):
                            stk.pop()
                        elif abs(a) < abs(stk[-1]):
                            a=None
                if a:
                    stk.append(a)
            return stk

| Break into cases:
| 1)Direction
| asteroids = [-1,1]
| <-- --> no collision  (also -1,-1 or 1,1 no collision)
| 
| asteroids = [1,-1]
| --> <--
| COLLISION
| So there is collision <<only when the next asteroid is negative>>, a<0; and left a>0.
| 
| 2)Value
| asteroids = [1,-3]
| a is greater then the left one.
| a wins, asteroids=[-3]   (and if more on the left with lesser value, a will destroy them)
| 
| asteroids = [3,-1]
| a looses, left one stays.
| asteroids = [3]

183. (LC 394) Decode String
------------------------------
`394. Decode String <https://leetcode.com/problems/decode-string/description/>`_
Medium

| **Keys:**
| -Accounting for nested code, 'build substring from within nested chars'.
| E.g. s='a11[c2[dd]]z'
| Append to stack till you meet ']'.
| stack = ['a', '1', '1' '[', 'c', '2', '[', 'd', 'd']
| When ]: start building substring from backwards till you meet '['.
| After '[' pop all digits, that's your multiplier. 
| Substring * multiplier.
| Append substring to stack.
| -So you collect the answer into the stack. Because there also can be
| s='a2[b]z'
 
| +Edge case: encoding digit can have multiple digits.
| =>All the integers in s are in the range [1, 300].

**My V** (LC accepted 93,64, same code also 20,99%) ::

    class Solution:
        def decodeString(self, s: str) -> str:
            stack = []
            for c in s:
                if c != "]":
                    if stack and c.isdigit() == True and stack[-1].isdigit() == True:
                        stack[-1] += c  #if several digits num like s='100[ab]'
                    else:
                        stack.append(c)
                else:
                    char = ""
                    while stack[-1] != "[":
                        char = stack.pop() + char
                    stack.pop()
                    char = int(stack.pop()) * char
                    stack.append(char)
            return "".join(stack)

**My V2** (LC accepted 20, 50%) ::

    class Solution:
        def decodeString(self, s: str) -> str:
            stack = []
            for c in s:
                if c == ']':
                    letters = ''
                    while stack:
                        item = stack.pop()
                        if item != '[':
                            letters = item + letters
                        else:
                            break
                    stack.append(letters * int(stack.pop()))
                else:
                    if stack and c.isdigit() and stack[-1].isdigit():  #>1 digits case
                        stack[-1] += c
                    else:
                        stack.append(c)
            return ''.join(stack)

**Solution 1** [:ref:`10 <ref-label>`] ::

    class Solution:
        def decodeString(self, s: str) -> str:
            stack = []
            for char in s:
                if char is not "]":
                    stack.append(char)
                else:
                    sub_str = ""
                    while stack[-1] is not "[":
                        sub_str = stack.pop() + sub_str
                    stack.pop()

                    multiplier = ""
                    while stack and stack[-1].isdigit():
                        multiplier = stack.pop() + multiplier

                    stack.append(int(multiplier) * sub_str)
            return "".join(stack)

184. (LC 895) Maximum Frequency Stack
------------------------------------------
`895. Maximum Frequency Stack <https://leetcode.com/problems/maximum-frequency-stack/description/>`_
Hard

**My V** (LC accepted 60, 50%) ::

    class FreqStack:
        def __init__(self):
            self.cnt = {}     #{value: freq}
            self.freq = {}    #{freq: [value1, value2..]}
            self.max_freq = 0

        def push(self, val: int) -> None:
            self.cnt[val] = 1 + self.cnt.get(val, 0)
            if self.cnt[val] in self.freq:
                self.freq[self.cnt[val]].append(val)
            else:
                self.freq[self.cnt[val]] = [val]
                self.max_freq +=1

        def pop(self) -> int:
            res = self.freq[self.max_freq].pop()
            self.cnt[res] -=1                 #<--don't forget to decrement cnt of that value
            if len(self.freq[self.max_freq]) == 0:
                self.freq.pop(self.max_freq)
                self.max_freq -= 1
            return res


**Solution 1** [:ref:`10 <ref-label>`] ::

    class FreqStack:
        def __init__(self):
            self.cnt = {}
            self.maxCnt = 0
            self.stacks = {}

        def push(self, val: int) -> None:
            valCnt = 1 + self.cnt.get(val, 0)
            self.cnt[val] = valCnt
            if valCnt > self.maxCnt:
                self.maxCnt = valCnt
                self.stacks[valCnt] = []
            self.stacks[valCnt].append(val)

        def pop(self) -> int:
            res = self.stacks[self.maxCnt].pop()
            self.cnt[res] -= 1
            if not self.stacks[self.maxCnt]:
                self.maxCnt -= 1
            return res

| **Keys**
| -Internal data structures
|     self.cnt = {}
| To keep track of the count for Each value. E.g.:
| cnt = {5:3, 4:2, 3:2, 2:1}
 
|     self.maxCnt = 0
| Keep track of the max count Overall.
| E.g. 3 here.
 
|     self.stacks = {}
| Making groups of how many of each value currently.
| obj.push(5)
| self.stacks = {
|     1 : [5]
| }
 
| obj.push(4)
| self.stacks = {
|     1 : [5,4]
| }
 
| obj.push(5)
| self.stacks = {
|     1 : [5,4],
|     2 : [5]
| }

NOTE: each new value gets added to the end of the list of the group.
So we can pop from a group and it will be the most recent value (relative to other 
items in the group).

...After pushing all values::

    self.stacks = {
        1 : [5,4,3,2],
        2 : [5,4,3],
        3 : [5]
    }

| -Pushing
| Just push from the group with the highest key.

185. (LC 901) Online Stock Span
---------------------------------
`901. Online Stock Span <https://leetcode.com/problems/online-stock-span/description/>`_
Medium

| **Keys**
| - Stack, store 2 values [price, span]
| - With each new price pop lesser values from stack

**Solution 1** [:ref:`10 <ref-label>`] ::

    class StockSpanner:
        def __init__(self):
            self.stack = []  # pair: (price, span)

        def next(self, price: int) -> int:
            span = 1
            while self.stack and self.stack[-1][0] <= price:
                span += self.stack[-1][1]
                self.stack.pop()
            self.stack.append((price, span))
            return span

**My V** (LC accepted 98, 95) ::

    class StockSpanner:
        def __init__(self):
            self.stack = []  #[[price, span], ..]

        def next(self, price: int) -> int:
            span = 1
            while self.stack and self.stack[-1][0] <= price:
                span += self.stack[-1][1]
                self.stack.pop()
            self.stack.append([price, span])
            return span

| Monotonic decreasing stack
| -Compute the span for each price as you are given that price.
| -Store as (price, span) pair in a stack.
| -Recognize that you can compare current price with previous. If cur>prev,
| lookup the span of prev, you can ignore span number of prices after that. 
| And keep doing this cur>prev check till you meet prev>cur.
| prices = [100,80,60,70,60,75,85]
| stack = [(100,1),(80,1),(60,1),(70,2),(60,1),(75,4),(85,6)]
| -More so, if cur>prev, pop prev from stack
| Cur price=70 

::

    # prices = [100,80,60,70 ..]
    #                     ^

| 70>60, pop (60,1) from stack. Before popping, set span of 70 = 1 + 2 (default + span of 60).
| Append 70 and its span.
| stack = [(100,1),(80,1),>(60,1)<,(70,2)]

We pop because having [60, 70], we will never get past the bigger value of 70.

186. Implement queue
--------------------------
Implement the basic queue API: enqueue, dequeue.

| **deque and popleft()**
| We will use Python collections.deque 
| (doubly linked list)

::

    import collections
    class Queue:
        def __init__(self):
            self._data = collections.deque()

        def enqueue(self, v):
            self._data.append(v)

        def dequeue(self):
            return self._data.popleft()

187. (LC 102) Binary Tree Level Order Traversal
--------------------------------------------------
`102. Binary Tree Level Order Traversal <https://leetcode.com/problems/binary-tree-level-order-traversal/description/>`_
Medium

**My V** (LC accepted 70, 50) ::

    class Solution:
        def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
            q = collections.deque()
            res = []
            if root:
                q.append(root)
            while q:
                res.append([])
                for i in range(len(q)):
                    node = q.popleft()
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
                    res[-1].append(node.val)
            return res


| **Solution 1** [:ref:`2 <ref-label>`]
| -List comprehensions
| Here we empty the queue each time. To be precise, we REPLACE it with children.
| We don't use queue directly. We manage the task of getting the values in order by using list comprehensions.
| (LC accepted 70,90%)

::

    def binary_tree_depth_order(tree):  # tree=root
        res = []
        if not tree:
            return res
        curr_depth_nodes = [tree]  # initially one value, [3]
        while curr_depth_nodes:
            res.append(
                [curr.val for curr in curr_depth_nodes]
            )  # appends from front to end of queue
            curr_depth_nodes = [     #REPLACE curr values with their children
                child
                for curr in curr_depth_nodes
                for child in (curr.left, curr.right)
                if child
            ]
        return res

| **Solution 2** [:ref:`10 <ref-label>`]
| -collections.deque
| (Here we don't empty the queue each level, we classically use queue/deque:
| leftpop() values, and append() children.)

::

    class Solution:
        def levelOrder(self, root: TreeNode) -> List[List[int]]:
            res = []
            q = collections.deque()
            if root:
                q.append(root)

            while q:
                val = []

                for i in range(len(q)):
                    node = q.popleft()
                    val.append(node.val)
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
                res.append(val)
            return res

| **BFS**
| -We use queue 'q' using collections.deque

::

    # q = [pop, add    ]
    #      ^    --> adding to the end
    #     popping from front 
 
| -And sublist where we collect values for one tree level.
| -How we access children, just node.left, node.right
 
| **Procedure**
| root = [3,9,20,null,null,15,7]

::

    #  3
    # 9  20
    #   15 7

| Put tree root to q.
| q=[3]
| popleft to sublist. We define when to stop by the initial len(q). Append sublist to ans.
| -As we pop from q, we add to q the children of current values. <===
| [9, 20]
| sublist=[3]

188. (LC 622) Design Circular Queue
----------------------------------------
`622. Design Circular Queue <https://leetcode.com/problems/design-circular-queue/description/>`_
Medium

| **Task understanding:**
| You wouldn't overwrite when q is full.
| q=[1,2,3] enqueue(4) Does not overwrite 1. Return False. Wait for the dequeue() operation 
| before being able to enqueue.
 
| **Keys (array):**
| -Array. One pointer front. Capacity(possible), size(actual so far). 
| We don't remove values, just move the front pointer.
| (Get everything else: rear, where to enQueue using front pointer, size, % capacity)
| rear = q[(front + size-1) % capacity]
| enqueue_index = (self.front + self.size) % self.capacity
 
| **Ways to implement:**
| - using an array
| - using a linked list
|  -doubly linked
|  -singly linked 
 
| Probable issue with linked list implementation.
| We keep creating new nodes.
| But when using array, we just overwrite the same spot at index.

ARRAY ::

    class MyCircularQueue:
        def __init__(self, k: int):
            self.q = [0] * k
            self.front = 0
            self.size = 0
            self.capacity = k

        def enQueue(self, value: int) -> bool:
            if self.isFull():
                return False
            idx = (self.front + self.size) % self.capacity
            self.q[idx] = value
            self.size += 1
            return True

        def deQueue(self) -> bool:
            if self.isEmpty():
                return False
            self.front = (self.front + 1) % self.capacity
            self.size -= 1
            return True

        def Front(self) -> int:
            return -1 if self.isEmpty() else self.q[self.front]

        def Rear(self) -> int:
            if self.isEmpty():
                return -1
            idx = (self.front + self.size - 1) % self.capacity
            return self.q[idx]

        def isEmpty(self) -> bool:
            return self.size == 0

        def isFull(self) -> bool:
            return self.size == self.capacity

| **Details:**
| - we initiate array of size capacity.
| Capacity=3
| q = [0,0,0]
| - enQueue(1), enQueue(2), enQueue(3)
| q = [1,2,3]
| front stays 0.
| - deQueue()
| We don't remove the actual value.
| We move front+=1
| Decrement size-=1
| Return True (operation successful).
| - enQueue(4)
| q = [4,2,3]
 
| How we get the index
|     ``idx = (self.front + self.size) % self.capacity``
| Basically, index always = front. It's just that to prevent moving beyond q size 
| we perform %capacity, and move in a circle.
| Move front+=1 each time we deQueue(). Means 1 space has been freed up.

189. (LC 232) Implement Queue using Stacks
----------------------------------------------
`232. Implement Queue using Stacks <https://leetcode.com/problems/implement-queue-using-stacks/description/>`_
Easy

O(m) because each elem is pushed no more than twice. ::

    # State
    # push 1,2,3
    # stk1  [1,2,3]  
    # stk2  []       
    
    # pop
    # stk1  []        []
    # stk2  [3,2,1]   [3,2]
    # ->1
    
    # push 4,5
    # stk1  [4,5]
    # stk2  [3,2] 

**Solution 1** [:ref:`7 <ref-label>`] ::

    class MyQueue:
        def __init__(self):
            self.stk1 = []
            self.stk2 = []

        def push(self, x: int) -> None:
            self.stk1.append(x)

        def pop(self) -> int:
            self.move()
            return self.stk2.pop()

        def peek(self) -> int:
            self.move()
            return self.stk2[-1]

        def empty(self) -> bool:
            return not self.stk1 and not self.stk2

        def move(self):
            if not self.stk2:
                while self.stk1:
                    self.stk2.append(self.stk1.pop())

190. Implement a Queue with max API
---------------------------------------

| **Key:**
| When max will never return an element, regardless of future updates.
| Two deques.
 
| Each dequeue operation O(n). 
| A single enqueue may need several operations.
| Amortized T complexity of n enqueues, dequeues is O(n).

::

    class QueueWithMax:
        def __init__(self):
            self.q = collections.deque()
            self.qmax = collections.deque()

        def enqueue(self, x):
            self.q.append(x)
            while self.qmax and self.qmax[-1] < x:
                self.qmax.pop()
            self.qmax.append(x)

        def dequeue(self):
            if self.q:
                res = self.q.popleft()
                if res == self.qmax[0]:    #leftpop() from maxq only if ==
                    self.qmax.popleft()
                return res
            raise IndexError('empty queue')
        
        def return_max(self):
            if self.qmax:
                return self.qmax[0]
            raise IndexError('empty queue')

| **Trace table**
| (We pop always from left)

::

    #       enq(1) enq(2) enq(3)  deq() enq(4)
    # q     [1]    [1,3]  [1,3,6] [3,6] [3,6,4]
    # qmax  [1]    [X,3]* [X,X,6] [6]** [6]
    
    # * if cur value is greater, pop from qmax
    # ** if value we dequeue is smaller, don't touch qmax

| My note:
| but what if values are not unique?
| [1,6,3,6]
| qmax=[X,6,6] #We should append another 6.

**My V** using 2 stacks ::

    class Q:
        def __init__(self):
            self.stack1 = []
            self.stack2 = []
            self.maxes = []

        def enqueue(self, data):
            self.stack1.append(data)

        def dequeue(self):
            if self.is_empty():
                return None
            if not self.stack2:
                self.move()
            value = self.stack2.pop()
            if value == self.maxes[-1]:
                self.maxes.pop()
            return value

        def return_max(self):
            if self.is_empty():
                return None
            if not self.maxes:
                self.move()
            return self.maxes[-1]

        def move(self):
            while self.stack1:
                value = self.stack1.pop()
                if not self.maxes or self.maxes[-1] <= value:
                    self.maxes.append(value)
                self.stack2.append(value)

        def is_empty(self):
            if not self.stack1 and not self.stack2:
                return True
            return False

    q = Q()
    q.enqueue(1)
    q.enqueue(3)
    q.enqueue(5)
    q.enqueue(2)
    q.enqueue(5)
    print(q.stack1)
    print(q.return_max())
    print(q.dequeue())
    print(q.dequeue())
    print(q.dequeue())
    print(q.return_max())
    # [1, 3, 5, 2, 5]
    # 5 #max
    # 1
    # 3
    # 5
    # 5 #max