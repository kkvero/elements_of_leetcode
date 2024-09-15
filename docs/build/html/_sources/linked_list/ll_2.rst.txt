
Linked Lists Questions Part 2
================================
161. (LC 19) Remove Nth Node From End of List
-------------------------------------------------
`19. Remove Nth Node From End of List <https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/>`_
Medium

| Other names: Remove the Nth last element from a list
| **Keys** (final conclusion):
| -Two pointers. Start each at dummy.
| Move p1 for i in range(n).
| -Then move both p1,p2 while p1.next.
| (So we advance p1 n-steps forward before moving p1, p2 in unison until p1 reaches list last node.)

::

    ### My V (LC accepted 20, 98)
    class Solution:
        def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
            dummy=ListNode()
            dummy.next=head
            p1=p2=dummy
            for _ in range(n):
                p1=p1.next
            while p1.next:
                p1=p1.next
                p2=p2.next
            p2.next=p2.next.next
            return dummy.next

::

    ### Illustration 1 (general)
    # We use the principle of two iterators.
    # Dummy X X X X X X X
    #         ^

| To get to 3rd X from the end.
| -i1 (iterator 1) from head 3 times.
| -i2 from <dummy head> till i1 runs off the list.
| -i2 is now at the node before the marked X

::

    # X  X  X  X  X  X  X
    #          i1 ^
     
    # X  X  X  X  X  X  X
    #          i2 ^       i1

| NOTE:
| I1 starts at the head.
| I2 starts at the dummy_head (to get the node before the node in question)

::

    ### Illustration 2 (details)
    # E.g. n=3
    # #1
    # L 0->A->B->C->D->E->F->G->H
    #   DH f             Nth
    # -Create dummy head
    # -first=dummy.next (=list head)
     
    # #2
    # Loop n times
    # 0->A->B->C->D->E->F->G->H
    #    f  f1 f2 f3    N
     
    # #3
    # second=dummy_head
    # -move second till first doesn't run off the list
    # 0->A->B->C->D->E->F->G->H
    # s              s5 N       f3-5
    # Here it takes 5 steps.
    # Now second is just before Nth node.
    # We point second PAST the Nth: 
    # second.next = second.next.next

| **Solution 1** 
| Time O(n), Space O(1)

::

    class Solution:
        def removeNthFromEnd(self, head, n: int):
            dummy_head = ListNode(0, head)
            first = dummy_head.next
            for _ in range(n):
                first = first.next
            second = dummy_head
            while first:
                first, second = first.next, second.next
            second.next = second.next.next
            return dummy_head.next

162. (LC 83) Remove Duplicates from Sorted List
---------------------------------------------------
`83. Remove Duplicates from Sorted List <https://leetcode.com/problems/remove-duplicates-from-sorted-list/description/>`_
Easy

| **Keys:**
| -temporary node (tmp=cur.next)
| -nested while loop (delete while the same value)
 
| Input: singly linked sorted list
| Exploit the sorted nature of the list. 
| Remove all successive nodes with the same value as the current node.

Move next while cur==next. Delete. Make next cur. ::

    # 2->2->3->5
    # C  N  N

::

    ### Solution 1 (LC accepted)
    class Solution:
        def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
            cur = head
            while cur:
                next = cur.next  #**
                while next and next.val == cur.val:
                    next = next.next
                cur.next = next
                cur = next
            return head

T O(n), M O(1).

| **Issue**
| This way does not work for case [0,0,0,0,0] (returns [] instead of [0])
| (Though works for other cases including [1,1,1,1,1], [10,10,10]).

Initiating pointers like::

    # dummy > 0,0,0
    #   p1    p2

    def deleteDuplicates(self, head: Optional[ListNode]):
        dummy = ListNode()
        p1 = dummy
        p2 = dummy.next = head
        while p2:
            while p2 and (p1.val == p2.val):
                p2 = p2.next
            p1.next = p2
            p1 = p2
            p2 = p2.next if p2 else None
        return dummy.next

This works, initiating pointers like::

    # 0,0,0
    # p1    (p2 initiated as first step of the loop) #**

    def deleteDuplicates(self, head: Optional[ListNode]):
        p1=head
        while p1:
            p2=p1.next  #**
            while p2 and p1.val == p2.val:
                p2 = p2.next
            p1.next = p2
            p1 = p2
        return head


Solution 1 My V (LC accepted 60,60%) ::

    class Solution:
        def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
            cur=head
            dummy=ListNode()
            dummy.next=head
            while cur and cur.next:
                tmp = cur.next
                while tmp and (cur.val == tmp.val):
                    tmp = tmp.next
                cur.next = tmp
                cur = tmp
            return dummy.next

163. (LC 61) Rotate List
---------------------------
`61. Rotate List <https://leetcode.com/problems/rotate-list/description/>`_
Medium

Other names: Right shift for singly linked list.

| **Solution 1** [:ref:`2 <ref-label>`] LC accepted
| Steps:
| -compute list len + find tail
| -k=k%n
| -make a cycle connecting tail to head
| -find node just before the new head:
| steps to new head = n-k
| iterate from tail 
| assign new_head
| -break cycle:
| point new_tail to None

::

    class Solution:
        def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
            if not head:
                return head

            # Compute len of list + find tail
            tail, n = head, 1  # tail is current, because we want to stop at tail, we call it tail NOW
            while tail.next:
                n += 1
                tail = tail.next

            k %= n
            if k == 0:
                return head

            # Connect tail to head making a cycle
            tail.next = head
            steps_to_new_head, new_tail = n - k, tail
            while steps_to_new_head:
                steps_to_new_head -= 1
                new_tail = new_tail.next

            # If we found new tail, then next node is new head
            new_head = new_tail.next
            # Break the cycle
            new_tail.next = None
            return new_head

My V (LC accepted 20,90%)::

    class Solution:
        def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
            if not head or not head.next or k==0:
                return head
            dummy=ListNode()
            dummy.next=head
            
            # find list len
            list_len=0
            p1 = dummy
            while p1.next:
                p1=p1.next
                list_len+=1

            #How many rotations we need (if k > list len)
            rotations = k % list_len
            if rotations == 0:
                return dummy.next

            #Advance new p2 to node at which rotations should start
            steps = list_len - rotations
            p2=dummy
            for _ in range(steps):
                p2=p2.next
            #make p2.next new head
            head = p2.next

            #p2 is new tail (breaking list)
            p2.next = None

            #connect 2 sublists
            p1.next = dummy.next

            return head


**Solution 2** [:ref:`7 <ref-label>`] LC accepted ::

    class Solution:
        def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
            if head is None or head.next is None:
                return head
            cur, n = head, 0  #1
            while cur:
                n += 1
                cur = cur.next
            k %= n           #2
            if k == 0:
                return head
            fast = slow = head
            for _ in range(k):    #3
                fast = fast.next
            while fast.next:
                fast, slow = fast.next, slow.next

            new_tail = slow
            new_head = slow.next  
            new_tail.next = None
            fast.next = head     #4
            return new_head

| #1
| Compute n, len of list.
| #2
| k can be greater than list len, remove that difference with k=k%n
| #3
| E.g., k=3, original list:
| 2->3->5->3->2
| Move F from head, k-rotations:
| 2->3->5->3->2
| F  F1 F2 F3
| Move S from head, till F runs off the list:
| 2->3->5->3->2
| S        F3
| 2->3->5->3->2
| S  S1    F  F1
| 2->3->5->3->2
|    nT nH     
| S1 is new tail. 
| S1.next is new head
 
| #4
| 2->3->5->3->2
| H  nT nH    F
| Connect Fast to head.

::

    class Solution:
        def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
            if head is None or head.next is None:
                return head
            cur, n = head, 0
            while cur:
                n += 1
                cur = cur.next
            k %= n
            if k == 0:
                return head
            fast = slow = head
            for _ in range(k):
                fast = fast.next
            while fast.next:
                fast, slow = fast.next, slow.next

            ans = slow.next
            slow.next = None
            fast.next = head
            return ans

164 (LC 328) Odd Even Linked List
------------------------------------
`328. Odd Even Linked List <https://leetcode.com/problems/odd-even-linked-list/description/>`_
Medium

Note, you might be asked to connect evens+odds, or odds+evens. ::

    ### My V (LC accepted 97,75%)
    # In place
    class Solution:
        def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
            if not head or not head.next:
                return head
            dummy = ListNode()
            dummy.next = head
            p1=head            #pointer for odds
            p2=head.next       #pointer for evens
            tmp=p2             #store head of evens (to then connect odds tail to evens head)
            while p1.next and p2.next:  #important and
                p1.next = p2.next
                p1 = p1.next
                p2.next = p1.next
                p2 = p2.next
            p1.next = tmp
            return dummy.next


| **In short** (evens+odds):
| -Initiate 2 new lists, for evens and odds (using dummy heads)
| -tails, turn = [even_dummy_head, odd_dummy_head], 0          
| -Iterate the original list
| Alternating between odds and evens tail, tail.next = cur
| Keep moving the tail.
| -point odds tail to None
| -connect 2 lists (point evens tail to odds head)

::

    ### Solution 1 (odds + evens)
    def odd_even_merge(head):
        if not head:
            return head
        even_dummy_head, odd_dummy_head = ListNode(0), ListNode(0)
        tails, turn = [odd_dummy_head, even_dummy_head], 0
        cur=head
        while cur:
            tails[turn].next = cur
            cur=cur.next
            tails[turn] = tails[turn].next
            turn ^= 1
        tails[1].next = None
        tails[0].next = even_dummy_head.next
        return odd_dummy_head.next


    ### Solution 1-2 (evens+odds) EPI
    def even_odd_merge(head):
        if not head:
            return head
        even_dummy_head, odd_dummy_head = ListNode(0), ListNode(0)  #*1
        tails, turn = [even_dummy_head, odd_dummy_head], 0          #*2
        cur=head
        while cur:
            tails[turn].next = cur                  #*3
            cur=cur.next
            tails[turn] = tails[turn].next          
            turn ^= 1                            # Alternate between even/odd
        tails[1].next = None                        #4
        tails[0].next = odd_dummy_head.next         #5
        return even_dummy_head.next

| **Explained**
| 0->1->2->3->4
| #1
| Initiate two dummy heads.
| Start building two new lists, for evens and odds.
| De
| Do
 
| #2
| Keep track of the current tail for each list.
| turn variable to track which list we work with, alternate with XOR 1.
| De
| Tail e
 
| #3
| Point current tail to current node we work with.
| Move current, move tail.
| De->0
| Te..Te
| Do
 
| #4
| We will connect evens+odds, so odds tail will be the merged list Tail that points to None.
| #5
| Connecting even + odd (evens tail next = odds_dummy next, i.e. odds head)
 
| T O(N), S O(1)
| We avoid extra space by reusing the existing list nodes.

165. (LC 234) Palindrome Linked List
------------------------------------------
`234. Palindrome Linked List <https://leetcode.com/problems/palindrome-linked-list/description/>`_
Easy

| **Steps:**
| -Find half of the list with slow, fast (slow stops at half)
| -helper func to reverse second half (reverseList(slow))
| -compare two halves, iterating from respective heads
| (we do change the list in place. There is no req not to. When so, reverse back again.)

My note, finding the half works for both odd and even num of nodes in list. ::

    # 2->3->5->3->2
    # s s1 s2
    # f    f1     f2
    #      M
     
    # 2->3->3->2
    # s s1 s2
    # f    f1    f2
    #      M

| Second half might be longer, but we compare halves till we run out of ONE of them: 
|     while second_half_iter and first_half_iter:

::

    ### Solution 1 (T O(N), S O(1))
    class Solution:
        def isPalindrome(self, head: Optional[ListNode]) -> bool:
            # find the middle (slow)
            slow = fast = head
            while fast and fast.next:
                slow, fast = slow.next, fast.next.next

            def reverse_linked_list(head: ListNode) -> ListNode:
                prev, curr = None, head
                while curr:
                    temp = curr.next
                    curr.next = prev
                    prev = curr
                    curr = temp
                return prev

            first_half_iter = head
            second_half_iter = reverse_linked_list(slow)
            while second_half_iter and first_half_iter:
                if second_half_iter.val != first_half_iter.val:
                    return False
                second_half_iter, first_half_iter = second_half_iter.next, first_half_iter.next
            return True

    ### My V (LC accepted 95,70)
    class Solution:
        def isPalindrome(self, head: Optional[ListNode]) -> bool:
            if not head.next:
                return True
            dummy = ListNode()
            dummy.next = head
            s=f= dummy
            # Find last node of 1st sublist (i.e. linked list middle)
            while f and f.next:  #takes care of cases: list len odd, len even
                s = s.next
                f = f.next.next
            
            # Reverse 2nd sublist
            prev = None
            cur = s.next
            while cur:
                tmp = cur.next
                cur.next = prev
                prev = cur
                cur = tmp
            
            # Compare nodes from both ends
            iter1 = dummy.next
            iter2 = prev
            while iter2:
                if iter1.val != iter2.val:
                    return False
                iter1 = iter1.next
                iter2 = iter2.next
            return True

166. (LC 86) Partition List
--------------------------------
`86. Partition List <https://leetcode.com/problems/partition-list/description/>`_
Medium

| Other names: Implement linked list pivoting
| NOTE
| We preserve the relative order. We are NOT sorting.
| (If we wanted an ideal sorting, we would need a 3rd list, with values =x.
| So LC just makes it easier for you.)


| T O(N), S O(1).
| **Keys:**
| -Make 2 new separate lists for 1)values<x, 2)values>=x
| Iterate the input list to populate the 2 lists.
| -Combine 2 lists (don't forget to point tail2 to None).

::

    ### My submit 2 (LC accepted 40, 98)
    class Solution:
        def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
            head1 = tail1 = ListNode()
            head2 = tail2 = ListNode()
            while head:
                if head.val < x:
                    tail1.next = head
                    tail1 = tail1.next
                else:
                    tail2.next = head
                    tail2 = tail2.next
                head = head.next

            # Combine 2 lists
            tail1.next = head2.next
            tail2.next = None   #omitting this gets you "Error - Found cycle in the ListNode"
            return head1.next

167. (LC 2) Add Two Numbers
--------------------------------
`2. Add Two Numbers <https://leetcode.com/problems/add-two-numbers/description/>`_
Medium

| **Notes on task**
| List stores a number digits in reverse order.
| Better thinking: the least significant digit comes first 
 
| **Why**
| (this is how we add anyway, starting with the LSD).
| The list that you return is in the same order.
| Such a representation can be used to represent unbounded integers.
 
| **Hints**
| Use grade-school alg. First solve assuming no carry.
| (Why not convert to int. Ints word length is fixed by the machine architecture, 
| while lists can be arbitrary long.)

**Solution 1 V1** [:ref:`2 <ref-label>`] (LC accepted 90, 70) ::

    def add_two_nums(L1, L2):
        place_iter = dummy_head = ListNode()
        carry = 0
        while L1 or L2 or carry:
            data = carry + (L1.val if L1 else 0) + (L2.val if L2 else 0)
            L1 = L1.next if L1 else None
            L2 = L2.next if L2 else None
            place_iter.next = ListNode(data % 10)
            carry, place_iter = data // 10, place_iter.next
        return dummy_head.next

**Solution 1 V2** [:ref:`10 <ref-label>`] ::

    class Solution:
        def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
            dummy = ListNode()
            cur = dummy

            carry = 0
            while l1 or l2 or carry:
                v1 = l1.val if l1 else 0
                v2 = l2.val if l2 else 0

                # new digit
                val = v1 + v2 + carry
                carry = val // 10
                val = val % 10
                cur.next = ListNode(val)

                # update ptrs
                cur = cur.next
                l1 = l1.next if l1 else None
                l2 = l2.next if l2 else None

            return dummy.next

**My V** (LC accepted 50, 9%) ::

    class Solution:
        def addTwoNumbers(self, L1: Optional[ListNode], L2: Optional[ListNode]) -> Optional[ListNode]:
            dummy = iter3 = ListNode()
            iter1 = L1
            iter2 = L2
            carry=0
            while iter1 or iter2 or carry:
                val1 = iter1.val if iter1 else 0
                val2 = iter2.val if iter2 else 0
                res = val1 + val2 + carry
                ans = res % 10
                carry = res // 10
                iter3.next = ListNode(ans)
                iter1 = iter1.next if iter1 else iter1
                iter2 = iter2.next if iter2 else iter2
                iter3 = iter3.next
            return dummy.next

168. (LC 138) Copy List with Random Pointer
-----------------------------------------------
`138. Copy List with Random Pointer <https://leetcode.com/problems/copy-list-with-random-pointer/description/>`_
Medium

| **Notes on task**
| They speak of random being an index.
| But really node.random works in the same way as node.next (just points not to the
| next node, but to some random node.)
| Also, they confusedly refer to a node having [val, random_index]. While really
| node does NOT have the .random_index attribute on it. So a node has no info about
| the index of random node it points to. A node just points to AN index with node.random.
 
| **Steps**
| -Use two passes and a hash map
| Pass 1: creating hash map {oldNode: newNode}
| Here we create copies of nodes (just separate nodes, even without links yet).
| (With hash map we want to solve the case when we have to random point node3 to node5 e.g.
| So we use hash map to locate the copy of the 5th node.)
 
| Pass 2: leverage the hash map
| copy = oldToCopy[cur]
| Get copy of the node and point its copy.next, copy.random to the node in the hash table:
| copy.next = oldToCopy[cur.next]
| copy.random = oldToCopy[cur.random]

**Solution 1** [:ref:`10 <ref-label>`] ::

    """
    # Definition for a Node.
    class Node:
        def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
            self.val = int(x)
            self.next = next
            self.random = random
    """
    class Solution:
        def copyRandomList(self, head: "Node") -> "Node":
            oldToCopy = {None: None}  

            cur = head
            while cur:
                copy = Node(cur.val)
                oldToCopy[cur] = copy
                cur = cur.next
            cur = head
            while cur:
                copy = oldToCopy[cur]
                copy.next = oldToCopy[cur.next]
                copy.random = oldToCopy[cur.random]
                cur = cur.next
            return oldToCopy[head]

**My Vs** ::

    ### My V2 (LC accepted 70, 60%)
    class Solution:
        def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
            if not head:
                return
            d = {}
            cur = head
            while cur:
                new_node = Node(cur.val)
                d[cur] = new_node
                cur = cur.next
            cur2 = head
            while cur2:
                d[cur2].next = d[cur2.next] if cur2.next else None
                d[cur2].random = d[cur2.random] if cur2.random else None
                cur2 = cur2.next
            return d[head]


    ### My V (LC accepted 35, 80%)
    class Solution:
        def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
            d = {None: None} # {oldNode: newNode}
            iter1 = head
            while iter1:
                node = Node(iter1.val)
                d[iter1] = node
                iter1 = iter1.next
            iter1 = head
            while iter1:
                iter2 = d[iter1]
                iter2.random = d[iter1.random] 
                iter2.next = d[iter1.next]
                iter1 = iter1.next
                # iter2 = iter2.next  Don't need it
            return d[head]

169. (LC 146) LRU Cache
---------------------------
`LC 146. LRU Cache <https://leetcode.com/problems/lru-cache>`_
Medium

**LL + HASH MAP**
The standard algorithm for this task is to use a linked list and a hash map.

.. admonition:: Example 1.

    ::

        # Hash map "_map":                LL list "_keys":
        # 1: <value, position>          1 < 2 < _ < _ < 5 _ < _
        # 2: <value, position>         ^----------------| _keys.push_front(5)
        #                               <Most recent>     <Least recent> 
        # ..                            E.g. get(5), we move node 5 up front in LL.
        # 5: <value, position>          Get iter of 5 querying the hash map.
        #                               New position of 5 is _keys.begin()

| **Why hash + queue is not the way**
| Example 2.
| size=2
| put(1,1), put(2,2), get(1), get(1), get(1), put(3,3)
| hash={1:1,2:2}
| Before put(3)

::

    # q=[1,2,1,1,1]
    #    X X X

Because just popping from one end doesn't give the correct key to remove from hash 
before put(3). We might need to remove from the middle, not just from one end.

| **Why LL**
| Because we can remove from LL in O(1).
| Remove 5 from LL in example 1.
| The problem is that accessing in LL is O(n).
| To access 5 in O(1), we store its position (iterator) as a second value in hash map.
 
| **OrderedDict**
| The combination LL + hash map is implemented in 
| -OrderedDict() in Python
| -LinkedHashMap<> in Java
| (These two use LL+map internally.)
| (The dict keeps the order of keys insertion.) 
| So we don't need to do it ourselves.
| (In C++ have to use LL + hash map.)

**C++** [:ref:`14 <ref-label>`] LC accepted 65, 40.

.. code-block:: cpp

    class LRUCache {
            int _capacity;
            list<int> _keys;
            unordered_map<int, pair<int, list<int>::iterator>> _map;
    public:
        LRUCache(int capacity) : _capacity(capacity){ }
        
        int get(int key) {
            if(_map.find(key) != _map.end()){
                _keys.erase(_map[key].second);    //get position of key from map, erase from LL
                _keys.push_front(key);             //move to front of LL
                _map[key].second = _keys.begin(); //record key new position in map
                return _map[key].first;
            }
            return -1;
        }
        
        void put(int key, int value) {
            //IF KEY IN MAP, replace it: 
            if(_map.find(key) != _map.end()){
                _keys.erase(_map[key].second);      //remove from LL
                _keys.push_front(key);              //add to front
                _map[key] = {value, _keys.begin()}; //record in map, {value, new position of key}
            } else {
            //KEY NOT IN MAP AND WE ARE BEYOND OUR CAPACITY
                if(_keys.size() == _capacity){
                    _map.erase(_keys.back());  //erase from map key that is least recent in LL
                    _keys.pop_back();          //also remove that key from LL
                }
                _keys.push_front(key);          //add new key to LL, map
                _map[key] = {value, _keys.begin()};
            }
        }
    };

**Python** ::

    ### Solution 1 (LC accepted 99,90%, submission 2: 88,90%). From LC site users solutions.
    from collections import OrderedDict
    class LRUCache:

        def __init__(self, capacity: int):
            self.capacity = capacity
            self.cache = OrderedDict()

        def get(self, key: int) -> int:
            if key in self.cache:
                # Move the accessed key to the end
                self.cache.move_to_end(key)  #OrderedDict method
                return self.cache[key]
            return -1

        def put(self, key: int, value: int) -> None:
            if key in self.cache:
                # Update the value and move the key to the end
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add a new key-value pair
                if len(self.cache) >= self.capacity:
                    # Evict the least recently used key (first key in OrderedDict), FIFO
                    self.cache.popitem(last=False) #if last=True (default) then LIFO
                self.cache[key] = value

If you can't remember ".move_to_end()", 
can pop(key) and then insert it back into dict.

170. (LC 707) Design Linked List
-------------------------------------
`707. Design Linked List <https://leetcode.com/problems/design-linked-list/description/>`_
Medium

(Hints = optimizations)

| **Note:**
| You can be give invalid/out of range indices for methods addAtIndex, deleteAtIndex.
 
| Note, here we have to add ListNode class ourselves.
| (But it seem you can instantiate from it without defining it, although the 
| boilerplate omits it.)
 
| **Optimizations:**
| -keep track of list size!!
| -don't implement addAtHead, addAtTail. addAtIndex has all the edge cases.
| -consider instead of self.head=None, having self.dummy=ListNode() (so actual node with val 0) 

-Common error is to omit the <if cur.next> test in A LOT of places.

**Solution 1** [:ref:`7 <ref-label>`] ::

    class MyLinkedList:
        def __init__(self):
            self.dummy = ListNode()
            self.cnt = 0

        def get(self, index: int) -> int:
            if index < 0 or index >= self.cnt:
                return -1
            cur = self.dummy.next
            for _ in range(index):
                cur = cur.next
            return cur.val

        def addAtHead(self, val: int) -> None:
            self.addAtIndex(0, val)

        def addAtTail(self, val: int) -> None:
            self.addAtIndex(self.cnt, val)

        def addAtIndex(self, index: int, val: int) -> None:
            if index > self.cnt:
                return
            pre = self.dummy
            for _ in range(index):
                pre = pre.next
            pre.next = ListNode(val, pre.next)
            self.cnt += 1

        def deleteAtIndex(self, index: int) -> None:
            if index >= self.cnt:
                return
            pre = self.dummy
            for _ in range(index):
                pre = pre.next
            t = pre.next
            pre.next = t.next
            t.next = None
            self.cnt -= 1

::

    ### My V (LC accepted: T30% S70%)
    class ListNode:                 #you don't have to include it, though the boilerplate omits it
        def __init__(self, val=0):
            self.val = val
            self.next = None

    class MyLinkedList:
        def __init__(self):
            self.head = None

        def get(self, index: int) -> int:
            if not self.head:
                return -1
            elif index == 0:
                return self.head.val

            i = 0
            cur = self.head
            while cur.next and i != index:
                cur = cur.next
                i+=1
            if i != index:
                return -1
            else:
                return cur.val

        def addAtHead(self, val: int) -> None:
            if not self.head:
                self.head = ListNode(val)
            else:
                tmp = self.head
                self.head = ListNode(val)
                self.head.next = tmp

        def addAtTail(self, val: int) -> None:
            new_node = ListNode(val)
            if not self.head:
                self.head = new_node
            else:
                cur = self.head
                while cur.next:
                    cur = cur.next
                cur.next = new_node

        def addAtIndex(self, index: int, val: int) -> None:
            if not self.head and index > 0:
                return
            new_node = ListNode(val)
            if index == 0:
                if not self.head:
                    self.head = new_node
                else:
                    tmp = self.head
                    self.head = new_node
                    self.head.next = tmp
            else:
                cur = self.head
                i=0
                while cur.next and i != index-1:
                    cur = cur.next
                    i+=1
                if i == index-1:
                    tmp = cur.next
                    cur.next = new_node
                    new_node.next = tmp

        def deleteAtIndex(self, index: int) -> None:
            if index == 0:
                self.head = self.head.next
            else:
                cur = self.head
                i = 0
                while cur.next and i != index - 1:
                    cur = cur.next
                    i+=1
                if i == index-1 and cur.next: # and cur.next for when we are given index>len
                    tmp = cur.next
                    cur.next = cur.next.next
                    tmp.next = None

171. (LC 23) Merge k Sorted Lists
------------------------------------
`23. Merge k Sorted Lists <https://leetcode.com/problems/merge-k-sorted-lists/description/>`_
Hard

| **Clarification:**
| lists[index] gives you the head of a linked list.
 
| **Keys:**
| -Yes, reminiscent of a brute force. Because we merge lists in pairs. No fancies there.
| -We use a helper function to merge 2 lists. 
| My V:
| -Take 2 lists from list, append merged list to lists.
| OR:
| -Main alg: while + in range loops
| Take lists in pairs, merge them, put into a temporary list of lists.
| After each full pass, make lists=newMergedLists.
| Start anew with this new list of lists, till you are left with 1 list.

**Solution 1** [:ref:`10 <ref-label>`] O(N log K) ::

    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def mergeKLists(self, lists: List[ListNode]) -> ListNode:
            if not lists or len(lists) == 0:
                return None

            while len(lists) > 1:  #**1
                mergedLists = []
                for i in range(0, len(lists), 2):
                    l1 = lists[i]
                    l2 = lists[i + 1] if (i + 1) < len(lists) else None  #*2
                    mergedLists.append(self.mergeList(l1, l2))
                lists = mergedLists
            return lists[0]

        def mergeList(self, l1, l2):
            dummy = ListNode()
            tail = dummy

            while l1 and l2:
                if l1.val < l2.val:
                    tail.next = l1
                    l1 = l1.next
                else:
                    tail.next = l2
                    l2 = l2.next
                tail = tail.next
            if l1:
                tail.next = l1
            if l2:
                tail.next = l2
            return dummy.next

#**1
Merge taking lists in pairs from the original list of lists.
You will end up with twice as little lists. ::

    #     in range loop
    # l1,l2,l3,l4
    #   l5   l6
    #     while loop

Make the resulting new lists -> the new list of lists
Repeat while till you are left with 1 list.

| #**2
|     l2 = lists[i + 1] if (i + 1) < len(lists) else None  #*2
| Account for the case when there is an odd num of lists.
| Then it is ok to use our mergeList() func on list1 + None.

::

    ### My V (LC accepted 20, 90%)
    class Solution:
        def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
            def merge_lists(l1, l2):
                dummy = ListNode()
                cur = dummy
                while l1 and l2:
                    if l1.val <= l2.val:
                        cur.next = l1
                        l1 = l1.next
                    else:
                        cur.next = l2
                        l2 = l2.next
                    cur = cur.next
                cur.next = l1 if l1 else l2
                return dummy.next
            
            if len(lists) == 1:
                return lists[0]
            if not lists:
                return None
            while len(lists) > 1:
                lists.append(merge_lists(lists.pop(), lists.pop()))
            return lists[0]

172. (LC 25) Reverse Nodes in k-Group
----------------------------------------
`25. Reverse Nodes in k-Group <https://leetcode.com/problems/reverse-nodes-in-k-group/description/>`_
Hard
CHALLENGING

| The difficulty:
| E.g.
| 1>2>3>4>5 -->
| 2>1>4>3>5 (after reversal 1 points at 4)
 
| **Notes:**
| -we have a dummy node because we are potentially changing the list's head.
| -Save node right before our group (that node is not part of the group)
|     groupPrev = dummy
| -Save node after our group (not part of the group)
|     groupNext = kth.next

| #**1
| Reversing group
| Note that in the classic 'reverse LL' alg we have:
| prev=None
| But because here we reverse a group in LL, not the entire LL, if we use prev=None,
| we will break the LL. Hence we use:
| prev=kth.next

**Solution 1** [:ref:`10 <ref-label>`] ::

    class Solution:
        def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
            dummy = ListNode(0, head)
            groupPrev = dummy

            while True:
                kth = self.getKth(groupPrev, k)
                if not kth:
                    break
                groupNext = kth.next

                # reverse group    **1
                prev, curr = kth.next, groupPrev.next
                while curr != groupNext:
                    tmp = curr.next
                    curr.next = prev
                    prev = curr
                    curr = tmp

                tmp = groupPrev.next
                groupPrev.next = kth
                groupPrev = tmp
            return dummy.next

        def getKth(self, curr, k):
            while curr and k > 0:
                curr = curr.next
                k -= 1
            return curr






