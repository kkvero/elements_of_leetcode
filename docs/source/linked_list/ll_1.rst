
Linked Lists Questions Part 1
================================
152. (LC 21) Merge Two Sorted Lists
-----------------------------------------
`21. Merge Two Sorted Lists <https://leetcode.com/problems/merge-two-sorted-lists/description/>`_
Easy

| **Keys (iterative approach)**
| -put the merged list into a new LL
| -use tail = dummy (as iterator for the new list and making dummy point to the head of the new list)
| -consider that l1, l2 can be of different len

::

    ### Solution 1
    def mergeList(l1, l2):
        dummy = ListNode()
        tail = dummy       #solution for making an empty node and making dummy.next point to newList head in one go

        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next    #we are given list HEADS, so we assign new head for l1
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next    #**
        if l1:               #if l1, l2 are of different sizes, stick the remainder of longer L to our new LL
            tail.next = l1
        if l2:
            tail.next = l2
        return dummy.next

    ### Solution 1-2 (Iterative)
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next

    class Solution:
        def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
            dummy = node = ListNode()
            while list1 and list2:
                if list1.val < list2.val:
                    node.next = list1
                    list1 = list1.next
                else:
                    node.next = list2
                    list2 = list2.next
                node = node.next
            node.next = list1 or list2
            return dummy.next

| **Iterative explained**
|     dummy = node = ListNode()
| Or
| dummy = current_node
| dummy = tail
| Empty node for list start and current node that we will be moving.
 
|     return dummy.next
| Because we are to return the head of the new merged list, if dummy.next is the head,
| then dummy must be an empty node.
 
|     list1 = list1.next
| Because initially we are given list1, list2 <<heads>>.
| So list1 represents not the whole list, but list1's head node.
| So by list1=list1.next we set a different head for the list.
 
|     node.next = list1 or list2
| Means the remaining of either list1 or list2.
| I.e. for the situation when e.g.:
| L1=1>2>3
| L2=1>4>5>6>7
| When we run out of nodes in L1, we just add the remainder of L2 to the answer List.

.. admonition:: Python x = a or b

    returns a if bool(a) evaluates True, else it evaluates b.
    Has the effect of returning the first item that evaluates True, or the last item 
    (even if it evaluates to False).

    Equivalent to:
    determination = arg_1 if arg_1 else arg_2 if arg_2 else 'no arguments given!'

**Solution 2 (Recursive)** ::

    # V1 [10]
    class Solution:
        def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
            if not list1:
                return list2
            if not list2:
                return list1
            lil, big = (list1, list2) if list1.val < list2.val else (list2, list1)
            lil.next = self.mergeTwoLists(lil.next, big)
            return lil

    # V2 [7]
    class Solution:
        def mergeTwoLists(
            self, list1: Optional[ListNode], list2: Optional[ListNode]
        ) -> Optional[ListNode]:
            if list1 is None or list2 is None:
                return list1 or list2
            if list1.val <= list2.val:
                list1.next = self.mergeTwoLists(list1.next, list2)
                return list1
            else:
                list2.next = self.mergeTwoLists(list1, list2.next)
                return list2

Iterative my versions::

    ### Iterative my V1 (LC accepted 92,71%)
    def merge(l1, l2):
        dummy = ListNode()
        cur = dummy
        while l1 or l2:
            if not l1:
                cur.next = l2
                l2 = l2.next
            elif not l2:
                cur.next = l1
                l1 = l1.next
            elif l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next     #<========*
        return dummy.next

| #*Remember that with e.g. cur.next=l2 we don't yet move the node, just the pointer.
| cur.next=l2
| D---->L2
| cur   cur.next
 
| cur=cur.next
| D---->L2
|       cur

::

    ### My V2 iterative (LC accepted 75, 88)
    class Solution:
        def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
            if not list1 and not list2:
                return None
            if not list1:
                return list2
            if not list2:
                return list1
            
            dummy = ListNode()
            l3 = ListNode()
            dummy.next = l3
    
            while list1 and list2:
                if list1.val < list2.val:
                    l3.val = list1.val
                    list1 = list1.next
                elif list2.val <= list1.val:
                    l3.val = list2.val
                    list2 = list2.next
                if list1 and list2:      #then making a new empty node
                    l3.next = ListNode()
                    l3 = l3.next
            if list1 or list2:
                l3.next = list1 if list1 else list2
            return dummy.next

153. (LC 206) Reverse Linked List
--------------------------------------
`LC 206. Reverse Linked List <https://leetcode.com/problems/reverse-linked-list/description/>`_
Easy

| **ITERATIVE**
| **Keys:**
| -prev=None
| Loop:
| -use temp
| -<point current to prev>
| -shift prev/cur/temp
| -return prev because that's the new head (while current is  pointing to None)

Solution [:ref:`10 <ref-label>`]::

    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    class Solution:
        def reverseList(self, head: ListNode) -> ListNode:
            prev, curr = None, head
            while curr:
                temp = curr.next
                curr.next = prev
                prev = curr
                curr = temp
            return prev

| The above is the iterative approach, T O(n) M O(1).
| There is also recursive approach, but it is T O(n), M O(n).

**RECURSIVE** ::

    ### LC accepted 70, 60.
    class Solution:
        def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
            if not head or not head.next:
                return head
            new_head = self.reverseList(head.next)  #1
            head.next.next = head                   #2
            head.next = None
            return new_head                         #3

| 1->2->3->4
| #1
| Calling recursively on the sublist 2->3->4

::

# 1 -> 2->3->4
# h   h.n

| #2
| After calling on the subproblem, what we are left to do is point 2 to 1, point 1 to None.

::

# 1 -> 2->3->4
# h   h.n
# None <- 1 <- 2

| 2 = head.next 
| 1 = head
| To point 2 to 1: 2.next = 1, so head.next.next = head
 
| #3
| Because we need to return the head of the reversed LL.
| We store the call to the function in a variable new_head, means it will store
| the return value of the function when no subproblems left, when no head.next,
| 1>2>3>4, head=4, no head.next, so returns head, 4. And starts working on the call stack,
| the actual reversing None<3<4, None<2<3<4, None<1<2<3<4.

154. (LC 92) Reverse Linked List II
---------------------------------------
`92. Reverse Linked List II <https://leetcode.com/problems/reverse-linked-list-ii/description/>`_

| Medium
| Other names: Reverse sublist, reverse between
 
| **Hooks:**
| -left, right are integers, not nodes.
| -So you reach the node at left with iteration.
| -Don't break the sublist from the main list. Keep the node before left pointing to 
| the node at left.

::

    # Visualization
    #  1 -> 2 -> 3 -> 4 -> 5
    #       L         R
    
    # prev<-|
    #  1 -> 2 <- 3 <- 4 X 5
    
    #                 |->prev
    #  1    4 -> 3 -> 2 X 5
    #  |--------------^
    #       R         L
    
    #  1 -> 4 -> 3 -> 2 -> 5

| After reversing the sublist between L-R:
| (L->5) point the L node to the node immediately after R
| (1->R) Node immediately before L point to R.

**Solution** [:ref:`10 <ref-label>`] ::

    # Python3
    class Solution:
        def reverseBetween(
            self, head: Optional[ListNode], left: int, right: int
        ) -> Optional[ListNode]:
            dummy = ListNode(0, head)

            # 1) reach node at position "left"
            leftPrev, cur = dummy, head
            for i in range(left - 1):
                leftPrev, cur = cur, cur.next

            # Now cur="left", leftPrev="node before left"
            # 2) reverse from left to right
            prev = None
            for i in range(right - left + 1):
                tmpNext = cur.next
                cur.next = prev
                prev, cur = cur, tmpNext

            # 3) Update pointers
            leftPrev.next.next = cur  # LP.next.next means pointing L.next to: cur which is node after "right" 
            leftPrev.next = prev  # prev is "right"
            return dummy.next

| **Explained**
| 1) Reach node at L.
| -dummy points to the head
| -leftprev, with it we keep track of the node immediately before L 
| (after reaching L, we save it and not move it in the next step)
| 2) Reverse sublist L-R (iterating till R).
| Just normal reverse.
|             temp = curr.next
|             curr.next = prev
|             prev = curr
|             curr = temp
| 3) Connect the reversed sublist L-R to the main list.
| L->cur (point L to node after R, which is stored in cur)
| leftPrev -> R (node before L point to R, R is stored in prev)
| 
| Connecting:
| LeftPrev.next.next references the node at left because we didn't disconnect the list.
| Even after the reversal the node before left still points to the node at left.

::

    # 1>2>3>4>5
    #   l   r
    # 1 4>3>2> 
    # |-----^
    # And 4 still points to 5.
    # 1 4>3>2>N  5
    #   |-------^

| My rewrite 
| (mistakes: we don't ever break any pointers of nodes, i.e. pointing them to None:
|     # beforeL.next = None  <==Nope
|     # cur.next = None  <==Nope
| ) (LC accepted)

::

    # Python3
    class Solution:
        def reverseBetween(self, head: Optional[ListNode], L: int, R: int) -> Optional[ListNode]:
            dummy = ListNode(0, head)
            beforeL = dummy
            cur=head
            for _ in range(L-1):
                beforeL=cur
                cur=cur.next

            #reverse
            prev=None
            for _ in range(R-L+1):
                tmp=cur.next
                cur.next=prev
                prev=cur
                cur=tmp
            beforeL.next.next=cur
            beforeL.next=prev
            return dummy.next

155. (LC 141) Linked List Cycle
------------------------------------
`141. Linked List Cycle <https://leetcode.com/problems/linked-list-cycle/description/>`_
Easy ::

    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None


Solution [:ref:`10 <ref-label>`]::

    class Solution:
        def hasCycle(self, head: ListNode) -> bool:
            slow, fast = head, head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                if slow == fast:
                    return True
            return False

My V (LC accepted 60, 90)::

    class Solution:
        def hasCycle(self, head: Optional[ListNode]) -> bool:
            if not head or not head.next:
                return False
            slow = head.next
            fast = head.next.next
            while slow != fast:
                if not fast or not fast.next:
                    break
                slow = slow.next
                fast = fast.next.next
            return slow == fast

C++
^^^^^^

.. code-block:: cpp

    // My V (LC accepted 40, 20)
    class Solution {
    public:
        bool hasCycle(ListNode *head) {
            if(!head || !head->next)
                return false;
            ListNode* slow = head->next;
            ListNode* fast = head->next->next;
            while(slow != fast){
                if(!fast || !fast->next)
                    break;
                slow = slow->next;
                fast = fast->next->next;
            }
            return slow == fast;
        }
    };

**Code for two methods: Hash and Floyd's** [:ref:`14 <ref-label>`]

.. code-block:: cpp

    #include <iostream>
    #include <unordered_map>
    using namespace std;

    struct Node{
        int data;
        Node* next;
        Node(int d=0): data(d), next(nullptr){} //constructor
    };

    //PRINT LL
    void print_ll(Node* head){
        Node *n = head; //dummy Node not to advance the head pointer
        while(n){
            cout << n->data << "->";
            n = n->next;
        }
        cout << "NULL\n"; 
    }

    //HASH MAP
    bool detect_loop_map(Node *head){
        unordered_map<Node*, bool> visited;
        Node *cur = head;
        while(cur){
            if(visited[cur])
                return true;
            visited[cur] = true;
            cur = cur->next;
        }
        return false;
    }
    //FLOYD'S TWO POINTERS SLOW, FAST
    bool detect_loop_floyd(Node *head){
        Node * slow = head;
        Node *fast = head;
        while(slow && fast && fast->next){
            slow = slow->next;
            fast = fast->next->next;
            if(slow == fast)
                return true;
        }
        return false;
    }

    void create_loop(Node *head){
        Node *cur = head;
        while(cur->next){    //reach the last node
            cur = cur->next;
        }
        cur->next = head->next;  //point last node to the node after head
    }

    int main(){
        //1->2->3->4->5->6->nullptr
        Node n1(1), n2(2), n3(3), n4(4), n5(5), n6(6);
        n1.next = &n2;
        n2.next = &n3;
        n3.next = &n4;
        n4.next = &n5;
        n5.next = &n6;
        
        Node *head = &n1;
        
        //NO LOOP, check with 2 methods
        bool hasloop = detect_loop_map(head);
        cout << boolalpha << hasloop << endl; //false
        hasloop = detect_loop_floyd(head);
        cout << hasloop << endl; //false
        
        //CREATE LOOP and check again with 2 methods
        create_loop(head);
        hasloop = detect_loop_map(head);
        cout << hasloop << endl; //true
        hasloop = detect_loop_floyd(head);
        cout << hasloop << endl; //true
    }

156. (LC 142) Linked List Cycle II
--------------------------------------
`142. Linked List Cycle II <https://leetcode.com/problems/linked-list-cycle-ii/description/>`_
Medium

**Note on the task:**

From the problem description you might think they want us to return the index of 
the node from which the cycle starts:
Output: tail connects to node index 1.
But from the code boilerplate we see that we are to return the node itself.
+If no cycle, return None.

| **Solution 1** [:ref:`2 <ref-label>`]
| **Keys:**
| -Is there a cycle
| use S, F. Start both from head. While F.next and F.next.next loop till they meet.
| -if S and F met, from that node:
| 2nd loop. Start S at head, F at where they met. Move each +1. 
| Where they meet again is the start of the cycle.

::

    class Solution:
        def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
            fast = slow = head
            while fast and fast.next and fast.next.next:
                slow, fast = slow.next, fast.next.next
                if slow is fast:                           # There is a cycle, so find cycle start
                    slow = head
                    while slow is not fast:
                        slow, fast = slow.next, fast.next  # both advance +1
                    return slow                            # where S and F meet
            return None                                    # no cycle


**Solution 2** ::

    class Solution:
        def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
            fast = slow = head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                if slow == fast:
                    break
            
            # If there is no cycle
            if not fast or not fast.next: 
                return None

            ans = head
            while ans != slow:
                ans = ans.next
                slow = slow.next
            return ans

| **Explained**
| -Find if it is a cycle (find node where fast and slow meet).
| -separate block to return None if there is no cycle
| -Finding cycle start. Set two pointers
| 1)where fast and slow met, e.g. take slow
| 2)set new pointer to the head of the entire list, e.g. ans
| Move both slow and ans with the same speed. When they meet, that's your starting 
| node of the cycle. (There is a "mathematical" reason why it is so.)

::

    # a    a1,s4
    # s,f  s1,f2  f1,s2  s3,f3
    # 3->    2 ->   0 ->   -4
    #        ^--------------|

::

    ### My V (LC accepted 70, 60%)
    class Solution:
        def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
            if not head or not head.next:
                return None
            slow, fast = head, head
            while True:
                try:
                    slow = slow.next
                    fast = fast.next.next
                    if slow == fast:
                        break
                except:
                    return None
            ss1 = slow
            ss2 = head
            while ss1 != ss2:
                ss1 = ss1.next
                ss2 = ss2.next
            return ss1

157. (LC 160) Intersection of Two Linked Lists
-------------------------------------------------
| `LC 160. Intersection of Two Linked Lists <https://leetcode.com/problems/intersection-of-two-linked-lists/description/>`_
| Easy
| (Other names: Test for overlapping lists (lists are without cycle))

| **Notes on task interpretation** [:ref:`2 <ref-label>`]:
| (Practical usage - reducing memory footprint.)
| Intersection - node that is common to two lists.
| Lists overlap if they have the same tail node. 
| (Also, once the lists converge at a node,they cannot diverge at a later node.)

| O(m + n) time, space O(1)
| **Solution 1** [:ref:`10 <ref-label>`]

::

    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    class Solution:
        def getIntersectionNode(
            self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
            l1, l2 = headA, headB
            while l1 != l2:
                l1 = l1.next if l1 else headB  #IMPORTANT if l1, not if l1.next
                l2 = l2.next if l2 else headA
            return l1

    ### My V (LC accepted 40,50%)
    class Solution:
        def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
            p1=headA
            p2=headB
            while p1 != p2:
                if not p1:      #when p1=None
                    p1 = headB  #again, set p1 to headB, not p1.next
                elif not p2:
                    p2 = headA
                else:
                    p1=p1.next
                    p2=p2.next
            return p1

| **Keys:**
| -two pointers
| -increment +1
| -achieving the len difference of lists: 
| switching to the head of the <opposite> list when pointer=None, i.e. reaches list end.
| **Why it does not make an infinite loop:**
| Because we allow pointers to slide off the respective list to None value before 
| pointing it to the head of the opposite list. Then at some point both l1, l2 will 
| be None, i.e. the same, even if lists don't intersect.
 
| We point l1, l2 to opposite heads just once. 
| After that, they are pointed to the same (their own) heads. If no intersection, they 
| will eventually run off the lists into None.

::

    #    _->_->
    #             _->_->_
    # _->_->_->
     
    # 1)We set p1, p2 to the heads of the respective nodes.
    #    p1_->_->
    #             _->_->_
    # p2_->_->_->
     
    # 2)Increment each +=1, until one p reaches the end of its respective end
    # (p2 didn't yet reach the list end).
    #    _->_->
    #            _->p2_->p1_
    # _->_->_->
     
    # 3)Set that point to the head of the OPPOSITE list.
    #    _->_->
    #             _->_->p2_
    # p1_->_->_->
     
    # 4)When p2 reached the list end, set it to the head of the opposite list too.
    #    p2_->_->
    #             _->_->_
    # _->p1_->_->
     
    # 5)Loop goes on till node at p1 and p2 is the same
    #    _->_->
    #           p1,p2_->_->_
    # _->_->_->

| **Alternative solution 1** (uses memory O(n)):
| 1)Hash set. Add nodes in L1 to a hash set.
| 2)Iterate L2 nodes. If node in hash, you have the intersect node.
 
| **Alternative solution 2**
| Similar, more verbose alternative to Solution 1 (the same efficiency and main logic as Solution 1).
| 1)Recognize that L1,L2 can have different lengths (as above, len 5, len 6).
| 2)Start p1 at the head of a shorter list. p2 at the head of the longer list.
| 3)Increment only p2 by the diff in len of the two lists.
| 4)Begin the main alg. Compare nodes at p1, p2. if not the same: +=1.
| If the same, that's your intersect node.

158. Intersection of Two Linked Lists - lists may have cycles
------------------------------------------------------------------
(Other names: Test for overlapping lists (lists may have cycles))
The same as the previous task.
But this time one, both or non of the lists may have a cycle. ::
    
    # A->B
    #    V
    # C->D->E->F
    # ^--------|

Both C and D are acceptable answers.

| **CASE ANALYSIS**
| =>Test each LL for cycles
| (use 142. Linked List Cycle II, returns Node cycle start, or None if no cycle)
| 1)Neither list is cyclic ->
| Then just use solution for '160 LC (157 My numbering). Intersection of Two Linked Lists (lists don't have cycles)'
| 2)One is cyclic, the other is not. Then they cannot overlap. We are done.
| 3)Both are cyclic.
|     Subcases:

::

    # 1/ cycles are disjoint. No overlap.
    # 2/ Merge node is before the cycle start
    # A->B
    #    V
    # C->D->E->F
    #       ^--|
    # 3/ Merge node is in the cycle
    # A->B
    #    V
    # C->D->E->F
    # ^--------|

**Solution 1** ::

    def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:
        fast = slow = head
        while fast and fast.next and fast.next.next:
            slow, fast = slow.next, fast.next.next
            if slow is fast:                           # There is a cycle, so find cycle start
                slow = head
                while slow is not fast:
                    slow, fast = slow.next, fast.next  # both advance +1
                return slow                            # where S and F meet
        return None

    def getIntersectionNode(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        l1, l2 = headA, headB
        while l1 != l2:
            l1 = l1.next if l1 else headB
            l2 = l2.next if l2 else headA
        return l1

    def overlapping_lists(L1, L2):
        # Find cycle starts if any
        root1, root2 = detectCycle(L1), detectCycle(L2)

        if not root1 and not root2:  # Both L1, L2 no cycle
            return getIntersectionNode(L1, L2)  # Note, the func assumes they do overlap
        elif (root1 and not root2) or (not root1 and root2):  # Only 1 list has cycle, so no overlap
            return None

        # Both have cycles
        # Test if they are not disjoint
        # If overlap: Starting at the cycle start of L2, you should meet the cycle start of L1.
        temp = root2
        while True:
            temp = temp.next
            if temp is root1 or temp is root2:
                break
            if temp is not root1:
                return None  # Disjoint cycles

        ### One L has cycle

        # Helper func. Distance from head to intersect node, i.e. unique stem_length
        def distance(a, b):
            dis = 0
            while a is not b:
                a = a.next
                dis += 1
            return dis

        # Overlap before cycle start
        stem1_length, stem2_length = distance(L1, root1), distance(L2, root2)
        if stem1_length > stem2_length:
            L2, L1 = L1, L2
            root1, root2 = root2, root1
            # List with longer unique stem, move it till both stems are the same dist from merge node **1
            for _ in range(abs(stem1_length - stem2_length)):
                L2 = L2.next

            # Takes care of both: 1)when overlap is before cycle (overlap=L1==L2)
            # 2)And the subcase when overlap is within the cycle (overlap=root1)  **2
            while L1 is not L2 and L1 is not root1 and L2 is not root2:
                L1, L2 = L1.next, L2.next
            return L1 if L1 is L2 else root1

#**1::

    #           L2  
    # (L1)    A->B
    #            V    
    # (L2) L1 C->D->E->F  (At E: root1, root2)
    #               ^--|

Pointers L2 at B and L1 at C are now the same distance from D (merge node).
Now we just have to move them .next till L1=L2

| #**2
| Merge node is in the cycle.
| A:root1, C:root2, if stem1_lenght > stem2_length, roots swap, making A:root2 

::

    # A->B
    #    V
    # C->D->E->F
    # ^--------|

Last comments [:ref:`2 <ref-label>`]:
If Ll == L2 before reaching root1, it means the overlap first occurs
before the cycle starts; otherwise, the first overlapping node is not
unique, we can return any node on the cycle.

159. (LC 143) Reorder List
-------------------------------
`143. Reorder List <https://leetcode.com/problems/reorder-list/description/>`_
Medium

| **Quick notes:**
| -to find halves, iter1=head, iter2=head.next, while iter2 and iter2.next
| -you are more likely to need 
| tmp = cur.next
| than
| tmp = cur
| If you will be modifying cur later on.

::

    ### Solution 1 (My rewrite V1, LC accepted: Memory beats 94%, Time 40%)
    class Solution:
        def reorderList(self, head: Optional[ListNode]) -> None:
            """ Do not return anything, modify head in-place. """

            slow=head
            fast=head.next
            #find 2 halves of the list
            while fast and fast.next:
                slow=slow.next
                fast=fast.next.next

            #reverse 2nd half
            cur=slow.next # slow=B, slow.next=cur=C (start reversing at C)
            slow.next=None # A>B C>D (breaking B>C link, B>None)
            prev=None
            while cur:
                tmp=cur.next
                cur.next=prev
                prev=cur
                cur=tmp

            #merge half1 and half2      #1
            cur1=head #head of half1
            cur2=prev #head of half2
            while cur2:
                tmp1=cur1.next
                tmp2=cur2.next
                cur1.next=cur2
                cur2.next=tmp1
                cur1=tmp1
                cur2=tmp2

    # #1 Alternative merge
    #         iter1 = head
    #         iter2 = prev
    #         while iter2:
    #             nex = iter1.next
    #             iter1.next = iter2
    #             iter1 = iter2
    #             iter2 = nex

| **Explained**
| # Overview
| Linked List
| A>B>C>D
| Where half would be.
| A B | C D  (For odd len list also: A B C | D E)
| 
| ###1 Determine the half of the list. Use slow, fast pointers.
| (Note, they begin at different positions.)

::

    # A B C D
    # S F
    # 
    # A B C D
    #   S   F
 
| ###2 We need to REVERSE THE 2ND HALF OF THE LIST.
| The head of the 2nd list is at S.next
|     second = slow.next
| The actual breaking of the original list consists in pointing the last node of the
| first half to None. I.e. S.next=None 
|     prev = slow.next = None
| 
| Recall how we reverse a LL (we use exactly the same code):
|         prev, curr = None, head
|         while curr:
|             temp = curr.next
|             curr.next = prev
|             prev = curr
|             curr = temp
|         return prev

###3 THE MERGE::

    # After reversing the 2nd half we have:
    # A>B C<D
    #   V V
    # Memorize B, C
    # A>B C<D
    #   V V
    # 
    # Point head of L1 to head of L2. 
    # Point head of L2 to memorized L1.next.
    #   V---|
    # A B C D
    # |-----^
    # 
    # Move heads of L1, L2 to memorized B, C.
    # A>D>B C
    #     V V

| # Merging in code
|     first, second = head, prev   #prev is the last node in list (head of the 2nd half)
|     while second:
|         tmp1, tmp2 = first.next, second.next
|         first.next = second
|         second.next = tmp1
|         first, second = tmp1, tmp2
| 
|     first, second = head, prev
| Assign first and second to list heads.
| first=A, second=D
| 
|     while second:
| Loop while 2nd list has no nodes.
| 
|     tmp1, tmp2 = first.next, second.next
| Put nodes after heads to temp vars. 
| tmp1=B, tmp2=C
| 
|     first.next = second
|     second.next = tmp1
| Point head of 1st list to head of 2nd list.
| Point head of L2 to memorized L1.next.
| 
|     first, second = tmp1, tmp2
| Reassign heads of L1, L2.

**Solution 1** formal ::

    class Solution:
        def reorderList(self, head: ListNode) -> None:
            # find middle
            slow, fast = head, head.next
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next

            # reverse second half
            second = slow.next       #the starting point/node for the list 2nd half (where slow sopped + 1)
            prev = slow.next = None  #break list into 2 halves (pointing to None breaks the list)
            while second:
                tmp = second.next
                second.next = prev
                prev = second
                second = tmp

            # merge two halves
            first, second = head, prev   #prev is the last node in list (head of the 2nd half)
            while second:
                tmp1, tmp2 = first.next, second.next
                first.next = second
                second.next = tmp1
                first, second = tmp1, tmp2

160. (LC 237) Delete Node in a Linked List
----------------------------------------------
`237. Delete Node in a Linked List <https://leetcode.com/problems/delete-node-in-a-linked-list/description/>`_
Medium

| The input node is guaranteed not to be the tail node.
| You are given the pointer to a node to delete.
| -What's the trouble?
| Deleting a node usually requires modifying its predecessor's next pointer and 
| the only way to get to the predecessor is to traverse the list from head, 
| which requires O(n) time.
| -The trick.
| Delete given node's successor. Copy next node's data into the current node. Delete next node.
| This takes O(1).

::

    class Solution:
        def deleteNode(self, node_to_delete):
            """
            :type node: ListNode
            :rtype: void Do not return anything, modify node in-place instead.
            """
            node_to_delete.val = node_to_delete.next.val
            node_to_delete.next = node_to_delete.next.next

        # OR 
        # def deleteNode(node):
        #     node.val = node.next.val
        #     node.next = node.next.next

