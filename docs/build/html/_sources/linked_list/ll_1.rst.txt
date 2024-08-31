
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

















