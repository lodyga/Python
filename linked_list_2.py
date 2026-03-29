from typing import Optional, List, Any

class ListNode:
    """
    Definition for singly-linked list.
    """
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LinkedList:
    def __init__(self, values: list[int] = None):
        """
        Initialize a linked list. 
        If values are provided, creates a linked list from them.
        """
        self.head = None
        self._length = 0
        if values:
            self.create_from_list(values)

    def __len__(self):
        """
        Get the length of the linked list.        
        """
        return self._length
    
    def __str__(self) -> str:
        """
        String epresentation of the linked list.
        """
        return f"LinkedList({self.to_list()})"
    
    def create_from_list(self, values: list) -> Optional[ListNode]:
        """
        Convert a Python list to a singly-linked list.
        """
        self.anchor = self.node = ListNode()
        for value in values:
            self.node.next = ListNode(value)
            self.node = self.node.next
            self._length += 1
        self.head = self.anchor.next

    def to_list(self) -> list:
        """
        Convert a singly-linked list back to a Python list.
        """
        values = []
        node = self.head
        while node:
            values.append(node.val)
            node = node.next
        return values

    def visualize(self):
        return " -> ".join("(" + str(char) + ")"
                           for char in self.to_list())


    def reverse_list(self) -> Optional[ListNode]:
        """
        Reverse a singly-linked list.
        Time complexity: O(n)
        Auxiliary space complexity: O(1)
        """
        node = self.head
        prev = None
        while node:
            node_next = node.next
            node.next = prev
            prev = node
            node = node_next
        self.head = prev
        return self


values = [5, 2, 13, 3, 8]
linked_list_1 = LinkedList(values)
print(linked_list_1.to_list())
print(linked_list_1.visualize())
print("Linked List length:", linked_list_1._length)
print(linked_list_1)
print(linked_list_1.reverse_list().to_list())
print(linked_list_1.visualize())




# define LinkedList class to iterate throungh linked list
class LinkedList:
    def __init__(self, head):
        self.head = head
        self.node = None

    # define iterator
    def __iter__(self):
        self.node = self.head
        return self

    # iterate
    def __next__(self):
        if self.node:
            val = self.node.val
            self.node = self.node.next
            return val
        else:
            raise StopIteration

linkedList1 = LinkedList(linked_list_1)

# iterate through LinkedList
for linkedlist_node in linkedList1:
    print(linkedlist_node)
