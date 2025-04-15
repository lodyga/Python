from typing import Optional

class ListNode:
    """
    Definition for singly-linked list.
    """
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverse a singly-linked list.
        Time complexity: O(n)
        Auxiliary space complexity: O(1)
        """
        node = head
        prev = None
        while node:
            node_next = node.next
            node.next = prev
            prev = node
            node = node_next
        return prev


def list_to_linked_list(values: list) -> Optional[ListNode]:
    """
    Convert a Python list to a singly-linked list.
    """
    anchor = node = ListNode()
    for value in values:
        node.next = ListNode(value)
        node = node.next
    return anchor.next

values = [5, 2, 13, 3, 8]
linked_list_1 = list_to_linked_list(values)

def linked_list_to_list(node: Optional[ListNode]) -> list:
    """
    Convert a singly-linked list to a Python list.
    """
    values = []
    while node:
        values.append(node.val)
        node = node.next
    return values

# validation test case
solution = Solution()
print(
    "Actual output: ", 
    linked_list_to_list(solution.reverseList(list_to_linked_list(values))), 
    "Expected output: ",
    values[::-1])
print(linked_list_to_list(solution.reverseList(list_to_linked_list(values))) == values[::-1])
