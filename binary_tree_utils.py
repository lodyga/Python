# PYTHONPATH=/home/ukasz/Documents/IT/Python py solution.py
# export PYTHONPATH=/home/ukasz/Documents/IT/Python/


import binarytree as tree
from collections import deque


class TreeNode:
    """
    Definition for a binary tree node.
    """
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(node_list: list[int], node_type: tree.Node | TreeNode = TreeNode, with_lookup: bool = False) -> TreeNode | None:
    """
    Build binary tree from level order traversal list.

    If with_lookup=True, returns (root, lookup)
    where lookup is node value to node map.
    """
    # if tree.Node from binarytree is used
    # Becaues of node_list = [1, None, 3, None, None, None, 4] need to use try/except
    if node_type == tree.Node:
        try:
            return tree.build2(node_list)
        except:
            return tree.build(node_list)

    while node_list and node_list[-1] is None:
        node_list.pop()

    if not node_list:
        return None
    elif type(node_list) not in (list, tuple):
        raise TypeError(
            "Expected a list, got " + str(type(node_list).__name__)
        )

    root = node_type(node_list[0])
    queue = deque([root])
    index = 1
    lookup = {root.val: root} if with_lookup else None

    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if (
            index < len(node_list) and
            node_list[index] is not None
        ):
            node.left = node_type(node_list[index])
            queue.append(node.left)
            if with_lookup:
                lookup[node.left.val] = node.left
        index += 1

        # Assign the right child if available
        if (
            index < len(node_list) and
            node_list[index] is not None
        ):
            node.right = node_type(node_list[index])
            queue.append(node.right)
            if with_lookup:
                lookup[node.right.val] = node.right
        index += 1

    return (root, lookup) if with_lookup else root


def get_tree_values(root: TreeNode) -> list[int]:
    """
    Return tree node values in level order traversal format.
    """
    if root is None:
        return []
    elif type(root) not in (TreeNode, tree.Node):
        raise TypeError("Expected tree node, got " + str(type(root).__name__))
    elif root.val == root.left == root.right == None:
        return []

    values = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if node:
            values.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            values.append(None)

    while values[-1] is None:
        values.pop()
    return values


def is_same_tree(root1: TreeNode | None, root2: TreeNode | None) -> bool:
    """
    Time complexity: O(n)
    Auxiliary space complexity: O(n)
    Tags: binary tree, dfs, recursion
    """
    def dfs(node1, node2):
        if node1 is None and node2 is None:
            return True
        elif node1 is None or node2 is None:
            return False

        if node1.val != node2.val:
            return False

        left = dfs(node1.left, node2.left)
        right = dfs(node1.right, node2.right)
        return left and right

    return dfs(root1, root2)


if __name__ == "__main__":
    import binarytree as tree

    def is_same_tree(root1: TreeNode | None, root2: TreeNode | None) -> bool:
        """
        Time complexity: O(n)
        Auxiliary space complexity: O(n)
        Tags: binary tree, dfs, recursion
        """
        def dfs(node1, node2):
            if node1 is None and node2 is None:
                return True
            elif node1 is None or node2 is None:
                return False

            if node1.val != node2.val:
                return False

            left = dfs(node1.left, node2.left)
            right = dfs(node1.right, node2.right)
            return left and right
        return dfs(root1, root2)

    values_1 = [1, None, 3, None, None, None, 4]
    values_1 = [2,1,None,None,3]
    binary_tree_1 = build_tree(values_1, node_type=tree.Node)  # from binary_tree
    # binary_tree_1 = tree.build2(values_1)  # from binarytree
    # binary_tree_1 = build_tree(values_1, with_lookup=True)
    print(binary_tree_1)
    # print(is_same_tree(binary_tree_1, binary_tree_1))
