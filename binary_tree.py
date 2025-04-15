import binarytree as tree
from collections import deque


class TreeNode:
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(node_list, node_type: tree.Node | TreeNode = TreeNode):
    # return tree.build(node_list)  # import binarytree as tree
    """
    Build binary tree from level order traversal list.
    """
    while node_list and node_list[-1] is None:
        node_list.pop()

    if not node_list:
        return []
    elif type(node_list) not in (list, tuple):
        raise TypeError("Expected a list, got " +
                        str(type(node_list).__name__))

    root = node_type(node_list[0])
    queue = deque([root])
    index = 1

    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


def tree_values(root):
    # return root.values  # import binarytree as tree
    if not root:
        return []

    values = []
    queue = deque([root])

    while any(queue):
        queue_for_level = deque()
        while queue:
            node = queue.popleft()
            values.append(node.val if node else None)
            queue_for_level.append(node.left if node else None)
            queue_for_level.append(node.right if node else None)
        queue = queue_for_level

    while values[-1] is None:
        values.pop()
    return values


if __name__ == "__main__":
    import binarytree as tree

    class Solution:
        def invertTree(self, root: TreeNode | None) -> TreeNode | None:
            """
            Time complexity: O(n)
            Auxiliary space complexity: O(n)
            Tags: birary tree, dfs, recursion
            """
            if not root:
                return None

            root.left, root.right = root.right, root.left
            self.invertTree(root.left)
            self.invertTree(root.right)

            return root

    values_1 = [4, 2, 7, 1, 3, 6, 9]
    values_inverted_1 = [4, 7, 2, 9, 6, 3, 1]

    binary_tree_1 = build_tree(values_1, node_type=tree.Node)  # from binary_tree
    # binary_tree_1 = tree.build(values_1)  # from binarytree
    # print(binary_tree_1)

    binary_tree_1_inverted = Solution().invertTree(binary_tree_1)
    # print(binary_tree_1_inverted)

    # print(tree_values(binary_tree_1_inverted) == [4, 2, 7, 1, 3, 6, 9])
    # print(binary_tree_1_inverted.values == values_inverted_1)  # from binarytree

    # print(tree_values(Solution().invertTree(build_tree([4, 2, 7, 1, 3, 6, 9]))) == [4, 7, 2, 9, 6, 3, 1])  # from binary_tree
    # print((Solution().invertTree(tree.build([4, 2, 7, 1, 3, 6, 9]))).values == [4, 7, 2, 9, 6, 3, 1])  # from binarytree

    print(tree_values(Solution().invertTree(build_tree([2, 1, 3]))) == [2, 3, 1])
    print(tree_values(Solution().invertTree(build_tree([4, 2, 7, 1, 3, 6, 9]))) == [4, 7, 2, 9, 6, 3, 1])
    print(tree_values(Solution().invertTree(build_tree([7, 3, 15, None, None, 9, 20]))) == [7, 15, 3, 20, 9])
    print(tree_values(Solution().invertTree(build_tree([]))) == [])
