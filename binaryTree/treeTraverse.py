class TreeNode:
    def __init__(self, number):
        self.num = number
        self.left = None
        self.right = None


def pre_order_traversal_recursive(head):
    if not head:
        return None
    print(head.num, end='')
    print(" ", end='')
    pre_order_traversal_recursive(head.left)
    pre_order_traversal_recursive(head.right)


def pre_order_traversal_cycle(head):
    stack = [head]
    while len(stack) > 0:
        print(head.num, end='')
        print(" ", end='')
        if head.right:
            stack.append(head.right)
        if head.left:
            stack.append(head.left)
        head = stack.pop()


def in_order_traverse_recursive(head):
    if not head:
        return None
    in_order_traverse_recursive(head.left)
    print(head.num, end='')
    print(" ", end='')
    in_order_traverse_recursive(head.right)


def in_order_traverse_cycle(head):
    stack = []
    inner_node = head
    while inner_node or len(stack) > 0:
        if inner_node:
            stack.append(inner_node)
            inner_node = inner_node.left
        else:
            inner_node = stack.pop()
            print(inner_node.num, end='')
            print(" ", end='')
            inner_node = inner_node.right


def post_order_traverse_recursive(head):
    if not head:
        return None
    post_order_traverse_recursive(head.left)
    post_order_traverse_recursive(head.right)
    print(head.num, end='')
    print(" ", end='')


def post_order_traverse_cycle(head):
    stack_1 = [head]
    stack_2 = []
    while len(stack_1) > 0:
        inner_node = stack_1.pop()
        stack_2.append(inner_node)
        if inner_node.left is not None:
            stack_1.append(inner_node.left)
        if inner_node.right is not None:
            stack_1.append(inner_node.right)
    while len(stack_2) > 0:
        print(stack_2.pop().num, end="")
        print(" ", end="")


def layer_traverse(head):
    if not head:
        return
    stack = [head]
    while len(stack) > 0:
        inner_node = stack.pop(0)
        print(inner_node.num, end="")
        print(" ", end="")
        if inner_node.left:
            stack.append(inner_node.left)
        if inner_node.right:
            stack.append(inner_node.right)


def tree_builder():
    node1 = TreeNode(1)
    node2 = TreeNode(2)
    node3 = TreeNode(3)
    node4 = TreeNode(4)
    node5 = TreeNode(5)
    node6 = TreeNode(6)
    node7 = TreeNode(7)

    node1.left = node2
    node1.right = node3

    node2.left = node4
    node2.right = node5

    node3.left = node6
    node3.right = node7

    return node1


if __name__ == "__main__":
    head_node = tree_builder()
    # pre order traversal
    print("pre order traversal*****************************")
    print("recursive")
    pre_order_traversal_recursive(head_node)
    print("\n")
    print("cycle")
    pre_order_traversal_cycle(head_node)
    print("\n")
    # in order traversal
    print("in order traversal*****************************")
    print("recursive")
    in_order_traverse_recursive(head_node)
    print("\n")
    print("cycle")
    in_order_traverse_cycle(head_node)
    print("\n")
    print("post order traversal*****************************")
    print("recursive")
    post_order_traverse_recursive(head_node)
    print("\n")
    print("cycle")
    post_order_traverse_cycle(head_node)
    print("\n")
    print("layer traversal*****************************")
    layer_traverse(head_node)