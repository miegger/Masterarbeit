class Node:
    def __init__(self, k, l=None, r=None):
        self.key = k
        self.left = l
        self.right = r


def in_order_rec(node):
    if node is None:
        node
    if node is not None:
        in_order_rec(node.left)
        print(node.key, end=' ')
        in_order_rec(node.right)


class Tree:
    def __init__(self):
        self.root = None

    def add(self, key):
        if self.root is None:
            self.root = Node(key)

        p = self.root

        while p is not None:
            if key == p.key:
                return # Key already exists, do nothing
            if key < p.key:
                if p.left is None:
                    p.left = Node(key)
                    return
                else:
                    p = p.left
            else:
                if p.right is None:
                    p.right = Node(key)
                    return
                else:
                    p = p.right
    
    def in_order(self):
        in_order_rec(self.root)

            

t = Tree()
t.add(6)
t.add(8)
t.add(4)
t.in_order()