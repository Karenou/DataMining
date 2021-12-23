
class TreeNode:
    
    def __init__(self, item_sets=None, left=None, mid=None, right=None,
                next=None, link_list_key=None) -> None:
        """
        @param item_sets: a list of item_sets stored in this node
        @param left: left subtree node
        @param mid: mid subtree node
        @param right: right subtree node
        """
        self.item_sets = [] if not item_sets else item_sets
        self.left = left
        self.mid = mid
        self.right = right

    def get_leaf_size(self) -> int:
        return len(self.item_sets)


class HashTree:

    def __init__(self, item_sets: str, max_leaf_size=3) -> None:
        """
        @param item_sets: a string in format of "{itemset_1}, {itemset_2}"
        @param max_leaf_size: default to be 3
        """
        item_set_list = item_sets.split(", ")
        self.item_sets = [list(map(int, c[1:-1].split(" "))) for c in item_set_list]
        self.max_leaf_size = max_leaf_size
        self.tree = TreeNode()
    
    def hash_function(self, item_set:list, depth: int, node:TreeNode) -> TreeNode:
        """
        @param item_set: prefix to be hashed
        @param depth: the current depth of hash tree
        @param node: parent node
        return: a child node the item_set belongs to 
        """
        # cannot further split the data, use link_list
        if depth >= len(item_set):
            depth = - 1
        hash_key = item_set[depth] % 3
        if hash_key == 1:
            return hash_key, node.left
        elif hash_key == 2:
            return hash_key, node.mid
        else:
            return hash_key, node.right
    
    def build_tree(self) -> None:
        """
        build the hash tree to store all item_sets in its nodes
        """
        self.tree.left = TreeNode()
        self.tree.mid = TreeNode()
        self.tree.right = TreeNode()
        
        for c in self.item_sets:
            # sort the item set in ascending order
            c.sort()

            depth = 0
            _, node = self.hash_function(c, depth, self.tree)
            
            # if the current node is an internal node, recursively search the leaf node
            while node.left and node.mid and node.right:
                depth += 1
                _, node = self.hash_function(c, depth, node)
            
            # reach the leaf node
            # insert the item_set if not exceeds max_leaf_size
            # !!! remove deplicate item_sets
            if node.get_leaf_size() < self.max_leaf_size:
                node = self.insert_item_set(node, c)
            # split the node when exceeds max_leaf_size
            else:
                depth += 1
                node = self.insert_item_set(node, c)
                if not self.check_link_list(node, depth):
                    node = self.split_node(node, depth)
    
    def insert_item_set(self, node:TreeNode, itemset: list) -> TreeNode:
        """
        @param node: the current node to insert itemset
        @param itemset: the new itemset to be inserted
        append a new itemset into the node if there is no existing one
        return the current node
        """
        for c in node.item_sets:
            if c == itemset:
                return node
        
        node.item_sets = node.item_sets + [itemset]
        return node

    def check_link_list(self, node:TreeNode, depth:int) ->bool:
        """
        check whether need to convert to link list when size exceeds max_leaf_size
        """
        hash_nodes = []
        for c in node.item_sets:
            hash_key, _ = self.hash_function(c, depth, node)
            hash_nodes.append(hash_key)
        # whether all hash_nodes are the same, if yes, then add link list
        hash_nodes.sort()
        for i in range(len(hash_nodes) - 1):
            if hash_nodes[i] != hash_nodes[i+1]:
                return False
        return True

    def split_node(self, node: TreeNode, depth: int) -> TreeNode:
        """
        @param node: the parent node to be splited
        @param depth: the current depth
        return: the parent node 
        """
        node.mid = TreeNode()
        node.left = TreeNode()
        node.right = TreeNode()

        for c in node.item_sets:
            hash_key, p = self.hash_function(c, depth, node)
            p.item_sets = p.item_sets + [c]

        node.item_sets = []
        return node

    def print_tree(self) -> None:
        """
        print the tree in inorder mode
        """
        # ascii code use to name the link list
        n_link_list = 96

        # helper function to recursively print out the trees
        def helper(node, depth, side=None):
            nonlocal n_link_list

            # reach leaf node
            if not node.left and not node.mid and not node.right:
                if node.item_sets:
                    # print out in linked link format
                    if node.get_leaf_size() > self.max_leaf_size:
                        n_link_list += 1
                        link_list_input = ["{" + ",".join(list(map(str, c))) + "}" for c in node.item_sets[self.max_leaf_size-1:]]
                        print("\t" * (depth-1), node.item_sets[:self.max_leaf_size - 1], \
                            "a link list %s (%s: %s)" % (
                                chr(n_link_list), 
                                chr(n_link_list), 
                                "->".join(link_list_input)
                                )
                        )
                    else:
                        print("\t" * (depth-1), node.item_sets)
                    
                    print()
                return
            
            helper(node.right, depth + 1, "right")
            helper(node.mid, depth + 1, "mid")
            helper(node.left, depth + 1, "left")

        if self.tree:
            helper(self.tree, 0)

candidate_str = "{1 2 3}, {1 4 5}, {1 2 4}, {1 2 5}, {1 5 9}, {1 3 6}, {2 3 4}, {2 5 9}, {3 4 5}, {3 5 6}, {3 5 9}, {3 8 9}, {3 2 6}, {4 5 7}, {4 1 8}, {4 7 8}, {4 6 7}, {6 1 3}, {6 3 4}, {6 8 9}, {6 2 1}, {6 4 3}, {6 7 9}, {8 2 4}, {8 9 1}, {8 3 6}, {8 3 7}, {8 4 7}, {8 5 1}, {8 3 1}, {8 6 2}"
ht = HashTree(candidate_str)
ht.build_tree()
ht.print_tree()