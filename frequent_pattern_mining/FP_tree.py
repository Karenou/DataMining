import csv
from csv import reader
import itertools


class TreeNode:
    def __init__(self, parent=None, item=None, frequency=0) -> None:
        self.item = item
        self.frequency = frequency
        # parent node
        self.parent = parent
        # a dict of children nodes
        self.next = {}
        # the nodes having the same item as the current node is linked to
        self.link_node = None


class FPTree:

    def __init__(self, data, min_support=2500, item="NULL", item_freq=0) -> None:
        self.min_support = min_support
        self.data = data  # List[List[str]]
        self.item_freq = {} # {item: count}
        self.header_table = {}  # {item: [frequency, TreeNode]}
        self.node_link = {}
        self.tree = TreeNode(item=item, frequency=item_freq)  # root node for FP/conditional tree

    def get_item_frequency(self):
        """
        get the frequency of unique item in the data
        filter out infrequent items
        sort by item frequency
        """
        for row in self.data:
            for item in row:
                if item in self.item_freq.keys():
                    self.item_freq[item] += 1
                else:
                    self.item_freq.update({item: 1})

        # remove infrequent 1-itemset
        for item in list(self.item_freq.keys()):
            if self.item_freq[item] < self.min_support:
                del self.item_freq[item]

        # sort the items by frequency
        self.item_freq = dict(sorted(self.item_freq.items(), key=lambda x: x[1], reverse=True))

    def construct_header_table(self):
        """
        construct the initial empty header table
        """
        for item in self.item_freq.keys():
            self.header_table[item] = None

    def build_tree(self):
        """
        build the fp tree
        """
        self.get_item_frequency()
        self.construct_header_table()

        # no frequent items, no need to build tree
        if not self.header_table:
            return 
 
        for row in self.data:
            # order transactions by item frequency
            sort_item = [i for i in row if i in self.item_freq.keys()]
            sort_item.sort(key=lambda x: self.item_freq[x], reverse=True)

            if sort_item:
                self.update_tree(self.tree, sort_item)
    
    def update_tree(self, node:TreeNode, transaction: list):
        """
        recursively insert new item in the fp tree
        @param node: the current node
        @param transaction: one sorted transaction to insert to the FP tree
        """
        item = transaction.pop(0)
        if item in node.next.keys():
            child = node.next.get(item, None)
            child.frequency += 1
        else:
            child = TreeNode(parent=node, item=item, frequency=1)
            node.next[item] = child
        
            # link the new node to header table
            if not self.header_table[item]:
                self.header_table[item] = child
            else:
                curr = self.header_table[item]
                while curr.link_node:
                    curr = curr.link_node
                curr.link_node = child

        # insert the remaining items to the fp tree and header table
        if transaction:
            self.update_tree(child, transaction)

    def check_single_path(self, node:TreeNode) -> bool:
        """
        check whether the current conditional fp tree has only one single path
        @param node: the current node to check number of children
        return: a boolean
        """
        if len(node.next.keys()) > 1:
            return False
        elif not node.next:
            return True
        else:
            return self.check_single_path(list(node.next.values())[0])

    def get_item_combinations(self) -> dict:
        """
        return: a dictionary with all possible combination of patterns and corresponding support
        """
        patterns = {}
        suffix = []

        if self.tree.item != "NULL":
            suffix = [self.tree.item]
            # !!! need to put the root.item as a pattern, need to convert to tuple as list cannot be dict's key
            patterns[tuple(suffix)] = self.tree.frequency

        # the smallest subset size is 1
        for size in range(1, len(self.item_freq) + 1):
            # generate all possible combinations without order and duplication
            for choice in itertools.combinations(self.item_freq.keys(), size):
                # append the suffix to be a frequent pattern, convert to tuple as list cannot be dict's key
                pattern = tuple(sorted(list(choice) + suffix))
                # support of the pattern equals the minimum support of items in the choice
                patterns[pattern] = min([self.item_freq[item] for item in choice])

        return patterns

    def find_conditional_pattern_base(self, node) -> list:
        """
        @param node: the first node in the header table that stores the item
        return: the list of conditional pattern bases
        """
        suffixes = []
        conditional_tree_input = []
            
        # get all nodes along link_node list, store in suffixes
        while node:
            suffixes.append(node)
            node = node.link_node

        # travel upstream to get the path till root, here path excludes the suffix
        for suffix in suffixes:
            path = []
            curr = suffix.parent
            while curr.parent:
                path.append(curr.item)
                curr = curr.parent

            # need to mimic the item frequency
            for i in range(suffix.frequency):
                conditional_tree_input.append(path)
            
        return conditional_tree_input

    def build_conditional_tree(self) -> dict:
        """
        build conditional tree resursively to get a final single path
        return: a dict of the frequent patterns of all conditional patterns trees built upon the current fp tree
        """
        freq_pattents = {}

        for item, node in self.header_table.items():
            conditional_tree_input = self.find_conditional_pattern_base(node)
            conditional_tree = FPTree(conditional_tree_input, self.min_support, item, self.item_freq[item])
            conditional_tree.build_tree()
            conditional_tree_patterns = conditional_tree.get_frequent_patterns()

            # add the conditional tree patterns into final frequent patterns
            for pattern in conditional_tree_patterns.keys():
                if pattern in freq_pattents.keys():
                    freq_pattents[pattern] += conditional_tree_patterns[pattern]
                else:
                    freq_pattents.update({pattern: conditional_tree_patterns[pattern]})

        return freq_pattents

    def get_frequent_patterns(self) -> dict:
        """
        mine all frequent patterns
        return: a dictionary of frequent patterns
        """
        # if the conditional pattern tree has merged to only one path, directly get the combinations
        if self.check_single_path(self.tree):
            return self.get_item_combinations()
        # otherwise, need to first build the conditioanl pattern trees and find the conditional base
        else:
            freq_patterns = self.build_conditional_tree()

            # append suffix in patterns if it is a conditional fp tree
            if self.tree.item != "NULL":
                patterns = {}
                for key in freq_patterns.keys():
                    patterns[tuple(sorted(list(key) + [self.tree.item]))] = freq_patterns[key]
                return patterns
            else:
                return freq_patterns


def read_data(path) -> list:
    """
    load data from csv, store in self.data
    return: a list of transaction data
    """
    data = []
    with open(path, "r") as f:
        csv_reader = reader(f)
        for row in csv_reader:
            # remove empty '' at the end
            data.append(row[:-1])
    return data

def print_pattern(patterns: dict) -> None:
    """
    filter out 1-itemset from frequent pattern, order by support descending and print out
    @param patterns: mined frequent patterns
    """
    patterns = dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
    for pattern, support in patterns.items():
        if len(pattern) > 1:
            print(pattern, ": ", support)


data = read_data("DataSetA.csv")
fp_tree = FPTree(data, min_support=2500, item="NULL", item_freq=0)
fp_tree.build_tree()
frequent_pattern = fp_tree.get_frequent_patterns()
print_pattern(frequent_pattern)

