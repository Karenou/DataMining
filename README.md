# DataMining
MSBD5002 data mining assignments

## Frequent Pattern Mining
### Hash Tree

Hash tree is a data structure to store frequent pattern for efficient counting and search. The implementation of hash tree can be found in `frequent_pattern_mining/hash_tree.py`. To build a hash tree, first need to put the item set in variable `candidate_str` in the format of "{1 2 3}, {1 4 5}, {1 2 4}". For simplicity, this implementation is only for max leaf size of 3, so as the depth of the final hash tree. If there are leaves which cannot further split, with leaf size larger than 3. In these cases, we will use linked list to append the additional item set when printing out the tree.


Then run the following command in terminal, a hash tree will be built and printed out. The printed tree rotates 90 degree and the indentation is related to the depth.

```
python3 frequent_pattern_mining/hash_tree.py
```

### Frequent Pattern Tree

Frequent pattern tree is used to mine frequent patterns within a transaction database. The item dataset is storeed in `frequent_pattern_mining/DataSetA.csv`. It contains 12,526 records and each record records every single transaction in the grocery store. Each line represents a single transaction with names of products. You could modify the minimum support threshold in `fp_tree = FPTree(data, min_support=2500, item="NULL", item_freq=0)`. The default minimum support threshold is 2500.

To run the program, run the following command in terminal.

```
python3 frequent_pattern_mining/FP_tree.py
```