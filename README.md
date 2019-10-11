# CSC422-DecisionTree
Implementation of a decision tree for the classification of mushroom's according to edibility. 

Generates a decision tree from ```train_shrooms.csv``` and validates it against ```test_shrooms.csv```. A representation of the decision tree is printed along with validation metrics (recall, precision, accuracy, F1 measure) and tuned hyperparameter values. 

For hyperparameter tuning, replace the ```split_positions``` and ```min_node_sizes``` arrays with their commented out definitions, then run the program. The values in these arrays will be used to construct decision trees and the best pair of values is chosen for constructing the final decision tree. Note that hyperparameter tuning may take hours with the full array definitions used.

Run as:
```
./decision_tree.py
```

See notes in source code for additional information.
