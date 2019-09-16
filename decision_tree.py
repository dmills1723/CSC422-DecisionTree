#!/usr/bin/env python3

'''
    Author: Daniel Mills ( demills ) 
  
    Trains decision tree on mushroom training dataset using ID3 decision tree
    algorithm. Some slight modifications have been made from the original algorithm.

    - An additional stopping criterion for node splitting has been added. If a node
      has less instances than Node.MIN_NODE_SIZE, splitting stops. 
    - All splits are binary. Theoretically, N-ary splits could be made by 
      discretizing the continuous variable into N partitions when splitting,
      but this is too computationally complex and difficult to implement.
    - When finding the best split condition value for an attribute, 
      the dataset is split on Node.split_positions condition values.
      This may be updated to check every split possible split position at 
      a later time.
'''

import sys 
import csv 
import numpy as np
import codecs
from sklearn import tree

###############################################
################# FUNCTIONS ###################
###############################################

'''
  Returns entropy of the class attribute column. Values are 
  assumed to be either a floating point 1.0 or 0.0.
'''
def calculateEntropy( class_column ) :
  num_total = class_column.size
  num_zero = 0
  num_one  = 0
  for val in class_column :
    if ( int( round( val ) ) == 0 ) :
      num_zero += 1
    else :
      num_one += 1

  rel_freq_zero = ( num_zero * 1.0 ) / num_total
  rel_freq_one  = ( num_one  * 1.0 ) / num_total

  entropy = 0
  # To avoid a divide by zero error.
  if (( num_zero == 0 ) or (num_one == 0 )) :
    return 0.0
  else :
    entropy -= ( rel_freq_zero * np.log2( rel_freq_zero ) )
    entropy -= ( rel_freq_one  * np.log2( rel_freq_one  ) )

  return entropy

'''
  Return the collective entropy of the child nodes created by
  splitting the column on "split_cond".

PLAGIARISM NOTE: This method was borrowed and refactored from code I wrote for HW2 
                 in the file "hw2q2A.py", submitted for homework group 14.
'''
def getCollectiveEntropy( dataset, split_cond, attr_idx ) : 
  
  # Splits dataset by "split_cond" for specified attribute.
  less_data = dataset[ dataset[:, attr_idx] <  split_cond ]
  grtr_data = dataset[ dataset[:, attr_idx] >= split_cond ]

  # Calculates total number of records in parent 
  # node, left node, and right node. (left is less_data)
  num_total = dataset.shape[0]
  num_less  = less_data.shape[0]
  num_grtr  = grtr_data.shape[0]
  
  # Calculates relative frequency of left and right node.
  less_rel_freq = ( num_less * 1.0 ) / num_total
  grtr_rel_freq = ( num_grtr * 1.0 ) / num_total 

  # Calculates entropy of each child node's class attribute.
  #print( less_data[:, 0 ] )
  less_data_entropy = calculateEntropy( less_data[:, 0] )
  grtr_data_entropy = calculateEntropy( grtr_data[:, 0] )


  # Calculates collective entropy of two child nodes.
  coll_entropy = ( less_rel_freq * less_data_entropy ) + ( grtr_rel_freq * grtr_data_entropy )

  return coll_entropy

def getSplits( min_val, max_val, split_positions ) :
  splits = []
  interval = ( max_val - min_val ) / ( split_positions + 1 )
  curr_split = min_val 
  for i in range(0, split_positions ) :
    curr_split += interval
    splits.append( curr_split )
  return splits

def getSplitInfo( column, split_cond ) :
  # Gets total number of records before split.
  num_total = column.size

  # Splits dataset of node into two sets for child nodes.
  smaller_vals  = column[ column[:] < split_cond ]
  larger_vals   = column[ column[:] >= split_cond ]

  # Base value for calculating split info
  split_info = 0

  # Calculates relative frequencies.
  rel_freq_smaller = ( smaller_vals.size * 1.0 ) / num_total
  rel_freq_larger = ( larger_vals.size * 1.0 ) / num_total
  
  # Calculates value of each iteration of sum.
  smaller_term = ( rel_freq_smaller * np.log2( rel_freq_smaller ) )
  larger_term  = ( rel_freq_larger  * np.log2( rel_freq_larger  ) )

  # Calcultes split info.
  split_info -= smaller_term
  split_info -= larger_term

  return split_info
  
def getSplittingCondition( dataset, global_entropy, split_positions ) :
  max_gain_ratio = -1.0
  max_idx = -1
  max_split_cond = -1.0

  # Iterate over columns of dataset. 
  for col_idx in range( 1, dataset.shape[1] ) :
    # Stores current column being evaluated.
    column = dataset[ :, col_idx ].astype( float )
    
    # Calc min and max, store split points.
    min_val = column.min()
    max_val = column.max()
    splits = getSplits( min_val, max_val, split_positions )

    # Iterate across splitting conditions. Get gain ratio for each.
    for split_cond in splits :
      # Calculate collective impurity I( children )
      child_entropy = getCollectiveEntropy( dataset, split_cond, col_idx )
      
      # global_entropy is I( parent )
      # Calculate information gain as IG = I( parent ) - I( children )
      info_gain = global_entropy - child_entropy
      
      # Calculate Split Info as SI
      split_info = getSplitInfo( column, split_cond ) 

      # Calculate gain ratio GR = IF / SI
      gain_ratio = ( info_gain / split_info )

      # If gain ratio is higher than current max, set new max gain
      # ratio, current split condition, and current index.
      if ( gain_ratio > max_gain_ratio ) :
        max_gain_ratio = gain_ratio
        max_split_cond = split_cond
        max_idx = col_idx
        
  return max_idx, max_split_cond

def getMajorityClass( class_column ) :
  num_zero = 0
  num_one  = 0
  for val in class_column :
    if ( int( round( val ) ) == 0 ) :
      num_zero += 1
    else :
      num_one += 1
  if ( num_zero > num_one ) :
    return "edible"
  else :
    return "poisonous"

###############################################
###############################################
################ Node Class ###################
###############################################
###############################################

'''
  This class is used to construct a binary decision tree. Each node stores 
  the attribute it is split on (like PC1 or COLOR) and the value at which
  this attribute is split. Additionally, the dataset before splitting is 
  also stored. The left child Node will store the subset of a Node's dataset
  with attribute values less than the splitting condition, and the right child
  will store the subset with attribute values greater than or equal to the 
  splitting condition.

  PLAGIARISM NOTE: To construct this class, I referenced a tutorialspoint.com demo 
        on Python binary trees . Since binary trees aren't relevant to this class,
        I think (and hope) it's okay to duplicate some of their code.
        LINK: https://www.tutorialspoint.com/python/python_binary_tree.htm
'''
class Node:
  #SPLIT_POSITIONS = 50
  def __init__(self, dataset, level, min_node_size, split_positions):
    # Records associated with node. 
    self.dataset = dataset
    # Level in tree node is located at.
    self.level = level

    # If Node is not a leaf, stores index in dataset of attribute split on by this node
    self.attr_idx = None

    # If node is not a leaf, this stores records split on (attr_label < split_cond)
    self.left_child = None
    # If node is not a leaf, this stores records split on (attr_label >= split_cond)
    self.right_child = None

    # If node is not a leaf, this is the decimal value the node is split on. 
    self.split_cond = None
    # Label of the attribute split on for this node.
    self.attr_label = None

    # If node is a leaf, this stores the class of the leaf.
    self.leaf_class = None
    
    # Sets number of split positions.
    self.split_positions = split_positions

    # Set to true if this node is a leaf (i.e. a stopping condition is satisfied)
    self.isLeaf = False
    # A stopping condition. If a Node has min_node_size or less elements it isn't split.
    self.min_node_size = min_node_size

  '''
    Recurisvely prints the structure of the tree.
  '''
  def printTree( self ) :
    dashes = '|    ' * self.level
    if ( self.isLeaf == True ) :
      print( dashes + "( " + self.leaf_class + " ) " )
      return
    else :
      print( dashes + "[ " + self.attr_label + " " + self.split_cond.astype(str) + " ] " )
      self.left_child.printTree()
      self.right_child.printTree()
    #print( "Level: %d\n" % self.level )
    #print( "Attribute label: %s\n" % self.attr_label )
    return


  '''
     Attempts to split the node. If the base case conditions are met, return.
     If the base case conditions are not met (i.e. the node is splittable and
     is not a leaf) split into two nodes and call split() on them. Then return.
     For debugging, print out this node's information.

     Also increment level before calling it on the child nodes. 
  '''
  def split( self ) :
    # Calculate "global" entropy.
    global_entropy = calculateEntropy( self.dataset[1:, 0].astype(float) )

    ##### Check if base conditions are met. Set as leaf and return if so. #####

    # If entropy = 0 (all features are of same class)
    if ( global_entropy == 0 ) :
      self.isLeaf = True
      self.leaf_class = getMajorityClass( self.dataset[1:, 0].astype(float) )
      return 

    # If number of attributes = 0
    elif ( np.shape( self.dataset )[1] == 1 ) :
      self.isLeaf = True
      self.leaf_class = getMajorityClass( self.dataset[1:, 0].astype(float) )
      return

    # If Node has less than MIN_NODE_SIZE instances.
    #elif ( np.shape( self.dataset )[0] <= self.MIN_NODE_SIZE ) :
    elif ( np.shape( self.dataset )[0] <= self.min_node_size ) :
      self.isLeaf = True
      self.leaf_class = getMajorityClass( self.dataset[1:, 0].astype(float) )
      return

    ##### Split dataset on highest gain ratio attribute & condition #####

    # Determine attribute and condition to split on. 
    self.attr_idx, self.split_cond = getSplittingCondition( self.dataset[1:, :].astype(float), global_entropy, self.split_positions )
    self.attr_label = self.dataset[ 0, self.attr_idx ]

    # Split dataset into attribute labels (top row) and samples.
    attr_labels = self.dataset[ 0, : ]
    samples = self.dataset[ 1:, : ].astype( float )
    
    # Split samples into two datasets. 
    left_data  = samples[ samples[:, self.attr_idx] < self.split_cond ]
    right_data = samples[ samples[:, self.attr_idx] >= self.split_cond ]

    # Converts labels into 2D array with one row, so 
    # we can concatenate it with other 2D arrays.
    attr_labels = np.array( [ attr_labels ] )

    # Adds back attribute labels to each dataset.
    left_data  = np.concatenate( (attr_labels, left_data), axis=0 )
    right_data = np.concatenate( (attr_labels, right_data), axis=0 )

    ##### Remove split-upon attribute. Create child nodes. #####
  
    # Remove split-upon attribute from each dataset.
    left_data  = np.delete( left_data, self.attr_idx, 1 )
    right_data = np.delete( right_data, self.attr_idx, 1 )
    
    # Create left node from split.
    #self.left_child = Node( left_data, self.level + 1 )
    self.left_child = Node( left_data, self.level + 1, self.min_node_size, self.split_positions )
    # Create right node from split.
    #self.right_child = Node( right_data, self.level + 1 )
    self.right_child = Node( right_data, self.level + 1, self.min_node_size, self.split_positions )

    ##### Split left then right child #####

    # Call split on left dataset.
    self.left_child.split()

    # Call split on right dataset.
    self.right_child.split()

    return

  '''
     Returns 1 to classify as "poisonous".
     Returns 0 to classify as "edible".
  '''
  def getClassification( self, sample_row ) :
    if ( self.isLeaf ) :
      if ( self.leaf_class == 'poisonous' ) :
        return 1
      else :
        return 0
    else :
      # "sample_row" does not contain the class attribute, 
      # so we decrement the index to check.
      idx = self.attr_idx - 1
      if ( sample_row[ idx ] < self.split_cond ) :
        return self.left_child.getClassification( sample_row )
      else :
        return self.right_child.getClassification( sample_row )

###############################################
################  Execution  ##################
###############################################

# Paths to training and testing datasets.
train_filename = "train_shrooms.csv"
test_filename = "test_shrooms.csv"

with codecs.open( train_filename, "r", encoding="utf-8-sig" ) as csv_file :
  # Creates 2D numpy array from CSV
  csv_reader = csv.reader( csv_file, delimiter=',')
  train_dataset = list( csv_reader )
  train_dataset = np.array( train_dataset ).astype( object )

  with codecs.open( test_filename, "r", encoding="utf-8-sig" ) as csv_file_test :
    # Creates 2D numpy array from CSV
    csv_reader_test = csv.reader( csv_file_test, delimiter=',')
    test_dataset = list( csv_reader_test )
    test_dataset = np.array( test_dataset ).astype( object )

    # Removes first row (labels) 
    samples = test_dataset[1:, :].astype( float )

    # Hyperparameters to perform grid search on. Uncomment the longer arrays 
    # and add values to tune hyperparameters. It is VERY slow, though.
    #split_positions = [ 10, 20, 30, 40, 50 ]
    split_positions = [ 50 ]
    #min_node_sizes = [ 1, 3, 5, 10, 20, 30, 50 ]
    min_node_sizes = [ 10 ]

    # Best validation measures for a pair of "split_positions" and "min_node_sizes" values.
    # The best measures are selected by according to what maximizes recall.
    best_recall = 0.0
    best_precision = 0.0
    best_accuracy = 0.0
    best_f1_measure = 0.0
    best_split_pos = None
    best_node_size = None
    
    # Checks 10 split position hyper-parameters to train dataset on.
    for split_position in split_positions :
      # Checks 10 minimum node size hyper-parameters to train dataset on.
      for min_node_size in min_node_sizes :
        # Create root node 
        root = Node( train_dataset, 0, min_node_size, split_position )
        
        # Begin recursive splitting. After this executes "root" 
        # will be the root of the constructed decision tree
        root.split()

        # Print details of tree. 
        root.printTree()

        # Initializing counters for calculation of validation 
        # metrics (recall, accuracy, etc.)
        false_negatives = 0
        false_positives = 0
        true_negatives  = 0
        true_positives  = 0

        # Iterates through validation set samples. Tallies 
        # number of true and false negatives and positives.
        for sample_row in samples : 

          # Predicted class from current decision tree.
          predicted_class = root.getClassification( sample_row[1:] )
          # Actual class value of the sample.
          actual_class = int( round( sample_row[0] ) )

          # If you predict it is POISONOUS: (positive)
          if ( predicted_class == 1 ) :
            # And it is POISONOUS (true)
            if ( actual_class == 1 ) :
              true_positives += 1
            # And it is EDIBLE
            else: 
              false_positives += 1

          # If you predict it is EDIBLE: (negative)
          else :
            # And it is EDIBLE (true)
            if( actual_class == 0 ) :
              true_negatives += 1 
            # And it is POISONOUS (false)
            else :
              false_negatives += 1

        # Calculates validation metrics for classifier.
        recall = ( true_positives * 1.0 ) / ( true_positives + false_negatives )
        precision = ( true_positives * 1.0 ) / ( true_positives + false_positives )
        f1_measure = ( 2.0 * true_positives ) / ( ( 2.0 * true_positives) + false_positives + false_negatives )
        accuracy = ( true_positives + true_negatives ) / ( samples.shape[0] * 1.0 )

        # If recall is higher than current max, reset best measures
        # and set the current hyperparameters as best.
        if ( recall >= best_recall ) :
          best_recall = recall
          best_f1_measure = f1_measure
          best_accuracy = accuracy
          best_precision = precision
          best_split_pos = split_position
          best_node_size = min_node_size

    # Prints raw counts of false and true negatives and positives.
    print( "True Positives:  %d"   %true_positives  )
    print( "False Positives: %d"   %false_positives )
    print( "True Negatives:  %d"   %true_negatives  )
    print( "False Negatives: %d\n" %false_negatives )

    # Prints validation metrics for classifier.
    print( "Recall:     %f" %best_recall     )
    print( "Precision:  %f" %best_precision  )
    print( "F1 Measure: %f" %best_f1_measure )
    print( "Accuracy:   %f" %best_accuracy   )
    print( "Min Node:   %f" %best_node_size  )
    print( "Split pos:  %f" %best_split_pos  )

####################
#       END        #
####################
