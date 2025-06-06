#include <iostream>
#include <cassert>
#include "src/cpp/include/bartmachine_tree_node.h"

/**
 * Test for Task 4.1: bartMachineTreeNode - Class Structure + Basic Methods
 * 
 * This test validates the implementation of the bartMachineTreeNode class structure
 * and its basic non-random methods.
 * 
 * The test ensures that:
 * 1. The class can be instantiated with different constructors
 * 2. Basic tree navigation methods work correctly
 * 3. Tree property methods return expected values
 * 4. Basic tree manipulation works correctly
 */

int main() {
    std::cout << "Testing Task 4.1: bartMachineTreeNode - Class Structure + Basic Methods" << std::endl;
    
    // Test constructors
    std::cout << "\nTesting constructors..." << std::endl;
    
    // Default constructor
    bartMachineTreeNode* root = new bartMachineTreeNode();
    assert(root != nullptr);
    assert(root->isLeaf);
    assert(root->parent == nullptr);
    assert(root->left == nullptr);
    assert(root->right == nullptr);
    assert(root->depth == 0);
    
    // Test basic tree navigation methods
    std::cout << "Testing basic tree navigation methods..." << std::endl;
    
    // Create a simple tree structure
    bartMachineTreeNode* left_child = new bartMachineTreeNode(root);
    bartMachineTreeNode* right_child = new bartMachineTreeNode(root);
    
    root->setLeaf(false);
    root->setLeft(left_child);
    root->setRight(right_child);
    
    assert(!root->isLeaf);
    assert(root->getLeft() == left_child);
    assert(root->getRight() == right_child);
    assert(left_child->parent == root);
    assert(right_child->parent == root);
    assert(left_child->depth == 1);
    assert(right_child->depth == 1);
    
    // Test tree property methods
    std::cout << "Testing tree property methods..." << std::endl;
    
    assert(!root->isStump());  // Not a stump because it has children
    assert(left_child->isStump() == false);  // Not a stump because it has a parent
    
    // Create a standalone node to test isStump
    bartMachineTreeNode* stump_node = new bartMachineTreeNode();
    assert(stump_node->isStump());
    
    // Test deepestNode
    assert(root->deepestNode() == 1);  // Both children are leaves, so deepestNode returns 1 + max(0, 0) = 1
    
    // Add another level to the tree
    bartMachineTreeNode* left_left_child = new bartMachineTreeNode(left_child);
    left_child->setLeaf(false);
    left_child->setLeft(left_left_child);
    
    assert(root->deepestNode() == 2);  // Now the deepest node calculation is 1 + max(1, 0) = 2
    
    // Test numLeaves
    assert(root->numLeaves() == 2);  // Two leaf nodes: right_child and left_left_child
    
    // Test numNodesAndLeaves
    assert(root->numNodesAndLeaves() == 4);  // root, left_child, right_child, left_left_child
    
    // Test numPruneNodesAvailable
    assert(root->numPruneNodesAvailable() == 1);  // Only left_child can be pruned (has two leaf children)
    
    // Test tree manipulation methods
    std::cout << "Testing tree manipulation methods..." << std::endl;
    
    // Test pruneTreeAt
    bartMachineTreeNode::pruneTreeAt(left_child);
    assert(left_child->isLeaf);
    assert(left_child->left == nullptr);
    assert(left_child->right == nullptr);
    
    // Test clone
    std::cout << "Testing clone method..." << std::endl;
    
    bartMachineTreeNode* cloned_root = root->clone();
    assert(cloned_root != nullptr);
    assert(cloned_root != root);
    assert(cloned_root->isLeaf == root->isLeaf);
    assert(cloned_root->depth == root->depth);
    assert(cloned_root->getLeft() != nullptr);
    assert(cloned_root->getRight() != nullptr);
    
    // Test stringLocation
    std::cout << "Testing stringLocation method..." << std::endl;
    
    std::string root_location = root->stringLocation();
    std::string left_location = left_child->stringLocation();
    std::string right_location = right_child->stringLocation();
    
    std::cout << "Root location: " << root_location << std::endl;
    std::cout << "Left child location: " << left_location << std::endl;
    std::cout << "Right child location: " << right_location << std::endl;
    
    assert(root_location == "P");
    assert(left_location == "PL");
    assert(right_location == "PR");
    
    // Clean up
    delete root;
    delete stump_node;
    delete cloned_root;
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
