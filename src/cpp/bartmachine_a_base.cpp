#include "include/bartmachine_a_base.h"
#include "include/bartmachine_tree_node.h"

/**
 * Exact port of bartMachine_a_base from Java to C++
 * 
 * The base class for any BART implementation. Contains
 * mostly instance variables and settings for the algorithm
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_a_base.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */

void bartmachine_a_base::FlushData() {
    // Remove unnecessary data from the Gibbs chain to conserve RAM
    for (int i = 0; i < num_gibbs_total_iterations; i++) {
        FlushDataForSample(gibbs_samples_of_bart_trees[i]);
    }
}

void bartmachine_a_base::FlushDataForSample(bartMachineTreeNode** bart_trees) {
    // Remove unnecessary data from an individual Gibbs sample
    for (int t = 0; t < num_trees; t++) {
        if (bart_trees[t] != nullptr) {
            bart_trees[t]->flushNodeData();
        }
    }
}

void bartmachine_a_base::StopBuilding() {
    // Must be implemented, but does nothing in the base class
}

void bartmachine_a_base::setThreadNum(int threadNum) {
    this->threadNum = threadNum;
}

void bartmachine_a_base::setVerbose(bool verbose) {
    this->verbose = verbose;
}

void bartmachine_a_base::setTotalNumThreads(int num_cores) {
    this->num_cores = num_cores;
}

void bartmachine_a_base::setMemCacheForSpeed(bool mem_cache_for_speed) {
    this->mem_cache_for_speed = mem_cache_for_speed;
}

void bartmachine_a_base::setFlushIndicesToSaveRAM(bool flush_indices_to_save_ram) {
    this->flush_indices_to_save_ram = flush_indices_to_save_ram;
}

void bartmachine_a_base::setNumTrees(int m) {
    this->num_trees = m;
}

void bartmachine_a_base::setData(std::vector<double*>& X_y) {
    n = X_y.size();
    
    // In the Java implementation, p is determined as the length of each row minus 1
    // (since the last element is the response variable)
    if (n > 0 && X_y[0] != nullptr) {
        // Count the number of elements in the first row
        int row_length = 0;
        double* first_row = X_y[0];
        while (first_row[row_length] != 0 || row_length == 0) {
            row_length++;
            // Safety check to avoid infinite loop
            if (row_length > 1000) break;
        }
        p = row_length - 1;
    } else {
        p = 0;
    }
}
