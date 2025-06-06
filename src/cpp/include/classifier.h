#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <cmath>
#include <limits>
#include <string>

/**
 * Exact port of Classifier from Java to C++
 * 
 * This class provides utility methods for classification tasks.
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/Classifier.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class Classifier {
protected:
    /** the name of this classifier (useful for debugging) */
    std::string unique_name = "unnamed";

public:
    /** The way we represent missing values from within our implementation */
    static constexpr double MISSING_VALUE = std::numeric_limits<double>::quiet_NaN();
    
    /**
     * Is this value missing?
     *
     * @param x The value we wish to check if it is missing
     * @return True if the value is missing
     */
    static bool isMissing(double x) {
        return std::isnan(x);
    }
    
    /**
     * Set a unique name for this classifier
     * 
     * @param unique_name The unique name to set
     */
    void setUniqueName(const std::string& unique_name) {
        this->unique_name = unique_name;
    }
    
    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~Classifier() = default;
};

#endif // CLASSIFIER_H
