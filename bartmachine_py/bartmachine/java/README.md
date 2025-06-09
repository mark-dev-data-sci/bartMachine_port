# Java Bridge for bartMachine

This directory contains the Java components required for the bartMachine Python package. The Java bridge allows the Python implementation to call the same Java methods as the R implementation, ensuring identical behavior and numerical results.

## Contents

- `bart_java.jar`: The main JAR file containing the bartMachine Java implementation
- `commons-math-2.1.jar`: Apache Commons Math library dependency
- `fastutil-core-8.5.8.jar`: FastUtil library dependency
- `trove-3.0.3.jar`: Trove library dependency
- `src/`: Source code for the Java wrapper classes

## BartMachineWrapper

The `BartMachineWrapper` class is a wrapper around the bartMachine Java implementation that exposes protected methods as public methods, allowing the Python code to access them through Py4J.

## Usage

The Java bridge is used internally by the Python implementation and should not be used directly by end users. The Python API provides a more convenient interface for using the bartMachine functionality.

## Implementation Details

The Java bridge uses Py4J to communicate between Python and Java. It handles:

1. JVM initialization and management
2. Data conversion between Python and Java
3. Method invocation on Java objects
4. Error handling and propagation

## Key Java Methods

The following Java methods are used by the Python implementation:

### Model Building
- `addTrainingDataRow`: Add a row of training data
- `finalizeTrainingData`: Finalize the training data
- `Build`: Build the BART model

### Model Configuration
- `setNumCores`: Set the number of cores to use
- `setNumTrees`: Set the number of trees
- `setNumGibbsBurnIn`: Set the number of burn-in iterations
- `setNumGibbsTotalIterations`: Set the total number of iterations
- `setAlpha`, `setBeta`, `setK`, `setQ`, `setNU`: Set prior parameters
- `setProbGrow`, `setProbPrune`: Set MH proposal step probabilities
- `setVerbose`: Set verbose output
- `setSeed`: Set random seed

### Prediction
- `getGibbsSamplesForPrediction`: Get posterior samples for prediction

### Variable Importance
- `getAttributeProps`: Get variable importance measures

### Node Information
- `getNodePredictionTrainingIndicies`: Get node prediction training indices
- `getProjectionWeights`: Get projection weights
- `extractRawNodeInformation`: Extract raw node information

### Interaction Constraints
- `intializeInteractionConstraints`: Initialize interaction constraints
- `addInteractionConstraint`: Add an interaction constraint

### Debugging
- `writeStdOutToLogFile`: Write standard output to log file
- `printTreeIllustations`: Print tree illustrations

## Extending the Java Bridge

To add support for additional Java methods:

1. Add the method to the `BartMachineWrapper` class if it's protected in the original implementation
2. Add a wrapper function in `java_bridge.py` that calls the Java method
3. Add appropriate data conversion code
4. Add error handling
5. Add tests for the new functionality
