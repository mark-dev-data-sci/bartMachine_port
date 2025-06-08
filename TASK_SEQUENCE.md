bartMachine Port - Detailed Task Sequence

## Task Breakdown Strategy
- Each task focuses on 3-5 related methods/functions
- Class structure established first, then method groups
- RNG dependencies respected throughout
- Each task includes validation tests

## Phase 1: RNG Foundation (CRITICAL FIRST)

### Task 1.1: MersenneTwisterFast - Class Structure
**Objective**: Port MersenneTwisterFast Java class structure to C++
**Files**:
- `src/cpp/include/exact_port_mersenne_twister.h`
- `src/cpp/exact_port_mersenne_twister.cpp`
**Methods**: Class structure, constructor, destructor, basic state management
**Validation**: Can instantiate, basic state operations work

### Task 1.2: MersenneTwisterFast - Core RNG Methods
**Objective**: Port core random number generation methods
**Methods**:
- `setSeed(long seed)`
- `nextDouble(boolean includeZero, boolean includeOne)`
- Internal state update methods
**Validation**: Identical sequences for same seeds vs Java implementation

### Task 1.3: StatToolbox RNG Interface
**Objective**: Port StatToolbox RNG wrapper methods
**Methods**:
- `setSeed(long seed)`
- `rand()` - wrapper for nextDouble(false, false)
**Validation**: StatToolbox.rand() produces identical sequences to Java

## Phase 2: Pure Mathematical Functions (RNG-Independent)

### Task 2.1: StatToolbox - Class Structure
**Objective**: Port StatToolbox class structure and constants
**Methods**:
- Class structure and constants
- Basic RNG interface
**Validation**: Can instantiate, basic operations work

### Task 2.2: StatToolbox - sample_average Methods
**Objective**: Port sample_average methods
**Methods**:
- `sample_average()` (all overloads)
**Validation**: Exact numerical equivalence on test datasets

### Task 2.3: StatToolbox - sample_median Method
**Objective**: Port sample_median method
**Methods**:
- `sample_median()`
**Validation**: Exact numerical equivalence on test datasets

### Task 2.4: StatToolbox - Min/Max Methods
**Objective**: Port min/max calculation methods
**Methods**:
- `sample_minimum()` (all overloads)
- `sample_maximum()` (all overloads)
**Validation**: Exact numerical equivalence, handle edge cases

### Task 2.5: StatToolbox - Variance Methods
**Objective**: Port variance calculation methods
**Methods**:
- `sample_standard_deviation()` (all overloads)
- `sample_variance()` (all overloads)
- `sample_sum_sq_err()` (all overloads)
**Validation**: Exact numerical equivalence, handle edge cases

### Task 2.6: StatToolbox - Utility Functions
**Objective**: Port remaining mathematical utility functions
**Methods**:
- `FindMaxIndex()`
- Any other pure mathematical functions
**Validation**: Exact results on test cases

## Phase 3: RNG-Dependent Statistical Functions

### Task 3.1: StatToolbox - Sampling Functions
**Objective**: Port random sampling functions (requires RNG)
**Methods**:
- `sample_from_inv_gamma(double k, double theta)`
- `sample_from_norm_dist(double mu, double sigsq)`
- `multinomial_sample(TIntArrayList vals, double[] probs)`
**Validation**: Identical sampling sequences for same seeds

## Phase 4: Tree Structure (Partially RNG-Dependent)

### Task 4.1: bartMachineTreeNode - Class Structure + Basic Methods
**Objective**: Port tree node class structure and non-random methods
**Methods**:
- Class structure, constructors, destructors
- Basic tree navigation (getParent, getLeftChild, getRightChild)
- Tree property methods (isLeaf, isStump, depth calculations)
- Data management (non-random operations)
**Validation**: Can create trees, navigate structure correctly

### Task 4.2: bartMachineTreeNode - Tree Manipulation (Non-Random)
**Objective**: Port deterministic tree operations
**Methods**:
- Tree cloning and copying
- Data propagation methods
- Tree structure queries
- Non-random tree modifications
**Validation**: Tree operations produce identical structures

### Task 4.3: bartMachineTreeNode - Random Operations
**Objective**: Port random tree operations (requires RNG)
**Methods**:
- `pickRandomSplitValue()`
- `pickRandomDirectionForMissingData()`
- Random node selection methods
**Validation**: Identical random choices for same seeds

## Phase 5: MCMC Engine (Heavily RNG-Dependent)

### Task 5.1: bartMachine Base Classes - Structure
**Objective**: Port base class hierarchy structure
**Classes**:
- `bartMachine_a_base` - basic structure and instance variables
- `bartMachine_b_hyperparams` - hyperparameter management
**Validation**: Can instantiate base classes, manage parameters

### Task 5.2: Gibbs Sampler - Basic Framework
**Objective**: Port basic Gibbs sampling framework
**Classes**:
- `bartMachine_e_gibbs_base` - basic Gibbs structure
- `bartMachine_f_gibbs_internal` - internal Gibbs methods
**Methods**: Basic sampling framework, parameter updates
**Validation**: Basic Gibbs steps work correctly

### Task 5.3: Metropolis-Hastings - Grow Operation
**Objective**: Port MH grow operation exactly
**Methods**:
- `doMHGrowAndCalcLnR()`
- `calcLnTransRatioGrow()`
- `calcLnLikRatioGrow()`
- `calcLnTreeStructureRatioGrow()`
- `pickGrowNode()`
**Validation**: Identical grow proposals and acceptance rates

### Task 5.4: Metropolis-Hastings - Prune Operation
**Objective**: Port MH prune operation exactly
**Methods**:
- `doMHPruneAndCalcLnR()`
- `calcLnTransRatioPrune()`
- Related prune calculation methods
- `pickPruneNodeOrChangeNode()`
**Validation**: Identical prune proposals and acceptance rates

### Task 5.5: Metropolis-Hastings - Change Operation
**Objective**: Port MH change operation exactly
**Methods**:
- `doMHChangeAndCalcLnR()`
- `calcLnLikRatioChange()`
- Related change calculation methods
**Validation**: Identical change proposals and acceptance rates

### Task 5.6: Metropolis-Hastings - Integration
**Objective**: Complete MH integration and step selection
**Methods**:
- `metroHastingsPosteriorTreeSpaceIteration()`
- `randomlyPickAmongTheProposalSteps()`
- Complete MH workflow
**Validation**: Full MH chains identical to Java implementation

## Phase 6: Complete Integration

### Task 6.1: Complete bartMachine Classes
**Objective**: Port remaining bartMachine classes
**Classes**:
- `bartMachine_c_debug`
- `bartMachine_d_init`
- `bartMachine_h_eval`
- `bartMachine_i_prior_cov_spec`
**Validation**: Complete class hierarchy works

### Task 6.2: Regression and Classification
**Objective**: Port specialized bartMachine implementations
**Classes**:
- `bartMachineRegression`
- `bartMachineClassification`
- Multi-threaded variants
**Validation**: Regression and classification produce identical results

## Phase 7: Code Review and Fixes

### Task 7.0: Setup for Code Review
**Objective**: Establish framework for systematic code review
**Methods**:
- Create structured template for documenting code discrepancies
- Set up tracking system for identified issues
- Establish consistent methodology for comparing Java and C++ code
**Validation**: Review framework is in place and ready to use

### Task 7.1: Random Number Generation Review
**Objective**: Review random number generation implementation
**Methods**:
- Compare `ExactPortMersenneTwister.java` with `exact_port_mersenne_twister.cpp`
- Review usage of random number generator in all files
- Document all discrepancies
**Validation**: Comprehensive report of RNG discrepancies

### Task 7.2: Random Number Generation Fixes
**Objective**: Fix random number generation implementation
**Methods**:
- Fix `ExactPortMersenneTwister` implementation
- Fix initialization of pre-computed random arrays
- Verify random number generation fixes
**Validation**: Identical random number sequences between Java and C++

### Task 7.3: Core Algorithm Review
**Objective**: Review core algorithm implementations
**Methods**:
- Review tree building algorithms
- Review MCMC sampling algorithms
- Review prediction algorithms
- Document all discrepancies
**Validation**: Comprehensive report of algorithm discrepancies

### Task 7.4: Core Algorithm Fixes
**Objective**: Fix core algorithm implementations
**Methods**:
- Fix tree building algorithms
- Fix MCMC sampling algorithms
- Fix prediction algorithms
- Verify core algorithm fixes
**Validation**: Identical algorithm behavior between Java and C++

### Task 7.5: Data Structure and Memory Management Review
**Objective**: Review data structures and memory management
**Methods**:
- Review data structures
- Review memory management
- Document all discrepancies
**Validation**: Comprehensive report of data structure and memory management discrepancies

### Task 7.6: Data Structure and Memory Management Fixes
**Objective**: Fix data structures and memory management
**Methods**:
- Fix data structures
- Fix memory management
- Verify data structure and memory management fixes
**Validation**: Identical data structure behavior between Java and C++

### Task 7.7: Hardcoded Values Review
**Objective**: Identify all hardcoded values
**Methods**:
- Search for magic numbers
- Check for hardcoded dimensions
- Document all hardcoded values
**Validation**: Comprehensive report of hardcoded values

### Task 7.8: Hardcoded Values Fixes
**Objective**: Replace hardcoded values with dynamic calculations
**Methods**:
- Remove hardcoded dimensions
- Replace magic numbers with named constants
- Verify hardcoded values fixes
**Validation**: All values are dynamically calculated based on input data

### Task 7.9: Missing Functionality Review
**Objective**: Identify missing functionality
**Methods**:
- Compare method signatures between Java and C++
- Check for missing methods or classes
- Document all missing functionality
**Validation**: Comprehensive report of missing functionality

### Task 7.10: Missing Functionality Implementation
**Objective**: Implement missing functionality
**Methods**:
- Add missing methods or classes
- Complete any incomplete implementations
- Verify missing functionality implementation
**Validation**: All functionality from Java implementation is present in C++

### Task 7.11: Comprehensive Verification
**Objective**: Verify all fixes with comprehensive test suite
**Methods**:
- Create comprehensive test suite
- Run comprehensive test suite
- Create final report
**Validation**: C++ implementation produces identical results to Java implementation

## Phase 8: C++ Validation via R Interface

### Task 8.1: R-to-C++ Bridge Implementation
**Objective**: Create R bindings for C++ implementation
**Files**:
- Rcpp interface files
- Modified R bartMachine package
**Validation**: Can call C++ functions from R

### Task 8.2: Validation with Original Datasets
**Objective**: Compare C++ implementation with Java using original R interface
**Methods**:
- Run identical workflows with both implementations
- Compare results for numerical equivalence
- Document any discrepancies
**Validation**: Identical results between Java and C++ backends

### Task 8.3: Comprehensive Validation Suite
**Objective**: Create comprehensive test suite for ongoing validation
**Methods**:
- Automated test suite for all key functions
- Edge case testing
- Performance benchmarking
**Validation**: Complete test coverage of all functionality

## Phase 9: Python API Layer

### Task 9.1: Python-C++ Bindings
**Objective**: Create Python bindings for C++ classes
**Files**: Python binding layer using pybind11 or similar
**Validation**: Can call C++ functions from Python

### Task 9.2: Python bartMachine API - Core Functions
**Objective**: Port core R bartMachine functions to Python
**Functions**:
- `bartMachine()` constructor
- Basic prediction methods
- Parameter setting methods
**Validation**: Python API produces identical results to R

### Task 9.3: Python bartMachine API - Advanced Functions
**Objective**: Port advanced R bartMachine functions
**Functions**:
- `calc_credible_intervals()`
- `calc_prediction_intervals()`
- `bart_machine_get_posterior()`
- Diagnostic functions
**Validation**: All R functionality available in Python

## Phase 10: Documentation and Release

### Task 10.1: Documentation
**Objective**: Create comprehensive documentation
**Files**:
- API reference
- User guide
- Installation instructions
**Validation**: Documentation is complete and accurate

### Task 10.2: Examples and Tutorials
**Objective**: Create examples and tutorials
**Files**:
- Jupyter notebooks
- R and Python examples
- Tutorial documents
**Validation**: Examples and tutorials are clear and functional

### Task 10.3: Final Testing and Release
**Objective**: Final testing and release preparation
**Methods**:
- Cross-platform testing
- Performance benchmarking
- Package preparation
**Validation**: Package is ready for release

## Task Execution Guidelines

### Before Each Task:
1. Read CONSTRAINTS.md
2. Read VALIDATION_STRATEGY.md
3. Identify specific Java/R code to port
4. Create tests first (where possible)

### During Each Task:
- Translate line-by-line where possible
- Preserve all comments and logic flow
- No improvements or optimizations
- Test continuously

### After Each Task:
- Update PROGRESS.md with completion status
- Document any issues in ISSUES.md
- Validate numerical equivalence
- Get approval before proceeding

This sequence ensures RNG foundation is solid before building dependent components, validates the C++ port thoroughly through comprehensive code review and fixes, and then proceeds with the R and Python interfaces.

## Note on Task Sequence Updates
This task sequence was updated on June 7, 2025 to include a dedicated code review and fixes phase (Phase 7) before proceeding with the R interface validation. This change was made based on validation findings that indicated significant discrepancies between the Java and C++ implementations. See VALIDATION_STRATEGY.md for detailed information on the validation approach and code_review_task_breakdown.md for a granular breakdown of the code review and fixes tasks.
