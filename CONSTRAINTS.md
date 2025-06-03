# BARTMACHINE PORT CONSTRAINTS - READ FIRST EVERY SESSION

## REPOSITORY LOCATIONS (CRITICAL)
- Original Repository: /Users/mark/Documents/Cline/bartMachine
- Port Repository: /Users/mark/Documents/Cline/bartMachine_port
- Original Java Source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine
- Original R Source: /Users/mark/Documents/Cline/bartMachine/bartMachine

## DUAL PORT REQUIREMENT
- Java classes → C++ classes (computational engine)  
- R functions → Python functions (user API)
- BOTH must be implemented - never just one

## TRANSLATION RULES
- NO improvements or modernizations
- NO missing functionality  
- NO placeholder methods
- Line-by-line translation where possible
- Preserve all comments and logic flow

## NUMERICAL EQUIVALENCE
- Identical random number generation (port MersenneTwister exactly)
- Same algorithm execution order
- Bit-for-bit identical results where possible

## VALIDATION
- Use existing R and Java tests as foundation
- Add tests for cross-language equivalence
- Test intermediate values, not just final outputs
- Ensure ALL methods and overloads from original Java implementation are covered in tests
- Verify that tests check for exact numerical equivalence with original implementation
- Include edge case handling that matches original implementation behavior

## RNG DEPENDENCIES (CRITICAL)
- MersenneTwisterFast must be ported FIRST
- All random sampling functions depend on exact RNG equivalence
- StatToolbox.rand() is used throughout the codebase
- Tree operations use random selection for splits and directions
- MCMC uses random draws for acceptance/rejection

## VERSION CONTROL
- Update the GitHub repository after completing each numbered task
- Use descriptive commit messages that clearly explain the changes made
- Push changes to the remote repository to ensure work is backed up
- Include test results in commit messages when applicable

## NEVER FORGET
This is an EXACT PORT - not an improvement project
Every algorithm must produce identical results to original
