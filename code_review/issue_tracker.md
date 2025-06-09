# bartMachine C++ Port - Issue Tracker

## Overview
This document tracks all discrepancies identified during the code review of the bartMachine C++ port. It serves as a central registry for all issues, their status, priority, and dependencies.

## Issue Status Definitions
- **Identified**: The discrepancy has been identified but not yet analyzed in detail
- **Analyzed**: The discrepancy has been analyzed, and a fix has been proposed
- **Fixed**: The fix has been implemented
- **Verified**: The fix has been verified to resolve the discrepancy

## Priority Level Definitions
- **Critical**: Discrepancy affects core functionality or numerical equivalence; must be fixed immediately
- **High**: Discrepancy has significant impact on functionality or performance; should be fixed soon
- **Medium**: Discrepancy has moderate impact; should be fixed but not urgent
- **Low**: Discrepancy has minimal impact; can be fixed when convenient

## Issue Registry

| Issue ID | Component | Description | Status | Priority | Dependencies | Assigned To | Due Date |
|----------|-----------|-------------|--------|----------|--------------|-------------|----------|
| DISC-001 | Example | Example discrepancy | Identified | Medium | None | | |
| DISC-002 | Random Number Generation | Naming discrepancy and incomplete implementation of MersenneTwisterFast | Identified | Critical | None | | |

## Critical Issues (Priority: Critical)

### DISC-002: Random Number Generation - Naming discrepancy and incomplete implementation
- **Component**: Random Number Generation
- **Description**: The original Java implementation is called `MersenneTwisterFast` but the C++ port is named `ExactPortMersenneTwister`. Additionally, many methods in the C++ implementation are incomplete with placeholder implementations.
- **Status**: Identified
- **Priority**: Critical
- **Dependencies**: None

## High Priority Issues (Priority: High)

*No high priority issues identified yet.*

## Medium Priority Issues (Priority: Medium)

*No medium priority issues identified yet.*

## Low Priority Issues (Priority: Low)

*No low priority issues identified yet.*

## Issue Dependencies

*No dependencies identified yet.*

## Recently Resolved Issues

*No resolved issues yet.*

## Notes
- This document should be updated whenever a new discrepancy is identified or the status of an existing discrepancy changes.
- Issues should be added to the appropriate priority section as well as the main registry.
- Dependencies between issues should be documented in the Dependencies section.
