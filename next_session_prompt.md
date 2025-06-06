Continue implementing the Metropolis-Hastings change operation for the bartMachine port from Java to C++. We're working on the task-5-5-mh-change branch and need to implement Task 5.5 as outlined in the task sequence.

Please refer to:
1. CONSTRAINTS.md for the exact porting requirements
2. PORTING_CHECKLIST.md for step-by-step guidance on exact porting
3. PORTING_GUIDELINES.md for detailed implementation workflow
4. TASK_SEQUENCE.md for this task's context in the overall project
5. task_5_4_progress_summary.md for our current progress (we're at ~60-70% completion)
6. task_5_5_prompt.md for the specific task requirements
7. task_5_5_plan.md for the implementation strategy
8. tests/test_task_5_5.cpp for the test cases to validate the implementation
9. The original Java source at /Users/mark/Documents/Cline/bartMachine/src/bartMachine/ for reference

Focus on implementing the following methods in bartmachine_g_mh.cpp:
- doMHChangeAndCalcLnR()
- calcLnLikRatioChange()
- Related change calculation methods

## Implementation Approach
1. For each method, first show the original Java code in a code block
2. Then implement the C++ version with line-by-line comments referencing the Java code
3. Explicitly go through the PORTING_CHECKLIST.md before, during, and after implementation
4. Focus on exact translation rather than improvements or optimizations

Remember that this is an exact port, so the C++ implementation should match the Java implementation's behavior precisely.
