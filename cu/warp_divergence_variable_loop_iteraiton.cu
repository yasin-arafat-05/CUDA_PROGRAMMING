/*
=============================Problem Statement=============================
The problem statement, Variable Loop Iteration (The Row-Length Escalation)

The Problem StatementYou are given a 1D array of floats representing a matrix of size 
NxN, flattened in row-major order. You need to calculate a running sum for a portion of each row.
The number of elements you need to sum in any given row is equal to that row's index+1.
For example:
    - Row 0 needs a sum of 1 element.
    - Row 1 needs a sum of 2 elements.
    - Row 31 needs a sum of 32 elements.
If you map one thread to one row, Thread 0 finishes its loop instantly, while Thread 31 takes 32 
times longer. Because they are in the same warp, Threads 0 through 30 will sit completely idle, 
stalled by Thread 31's long loop.Your goal is to write a solution where the workload is 
distributed evenly across the threads in a warp so that no threads are left idling while others 
finish long loops.

Functionality RequirementsThe Operation: 
- For row(R), calculate the sum of the first R+1 elements of that row. 
- The Output: Store the final sum back into the first element of that row in the
  matrix(matrix[R * N]).
- The Goal: Restructure the execution logic so that all threads within a warp execute a highly 
  uniform number of loop iterations.

*/

