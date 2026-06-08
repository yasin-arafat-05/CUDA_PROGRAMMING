/*
The Problem Statement: The Matrix Column Transpose
You are given a square matrix of floats of size 32x32. You want to load this matrix from global 
memory into Shared Memory, and then write it back to global memory in a transposed format 
(swapping rows and columns).To keep things simple and focused purely on shared memory, your
thread block size is exactly 32x32 threads (1024 threads total). This matches the matrix
dimensions perfectly.
- threadIdx.x represents the column of the matrix (0 to 31).
- threadIdx.y represents the row of the matrix (0 to 31).

**The Conflict Setup (The Naive Way):**
  In a naive implementation, you declare a shared memory 2D array: __shared__ float tile[32][32];
    - Step 1 (Coalesced Load): Each thread reads a value from global memory and stores it into 
      shared memory using its natural layout: tile[threadIdx.y][threadIdx.x] = global_in[...]; 
      (This part is fine!).

    - Step 2 (The Bank Conflict): To transpose the data when writing it back out, the threads
      read from shared memory by swapping their indices, meaning they read column-by-column 
      instead of row-by-row: float val = tile[threadIdx.x][threadIdx.y];


**Why this causes a massive Bank Conflict**
  Remember that shared memory has 32 banks, and successive 32-bit words (floats) are mapped to 
  successive banks.

  A warp in CUDA consists of 32 threads where threadIdx.y is constant and threadIdx.x spans from 
  0 to 31. When those threads attempt to read tile[threadIdx.x][threadIdx.y], all 32 threads in the 
  warp end up accessing the exact same memory bank at different offsets (a 32-way bank conflict!), 
  completely serializing the request.

  **Functionality Requirements** 
    - The Operation: Load a 32x32 matrix into a shared memory tile, and read it out in a 
      transposed layout.The Constraint: You must use shared memory to facilitate the transpose.
    - The Goal: Modify either the allocation size of the shared memory array or change the 
      indexing trick so that when the warp reads the data, all 32 threads land on different banks
      simultaneously (0 bank conflicts).

*/

#include<iostream>
using namespace std;


