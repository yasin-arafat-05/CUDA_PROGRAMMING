/*
=============================Problem Statement=============================
The Problem Statement: The Boundary Split

You are given a 1D array of floats of size N. You need to modify this array in-place.
The catch is that the operation you perform depends entirely on whether the element's 
index is even or odd. Because adjacent threads handle adjacent indices, a naive approach 
will cause every single thread in a warp to diverge from its neighbor.Your goal is to write a 
CUDA kernel (and any necessary host-side preparation) that achieves the correct final output 
array, but structures the execution so that threads within the same warp do not diverge.

Functionality Requirements
        **For Even Indices (Compute A):**
            You need to simulate a heavy transcendental calculation.For an element at an even index, 
            replace its value with the sine of the value, raised to the power of 2.5.
                newValue = powf(sinf(oldValue), 2.5f)

        **For Odd Indices (Compute B):** 
            You need to simulate a different heavy calculation. For an element at an odd index, 
            replace its value with the cosine of the value, multiplied by the square root of the value.
                newValue = cosf(oldValue) * sqrtf(oldValue)
            
The ConstraintsThe final output array in global memory must match the original 
index order (i.e., index 0 has the Compute A result, index 1 has the Compute B result, index 2 
has Compute A, etc.).You cannot change the math formulas.You must minimize or 
eliminate warp divergence during the execution of these two distinct mathematical paths.
*/
