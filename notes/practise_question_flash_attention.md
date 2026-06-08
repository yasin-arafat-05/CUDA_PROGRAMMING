
### Problem 1: Fast Row-wise Reduction (Max & Sum) <br>
Before you do softmax, you need to master reduction. In CUDA, calculating a single maximum or sum across a row requires threads to talk to each other.
 * The Problem: Given a massive $M \times N$ matrix, compute the maximum value of each row, and the sum of the exponentials of each row.
 * Why it matters for FlashAttention: FlashAttention relies heavily on tracking row-wise maximums $(m_i)$ and normalization sums $(l_i)$ across blocks.
 * Key Concepts to Master: Block-wide reductions, warp shuffles (__shfl_down_sync), and avoiding bank conflicts in shared memory.

### Problem 2: Numerically Stable Safe Softmax Kernel <br>
Standard $softmax (e^{x_i} / \sum e^{x_j})$ will overflow float16/float32 data types instantly if $x_i$ is large (like in attention scores). You must make it "safe" by subtracting the row maximum.
 * The Problem: Write a single kernel that takes an $M \times N$ matrix, finds the row max, subtracts it from every element in that row, computes $e^{x_i - max}$, sums them up, and divides to get the final probabilities.
 * Why it matters: Standard attention materializes this massive softmax matrix in global memory. FlashAttention avoids this, but you *must* understand the baseline physics of this bottleneck first.
 * Challenge: Do this in a single pass per row without constantly reading/writing back to Global Memory.

### Problem 3: The Naive Attention Kernel $(Fused QK^T + \text{Softmax} + PV)$ <br>
Now, put your matrix multiplication and softmax kernels together, but fuse them into a single pipeline.
 * The Problem: Given matrices Q ($N \times D$), K ($N \times D$), and V ($N \times D$), write a sequence of kernels (or one global wrapper) that computes S = QK^T, applies your safe softmax to get $P = \text{softmax}(S)$, and computes O = PV.
 * Why it matters: This is the exact baseline algorithm. By profiling this, you will physically see the memory bottleneck: storing S and P (which are $N \times N$) in global memory is incredibly slow. This pain is exactly why FlashAttention was invented!

### Problem 4: Online Safe Softmax (The "Aha!" Moment)
In standard softmax, you need to see *all* elements in a row to find the true maximum before you can compute the sum. FlashAttention blocks the inputs, meaning it only sees a fraction of the row at a time.
 * The Problem: Write a kernel that receives data in chunks. Calculate the softmax running statistics (m_i and l_i) *incrementally*. If you read block A, find its max, and then read block B and find a *new* higher max, you must rescale your previous sum on the fly without restarting.
 * Mathematical Formula to Implement: 
 * Why it matters: This is the core mathematical trick of FlashAttention. If you can code this online rescaling kernel, you have solved 80% of the conceptual hurdle of FlashAttention.
 
### Step 5: FlashAttention
Once you complete Problem 4, you are ready. You will combine your Tiled Matrix Multiplication from earlier with this Online Safe Softmax loop. You will load blocks of Q, K, and V into Shared Memory, update your running scale factors on the fly, and write the final output directly to Global Memory without ever saving the massive intermediate $N \times N$ attention matrix.
