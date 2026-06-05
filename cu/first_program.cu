#include<cmath>
#include<vector>
#include<random>
#include<chrono>
#include<iostream>
#include<cuda_runtime.h>
using namespace std;

// -----------------------------Simple Matrix Multiplication------------------------
__global__ void simpleMatMulKernel(float *d_M, float *d_N, float *d_P,int width){
    // maping 1D thread to 2D 
    // programmer prefer to map y with row.
    // theory(row,col) => coding(col,row)
   int  row = blockIdx.y * blockDim.y + threadIdx.y;
   int  col = blockIdx.x * blockDim.x + threadIdx.x;
   if(row<width && col<width){
        float productValue = 0;
        for(int k=0;k<width;k++){
            // d_M[row-major] * d_N[column-major]
            productValue += d_M[row*width+k] * d_N[k*width+col];
        }
        // save the element: d_p[row major]
        d_P[row*width+col] = productValue;
   }
}

// -------------------------------Coalesced-MatMul-------------------------
/*
let, 
    - row = 5,
    - width = 5,
    k = 1,2,3,4,5,...

    => For d_M[row*width+k] <=
    row*width+k = 26,27,28,29,30,.....

    => For d_N[k*width+col] <=
    k*width+col = 10,15,20,25,30,.....

Conclusion: d_M memory access is coalesced but d_M is not.
*/
__global__ void coalescedMatMul(float *d_M,float *d_N, float *d_P, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<width && col<width){
        float productValue = 0;
        for(int k=0;k<width;k++){
            productValue += d_M[row*width+k] * d_N[col*width+k];
        }
        d_P[row*width+col] = productValue;
    }

}


// -----------------------Matrix Multiplication in CPU----------------------
__host__ void matmulOnHost(float *A, float *B,float *ans, int width){
    for(int row=0;row<width;row++){
        for(int col=0;col<width;col++){
            float sum=0.0f;
            for(int k=0;k<width;k++){
                sum += A[row*width+k]  * B[k*width+col];
            }
            ans[row*width+col] = sum;
        }
    }
    
}


// ------------------------fillWithRandom Value----------------------------
__host__ void fillWithRandom(vector<float> &v,long long totalElement){
    // 1. define random seed
    random_device rd;
    // 2. initilize the mersenne_twister_engine for rm generation 
    mt19937 gen(rd());
    // 3. take random number from uniform distribution:
    uniform_real_distribution<float> ud(0.0f,1.0f);
    //fill value:
    for(int i=0;i<totalElement;i++){
        v[i] = ud(gen);
    }
}



int main(){
    const int width = 1024;
    //const int width = 3;
    const long long totalElement = (long long) width*width;
    const size_t size = totalElement * sizeof(float);

    cout<<"Allocating Memory: "<< (size*3)/(1024*1024) << "MB Need to allocate from GPU."<<endl;

    // allocating memeory on the host device:
    /*

    // mat1:
     1 2 3 
     4 5 6 
     7 8 9 

    // mat2:
     9 8 7 
     6 5 4 
     3 2 1 
    
     // ans: 
     30 24 18 
     84 69 54
     138 114 90
    */

    // for testing that my kernel funtion is correct.
    // vector<float> h_M={1,2,3,4,5,6,7,8,9};
    // vector<float> h_N={9,8,7,6,5,4,3,2,1};


    vector<float> h_M(totalElement);
    vector<float> h_N(totalElement);
    vector<float> h_P_GPU(totalElement);
    vector<float> h_P_CPU(totalElement);

    // fill elemnet with rm number:
    // fillWithRandom(h_M,totalElement);
    // fillWithRandom(h_N,totalElement);

    // allocate memery in gpu:
    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M,size);
    cudaMalloc(&d_N,size);
    cudaMalloc(&d_P,size);

    // copy from host to device:
    cudaMemcpy(d_M,h_M.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_N,h_N.data(),size,cudaMemcpyHostToDevice);

    // define block and grid:
    dim3 threadPerBlock(16,16,1);
    // we don't use ceill(it's slower than this trick)
    dim3 blockPerGrid((width+threadPerBlock.x-1)/threadPerBlock.x,
    (width+threadPerBlock.y-1)/threadPerBlock.y);

    // start the counter to measure time:
    auto start = chrono::high_resolution_clock::now();

    // launch the kernel:
    simpleMatMulKernel<<<blockPerGrid,threadPerBlock>>>(d_M,d_N,d_P,width);
    cudaDeviceSynchronize();

    // stop the counter:
    auto stop = chrono::high_resolution_clock::now();

    // calculate the time duration:
    chrono::duration<float,milli> duration = stop - start;
    cout<<"Time taken in GPU: "<< duration.count()<<endl;

    // copy result back to device to host:
    cudaMemcpy(h_P_GPU.data(),d_P,size,cudaMemcpyDeviceToHost);
    cout<<"Multiplication is on GPU complete"<<endl;
    // Print results
    // cout << "\n=== GPU Result ===\n";
    // for(int i = 0; i < width; i++){
    //     for(int j = 0; j < width; j++){
    //         cout << h_P_GPU[i*width + j] << "\t";
    //     }
    //     cout << endl;
    // }

    // ----------------For coalesced mat mul---------------
    // transpose the matrix:
    vector<float> h_N_coal(totalElement);
    for(int i=0;i<width;i++){
        for(int j=0;j<width;j++){
            h_N_coal[j*width+i] = h_N[i*width+j];
        }
    }
    cudaMemcpy(d_N,h_N_coal.data(),size,cudaMemcpyHostToDevice);
    start = chrono::high_resolution_clock::now();
    coalescedMatMul<<<blockPerGrid,threadPerBlock>>>(d_M,d_N,d_P,width);
    stop = chrono::high_resolution_clock::now();
    duration = stop - start;
    cout<<"Time taken in GPU (coalesced version): "<< duration.count()<<endl; 
    
    // copy result back to device to host:
    cudaMemcpy(h_P_GPU.data(),d_P,size,cudaMemcpyDeviceToHost);
    cout<<"Multiplication is on GPU complete: coalesced version"<<endl;
    // Print results
    // cout << "\n=== GPU Result ===\n";
    // for(int i = 0; i < width; i++){
    //     for(int j = 0; j < width; j++){
    //         cout << h_P_GPU[i*width + j] << "\t";
    //     }
    //     cout << endl;
    // }


    // Clean the GPU:
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);


    // ------------Do multiplication of cpu-----------------
    auto start1 = chrono::high_resolution_clock::now();
    matmulOnHost(h_M.data(),h_N.data(),h_P_CPU.data(),width);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<float,milli> duration1 = end1 - start1;
    cout<<"Time taken in CPU: "<<duration1.count()<<endl;
    // cout << "\n=== CPU Result ===\n";
    // for(int i = 0; i < width; i++){
    //     for(int j = 0; j < width; j++){
    //         cout << h_P_CPU[i*width + j] << "\t";
    //     }
    //     cout << endl;
    // }
}


