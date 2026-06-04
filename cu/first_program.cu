#include<cmath>
#include<vector>
#include<random>
#include<chrono>
#include<iostream>
#include<cuda_runtime.h>
using namespace std;
__global__ void simpleMatMulKernel(float *d_M, float *d_N, float *d_P,int width){
    // maping 1D thread to 2D 
    // programmer prefer to map y with row.
    // theory(row,col) => coding(col,row)
   int  row = blockIdx.y * gridDim.y + threadIdx.y;
   int  col = blockIdx.x * gridDim.x + threadIdx.x;
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
    const long long totalElement = (long long) width*width;
    const size_t size = totalElement * sizeof(float);

    cout<<"Allocating Memory: "<< (size*3)/(1024*1024) << "MB Need to allocate from GPU."<<endl;

    // allocating memeory on the host device:
    vector<float> h_M(totalElement);
    vector<float> h_N(totalElement);
    vector<float> h_P_GPU(totalElement);
    vector<float> h_P_CPU(totalElement);

    // fill elemnet with rm number:
    fillWithRandom(h_M,totalElement);
    fillWithRandom(h_N,totalElement);

    // allocate memery in gpu:
    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M,size);
    cudaMalloc(&d_N,size);
    cudaMalloc(&d_P,size);

    // copy from host to device:
    cudaMemcpy(&d_M,h_M.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(&d_N,h_N.data(),size,cudaMemcpyHostToDevice);

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
}

/*
===========Output==========
first_program.cu
tmpxft_00000120_00000000-10_first_program.cudafe1.cpp
Allocating Memory: 12MB Need to allocate from GPU.
Time taken in GPU: 66.9367
Multiplication is on GPU complete
Time taken in CPU: 3283.21
*/
