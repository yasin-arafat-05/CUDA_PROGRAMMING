#include<string>
#include<math.h>
#include<iostream>
#include<cuda_runtime.h>
using namespace std;

int main(){
    int deviceCount = 0;
    cudaError_t  error = cudaGetDeviceCount(&deviceCount);

    if(error != cudaSuccess){
        cerr << "CUDA ERROR: " << cudaGetErrorString(error) << endl;
        return 1;
    }

    if(deviceCount==0){
        cout<<" No CUDA Device Found. " <<endl;
    }

    for(int i=0;i<deviceCount;i++){
        // get the device properties:
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);

        cout<< "=============================================================="<<endl;
        cout << "Device: "<< i  << " : "<< prop.name << endl;
        cout<< "=============================================================="<<endl;

        // Compute capability:
        cout<< "Compute Capability:         " << prop.major << " . " << prop.minor <<endl;

        // Warp and threads:
        cout<< "warp size                   "<< prop.warpSize << " Threads " << endl;
        cout<< "Max Thread Per Block        "<< prop.maxThreadsPerBlock << endl; 
        cout<< "Max Thread per SM           "<< prop.maxThreadsPerMultiProcessor <<endl;

        // Registers:
        cout<< "Register per block          "<< prop.regsPerBlock << " `each registers = 32-bit = 32/8 = 4 bytes` " << endl;
        cout<< "Register per  SM            "<< prop.regsPerMultiprocessor << endl; 

        // Shared Memory:
        cout<< "Max Shared Memory per Block "<< prop.sharedMemPerBlock / 1024 << "KB" <<endl;
        cout<< "Max Shared Memory per SM    "<< prop.sharedMemPerMultiprocessor/1024 << "KB" <<endl;

        // Global and Constant Memory: 
        cout<< "Total Global Memory         "<< prop.totalGlobalMem/(1024*1024*1024) <<"GB"<<endl;
        cout<< "Total Constant Memory       "<< prop.totalConstMem/(1024) << "KB"<<endl;
        
        
        // Grid and Block Dimention:
        cout<< "Max Block Dimention         "<< prop.maxThreadsDim[0] << " ," << prop.maxThreadsDim[1] << " ," << prop.maxThreadsDim[2]<<endl;
        cout<< "Max  Grid Dimention         "<< prop.maxGridSize[0] << " ," << prop.maxGridSize[1] << " ," << prop.maxGridSize[2] <<endl;


        cout<< endl;
        cout<< "=================================================================="<<endl;
        float val = (float) prop.maxThreadsPerMultiProcessor/prop.maxThreadsPerBlock;
        cout<< "Thread per Block: "<<prop.maxThreadsPerBlock<<" Thread per SM: "<<prop.maxThreadsPerMultiProcessor<<endl;
        cout<<"- "<< to_string(prop.maxThreadsPerMultiProcessor)<< "/"<<to_string(prop.maxThreadsPerBlock) << " = " << val <<endl;
        cout << "- One SM can run " << val << " block of threads and one block is: "<< prop.maxThreadsPerBlock << " threads. "<<endl;
        cout << "- One block consists of "<< prop.maxThreadsPerBlock << " threads "<<endl; 
        cout << "- All "<< prop.warpSize <<" threads in a warp execute the same instruction at the same time (this is called SIMT — Single Instruction, Multiple Threads)"<<endl;
    }
}


