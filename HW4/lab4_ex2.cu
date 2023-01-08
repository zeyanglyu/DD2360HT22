
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda.h>

#define DataType double
#define TPB 64

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) 
{
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i< len)
     out[i] = in1[i] + in2[i];
}

//@@ Insert code to implement timer start
double StartClock() 
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
//@@ Insert code to implement timer stop
 double StopClock(double startime) 
 {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return (((double)tp.tv_sec + (double)tp.tv_usec*1.e-6) - startime);
}
DataType RandomNUM(DataType low, DataType high)
{
  DataType R;
  R = (DataType)rand() / ((DataType)RAND_MAX +1);
  return (low + R*(high - low)); 
}

int main(int argc, char **argv) 
{
  
  int inputLength;
  int nStream;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutputAsync;
  DataType *resultRef;
  DataType *deviceInput1Async;
  DataType *deviceInput2Async;
  DataType *deviceOutputAsync;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  nStream = atoi(argv[2]);

  printf("The input length is %d\n", inputLength);
  printf("nstream is %d\n", nStream);

  

  cudaStream_t stream[nStream];
  for(int i=0;i<nStream;++i)
    cudaStreamCreate(&stream[i]);

  //@@ Insert code below to allocate Host memory for input and output
  
  hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostOutputAsync = (DataType *)malloc(inputLength * sizeof(DataType));
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for(int i=0;i<inputLength;i++)
    {
      hostInput1[i]=RandomNUM(0,1);
      hostInput2[i]=RandomNUM(0,1);
    }
 resultRef = (DataType *)malloc(inputLength *sizeof *resultRef);

 double CLOCK_STRAT = StartClock();

    for(int i=0;i<inputLength;i++)
  {
    resultRef[i]=hostInput1[i]+hostInput2[i];
  }
   double StopCPU = StopClock(CLOCK_STRAT);


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1Async, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2Async, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutputAsync, inputLength * sizeof(DataType));
  //@@ Insert code to below to Copy memory to the GPU here
  
  
  int streamSize = inputLength/nStream;
  //@@ Initialize the 1D grid and block dimensions here
  int Dg, Db;
  Dg = (streamSize + TPB - 1) / TPB;
  Db = TPB;

  //@@ Launch the GPU Kernel here
  //double CLOCK_START_GPU = StartClock();
  //double CLOCK_STOP_GPU = StopClock(CLOCK_START_GPU);
  //@@ Copy the GPU memory back to the CPU here
  //double START_MEM_2=StartClock();
  double CLOCK_Async = StartClock();
  int StreamSize  = inputLength / nStream;
  int StreamByte  = StreamSize * sizeof(DataType);
  for(int i=0;i<nStream;++i)
  {
    int offset=i*StreamSize;
    cudaMemcpyAsync(&deviceInput1Async[offset], &hostInput1[offset], StreamByte, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(&deviceInput2Async[offset], &hostInput2[offset], StreamByte, cudaMemcpyHostToDevice,stream[i]);
    vecAdd<<<Dg,Db,0,stream[i]>>>(&deviceInput1Async[offset],&deviceInput2Async[offset],&deviceOutputAsync[offset],StreamSize);
    cudaMemcpyAsync(&hostOutputAsync[offset], &deviceOutputAsync[offset], StreamByte, cudaMemcpyDeviceToHost,stream[i]);
  }
  for(int i=0; i<nStream; i++)
		cudaStreamSynchronize(stream[i]);
  double CLOCK_STOP_Async= StopClock(CLOCK_Async);
  //@@ Insert code below to compare the output with the reference
   for(int i=0;i<inputLength;i++)
  {
    if(hostOutputAsync[i] !=  resultRef[i] && abs( resultRef[i]-hostOutputAsync[i])>0.001 )
    {
        printf("Error counting numbers: %f",abs( resultRef[i]-hostOutputAsync[i]) );
        return 0;
    }
  }

  printf("sum verified: Correct!\n");
  printf(" Async time: %f\n",CLOCK_STOP_Async);
  //@@ Free the GPU memory here
  cudaFree(deviceInput1Async);
  cudaFree(deviceInput2Async);
  cudaFree(deviceOutputAsync);
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutputAsync);
  return 0;
}
