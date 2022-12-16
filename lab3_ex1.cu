
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#define DataType double

#define TPB 32

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
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  
  hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
  
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
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  double CLOCK_MEM = StartClock();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double CLOCK_STOP_MEM = StopClock(CLOCK_MEM);

  //@@ Initialize the 1D grid and block dimensions here
  int Db = 128;
  int Dg = (inputLength + Db - 1) / Db;

  //@@ Launch the GPU Kernel here
  double CLOCK_START_GPU = StartClock();
  vecAdd<<<Dg,Db>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);
  cudaDeviceSynchronize();
  double CLOCK_STOP_GPU = StopClock(CLOCK_START_GPU);
  //@@ Copy the GPU memory back to the CPU here
  double START_MEM_2=StartClock();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double CLOCK_STOP_MEM_2= StopClock(START_MEM_2);
  //@@ Insert code below to compare the output with the reference
   for(int i=0;i<inputLength;i++)
  {
    if(resultRef[i] != hostOutput[i] && abs(resultRef[i]-hostOutput[i])>0.001 )
    {
        printf("Error counting numbers: %f",abs(resultRef[i]-hostOutput[i]) );
        return 0;
    }
  }

  printf("sum verified: Correct!\n");
  printf("Time Host->Device: %f - Time Device->Host: %f\n",CLOCK_STOP_MEM,CLOCK_STOP_MEM_2);
  printf("CPU time: %f - GPU time: %f\n",StopCPU,CLOCK_STOP_GPU);
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);
  return 0;
}
