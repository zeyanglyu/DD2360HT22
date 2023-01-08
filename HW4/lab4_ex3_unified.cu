
#include <stdio.h>
#include <sys/time.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns)
{
  //@@ Insert code to implement matrix multiplication here
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= numBColumns) || (row >= numARows)) return;

    DataType tmpSum = 0.0;
    for (int k = 0; k < numAColumns; ++k) {
        tmpSum += A[row*numAColumns + k] * B[k*numBColumns + col];
    }
    C[row*numBColumns + col] = tmpSum;
    #if DEBUG
    printf("C[%d, %d] = %f\n", row, col, C[row*numBColumns + col]);
    #endif

}
double getTimer() 
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);
  numCRows = numARows;
  numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaMallocManaged(&hostA,numARows * numAColumns * sizeof(DataType));
  cudaMallocManaged(&hostB,numBRows * numBColumns * sizeof(DataType));
  cudaMallocManaged(&hostC,numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows; ++i) 
  {
        for (int j = 0; j < numAColumns; ++j) 
        {
            DataType randomNumber = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]
            hostA[i*numAColumns + j] = randomNumber;
            #if DEBUG
            printf("hostA[%d, %d] = %f\n", i, j, hostA[i*numBColumns + j]);
            #endif
        }
  }
  for (int i = 0; i < numBRows; ++i) 
  {
        for (int j = 0; j < numBColumns; ++j) 
        {
            DataType randomNumber = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]
            hostB[i*numBColumns + j] = randomNumber;
            #if DEBUG
            printf("hostA[%d, %d] = %f\n", i, j, hostB[i*numBColumns + j]);
            #endif
        }
  }
  for (int i = 0; i < numARows; i++)
  {
    for (int j = 0; j < numBColumns; j++)
    {
      resultRef[i * numBColumns + j] = 0.0;
      for (int k = 0; k < numAColumns; k++)
      {
        resultRef[i * numBColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      }
    }
  }
  //@@ Insert code below to allocate GPU memory here
    
  //@@ Insert code to below to Copy memory to the GPU here
  double start = getTimer();
  //@@ Initialize the grid and block dimensions here
    int Dbx = 32;
    int Dby = 32;
    int Dgx = (numCColumns + Dbx - 1) / Dbx;
    int Dgy = (numCRows + Dby - 1) / Dby;

  //@@ Launch the GPU Kernel here
  gemm <<<dim3(Dgx, Dgy, 1), dim3(Dbx, Dby, 1)>>>(hostA, hostB, hostC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  double Stop = getTimer() -start;
  printf(" Time: %f\n", Stop);
  //@@ Copy the GPU memory back to the CPU here

  //@@ Insert code below to compare the output with the reference
  int equality = 1;
    for (int i = 0; i < numCRows; ++i) {
        for (int j = 0; j < numCColumns; ++j) {
            if (fabs(hostC[i*numCColumns + j] - resultRef[i*numCColumns + j]) > 0.01) 
            { 
                equality = 0;
                #if DEBUG
                printf("Position: [%d, %d], Difference: %f\n", i, j, fabs(hostC[i*numCColumns + j] - resultRef[i*numCColumns + j]));
                #endif
                break;
            }
        }
    }
    if (equality == 1) {
        printf("Results are equal.\n");
    } else {
        printf("Results are NOT equal.\n");
    }

   
  cudaFreeHost(hostA);
  cudaFreeHost(hostB);
  cudaFreeHost(hostC);
    free(resultRef);

  return 0;
}
