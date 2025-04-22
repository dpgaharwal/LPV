/*
Steps to Install CUDA on Windows:
Install NVIDIA Drivers
Install CUDA Toolkit
nvcc --version


Set Up Your Development Environment
Open Visual Studio
Install the CUDA Toolkit
Create a new CUDA project:
    Open Visual Studio.
    Go to File > New > Project.
    Search for CUDA and choose a CUDA C++ project template.


On Visual Studio Code (With nvcc Compiler):
Open a terminal in VS Code (`Ctrl + ``).
Use nvcc to compile your .cu file:
    nvcc -o vector_addition vector_addition.cu
Run the compiled program:
    ./vector_addition

*/

// CUDA Parallelism

#include <iostream>
#include <limits>
#include <cuda.h>

#define N 1024  // Size of the array
#define TPB 256 // Threads per block

// Kernel for reduction (Min, Max, Sum)
template <typename T, typename Op>
__global__ void reduceKernel(int *input, T *output, Op operation, T identity)
{
    __shared__ T shared[TPB];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (idx < N) ? input[idx] : identity;
    __syncthreads();

    // Reduction inside block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] = operation(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = shared[0];
    }
}

// Host functions for operations
struct MinOp
{
    __device__ int operator()(int a, int b) { return min(a, b); }
};

struct MaxOp
{
    __device__ int operator()(int a, int b) { return max(a, b); }
};

struct SumOp
{
    __device__ int operator()(int a, int b) { return a + b; }
};

int main()
{
    int h_input[N], *d_input;
    int *d_output, h_output[N / TPB];

    // Initialize array
    for (int i = 0; i < N; i++)
        h_input[i] = rand() % 1000;

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, (N / TPB) * sizeof(int));

    // MIN reduction
    reduceKernel<<<N / TPB, TPB>>>(d_input, d_output, MinOp(), INT_MAX);
    cudaMemcpy(h_output, d_output, (N / TPB) * sizeof(int), cudaMemcpyDeviceToHost);
    int finalMin = h_output[0];
    for (int i = 1; i < N / TPB; i++)
        finalMin = min(finalMin, h_output[i]);
    std::cout << "Minimum: " << finalMin << std::endl;

    // MAX reduction
    reduceKernel<<<N / TPB, TPB>>>(d_input, d_output, MaxOp(), INT_MIN);
    cudaMemcpy(h_output, d_output, (N / TPB) * sizeof(int), cudaMemcpyDeviceToHost);
    int finalMax = h_output[0];
    for (int i = 1; i < N / TPB; i++)
        finalMax = max(finalMax, h_output[i]);
    std::cout << "Maximum: " << finalMax << std::endl;

    // SUM reduction
    reduceKernel<<<N / TPB, TPB>>>(d_input, d_output, SumOp(), 0);
    cudaMemcpy(h_output, d_output, (N / TPB) * sizeof(int), cudaMemcpyDeviceToHost);
    int finalSum = 0;
    for (int i = 0; i < N / TPB; i++)
        finalSum += h_output[i];
    std::cout << "Sum: " << finalSum << std::endl;

    // AVERAGE
    float avg = static_cast<float>(finalSum) / N;
    std::cout << "Average: " << avg << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

/*
// CPU Parallelism
#include <iostream>
#include <omp.h>
using namespace std;
int minval(int arr[], int n)
{
    int minval = arr[0];
#pragma omp parallel for reduction(min : minval)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] < minval)
            minval = arr[i];
    }
    return minval;
}
int maxval(int arr[], int n)
{
    int maxval = arr[0];
#pragma omp parallel for reduction(max : maxval)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] > maxval)
            maxval = arr[i];
    }
    return maxval;
}
int sum(int arr[], int n)
{
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }
    return sum;
}
int average(int arr[], int n)
{
    return (double)sum(arr, n) / n;
}
int main()
{
    int n = 5;
    int arr[] = {1, 2, 3, 4, 5};
    cout << "The minimum value is: " << minval(arr, n) << '\n';
    cout << "The maximum value is: " << maxval(arr, n) << '\n';
    cout << "The summation is: " << sum(arr, n) << '\n';
    cout << "The average is: " << average(arr, n) << '\n';
    return 0;
}
*/