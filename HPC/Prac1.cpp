#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;
// Function to generate a random array
void generateArray(vector<int> &arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = rand() % 10000; // Random numbers between 0-9999
    }
}
// Sequential Bubble Sort
void bubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
// Parallel Bubble Sort using OpenMP
void parallelBubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
// Merge function for Merge Sort
void merge(vector<int> &arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;
    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }
    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}
// Sequential Merge Sort
void mergeSort(vector<int> &arr, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
// Parallel Merge Sort using OpenMP
void parallelMergeSort(vector<int> &arr, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;
#pragma omp parallel sections
        {
#pragma omp section
            parallelMergeSort(arr, left, mid);
#pragma omp section
            parallelMergeSort(arr, mid + 1, right);
        }
        merge(arr, left, mid, right);
    }
}
// Function to measure execution time
template <typename Func>
double measureTime(Func func, vector<int> &arr)
{
    auto start = high_resolution_clock::now();
    func(arr);
    auto stop = high_resolution_clock::now();
    return duration<double, milli>(stop - start).count();
}
int main()
{
    srand(time(0));
    int size = 1000; // Change size for testing
    vector<int> arr(size);
    // Bubble Sort Comparison
    generateArray(arr, size);
    vector<int> arrCopy = arr;
    cout << "Sequential Bubble Sort Time: " << measureTime(bubbleSort, arrCopy) << " ms\n";
    generateArray(arr, size);
    arrCopy = arr;
    cout << "Parallel Bubble Sort Time: "
         << measureTime(parallelBubbleSort, arrCopy) << " ms\n";
    // Merge Sort Comparison
    generateArray(arr, size);
    arrCopy = arr;
    cout << "Sequential Merge Sort Time: "
         << measureTime([&](vector<int> &a)
                        { mergeSort(a, 0, size - 1); }, arrCopy)
         << " ms\n";
    generateArray(arr, size);
    arrCopy = arr;
    cout << "Parallel Merge Sort Time: "
         << measureTime([&](vector<int> &a)
                        { parallelMergeSort(a, 0, size - 1); }, arrCopy)
         << " ms\n";
    return 0;
}
