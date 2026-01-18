#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

// Размер массива (1 000 000 элементов по условию задания)
const int N = 1'000'000;

// Множитель для поэлементной обработки
const float MULTIPLIER = 2.5f;

/*
Функция для измерения времени выполнения CUDA-ядра.
Используется механизм cudaEvent, который позволяет точно
измерять время выполнения операций на GPU.
*/
float measureTime(void (*kernel)(float*, float*, int),
                  float* d_in, float* d_out,
                  int n, dim3 grid, dim3 block) {

    // Создаём CUDA-события
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запоминаем момент начала выполнения ядра
    cudaEventRecord(start);

    // Запуск CUDA-ядра
    kernel<<<grid, block>>>(d_in, d_out, n);

    // Запоминаем момент окончания выполнения ядра
    cudaEventRecord(stop);

    // Ждём завершения ядра
    cudaEventSynchronize(stop);

    // Вычисляем прошедшее время в миллисекундах
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Освобождаем ресурсы
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

/*
 
TASK 1 — GLOBAL MEMORY
Поэлементное умножение массива
Используется ТОЛЬКО глобальная память GPU
 
*/
__global__ void multiply_global(float* in, float* out, int n) {

    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n) {
        // Чтение из глобальной памяти и запись в глобальную память
        out[idx] = in[idx] * MULTIPLIER;
    }
}

/*
 
TASK 1 — SHARED MEMORY
Поэлементное умножение с использованием shared memory
 
*/
__global__ void multiply_shared(float* in, float* out, int n) {

    // Разделяемая память для одного блока
    // Размер равен количеству потоков в блоке (256)
    __shared__ float shared[256];

    // Глобальный индекс элемента
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Загружаем элемент из глобальной памяти в shared memory
        shared[threadIdx.x] = in[idx];

        // Синхронизация всех потоков блока
        __syncthreads();

        // Выполняем вычисления в shared memory
        shared[threadIdx.x] *= MULTIPLIER;

        // Записываем результат обратно в глобальную память
        out[idx] = shared[threadIdx.x];
    }
}

/*
 
TASK 2 — VECTOR ADDITION
Поэлементное сложение двух массивов
C[i] = A[i] + B[i]
 
*/
__global__ void vector_add(float* a, float* b, float* c, int n) {

    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка границ
    if (idx < n) {
        // Поэлементное сложение
        c[idx] = a[idx] + b[idx];
    }
}

/*
 
TASK 3 — COALESCED MEMORY ACCESS
Коалесцированный доступ к глобальной памяти
 

Соседние потоки обращаются к соседним адресам памяти,
что позволяет GPU объединять обращения в одну транзакцию.
*/
__global__ void coalesced(float* in, float* out, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

/*
 
TASK 3 — NON-COALESCED MEMORY ACCESS
Некоалесцированный доступ к глобальной памяти
 

Индекс умышленно искажается (stride = 32),
что приводит к разрозненным обращениям к памяти
и снижению производительности.
*/
__global__ void non_coalesced(float* in, float* out, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Искусственно плохой паттерн доступа
    int bad_index = (idx * 32) % n;

    if (bad_index < n) {
        out[bad_index] = in[bad_index] * 2.0f;
    }
}

int main() {

    cout << "CUDA Assignment 3\n";

    // Размер массива в байтах
    size_t bytes = N * sizeof(float);

    // Выделяем память на хосте и инициализируем данные
    vector<float> h_a(N, 1.0f), h_b(N, 2.0f);

    // Указатели на память GPU
    float *d_a, *d_b, *d_c;

    // Выделение памяти на устройстве
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Конфигурация запуска (оптимальная)
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    /*
    
    TASK 1
    
    */
    cout << "\nTask 1: Global vs Shared Memory\n";

    float t_global = measureTime(multiply_global, d_a, d_c, N, grid, block);
    float t_shared = measureTime(multiply_shared, d_a, d_c, N, grid, block);

    cout << "Global memory time: " << t_global << " ms\n";
    cout << "Shared memory time: " << t_shared << " ms\n";

    /*
    
    TASK 2
    
    */
    cout << "\nTask 2: Block size influence\n";

    int block_sizes[] = {128, 256, 512};

    for (int bs : block_sizes) {

        dim3 b(bs);
        dim3 g((N + bs - 1) / bs);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        vector_add<<<g, b>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cout << "Block size " << bs << ": " << ms << " ms\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    /*
    
    TASK 3
    
    */
    cout << "\nTask 3: Coalesced vs Non-coalesced access\n";

    float t_coal = measureTime(coalesced, d_a, d_c, N, grid, block);
    float t_non = measureTime(non_coalesced, d_a, d_c, N, grid, block);

    cout << "Coalesced access time: " << t_coal << " ms\n";
    cout << "Non-coalesced access time: " << t_non << " ms\n";

    /*
    
    TASK 4
    
    */
    cout << "\nTask 4: Grid and block optimization\n";

    // Неоптимальная конфигурация
    dim3 bad_block(64);
    dim3 bad_grid((N + bad_block.x - 1) / bad_block.x);

    float t_bad = measureTime(coalesced, d_a, d_c, N, bad_grid, bad_block);
    float t_opt = measureTime(coalesced, d_a, d_c, N, grid, block);

    cout << "Unoptimized configuration time: " << t_bad << " ms\n";
    cout << "Optimized configuration time: " << t_opt << " ms\n";

    // Освобождение памяти
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
