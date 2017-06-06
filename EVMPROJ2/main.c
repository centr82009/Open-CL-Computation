//
//  main.c
//  EVMProj
//
//  Created by Mazeev Roman on 23.05.16.
//  Copyright © 2016 Mazeev Roman. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#include <time.h>

// Кол-во элементов массива
#define DATA_SIZE (524288)

// Вычисление квадрата массива
const char *KernelSource = "                                            \n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"                                                                       \n";

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    int err;
    
    float data[DATA_SIZE];              // Начальный массив
    float results[DATA_SIZE];           // Конечный массив
    
    size_t global;                      // Глобальный размер для вычисления
    size_t local;                       // Локальный размер для вычисления
    
    cl_device_id device_id;             // Id устройства
    cl_context context;                 // Контекст
    cl_command_queue commands;          // Команды
    cl_program program;                 // Программа
    cl_kernel kernel;                   // Ядро
    
    cl_mem input;                       // Количнство памяти устройства для входного массива
    cl_mem output;                      // Количнство памяти устройства для выходного массива
    
    // Заполняем память случайными числами с плав. точкой и выводим 1й эллемент массива
    long int start_time =  clock();
    
    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++)
        data[i] = rand() / (float)RAND_MAX;
    printf("Первый эллемент начального массива %f\n",data[1]);
    
    // Выбор на чем выполнять вычисления
    int ss = 1;
    printf("%s\n", "Введите 1 для вычисления на CPU, 2 для вычисления на GPU");
    int type;
    scanf("%d", &type);
    if (type == 1)
        err = clGetDeviceIDs(NULL, ss ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    else if (type == 2)
        err = clGetDeviceIDs(NULL, ss ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    else return 0;
    if (err != CL_SUCCESS)
    {
        printf("Ошибка1\n");
        return EXIT_FAILURE;
    }
    
    //Создание контекста
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Ошибка2\n");
        return EXIT_FAILURE;
    }
    
    // Создание очереди команд
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Ошибка3\n");
        return EXIT_FAILURE;
    }
    
    // Создание объекта программы
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Ошибка4\n");
        return EXIT_FAILURE;
    }
    
    // Компиляция
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Ошибка5\n");
    }
    
    // Создаем ядро
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Ошибка6\n");
        exit(1);
    }
    
    // Создание входящих и выходящих массивов в памяти видеокарты (Буферами)
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Ошибка7\n");
        exit(1);
    }
    
    // Запись команд в буфер
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Ошибка8\n");
        exit(1);
    }
    
    // Устанавливаем значения аргументов
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Ошибка9 %d\n", err);
        exit(1);
    }
    
    // Возвращение информации о ядре
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Ошибка10 %d\n", err);
        exit(1);
    }
    
    // Выполняем ядро
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Ошибка11\n");
        return EXIT_FAILURE;
    }
    
    // Ожидание конца выполнения команд
    clFinish(commands);
    
    // Чтение буфера
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Ошибка12 %d\n", err);
        exit(1);
    }
    
    // Результат
    printf("Первый элемент конечного массива %f\n",results[1]);
    
    // Завершение и чистка памяти
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    long int end_time = clock();
    long int search_time = end_time - start_time;
    printf("%ld Тактов\n\n",search_time);

    return 0;
}
