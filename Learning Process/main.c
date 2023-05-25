#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};
#define train_count (sizeof(train) / sizeof(train[0]))
// 1 000 000 000 000 parameters used in GPT-4
// we are using 1

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w, float b)
{
    float result = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float x = train[i][0];
        float y = x * w + b;
        float d = y - train[i][1];
        result += d * d;
    }
    result /= train_count;
    return result;
}

int main()
{
    // srand(time(0));
    srand(69);
    printf("%lu, %lu, %lu\n", train_count, sizeof(train[0]), sizeof(train));
    // y = x * w;
    float w = rand_float() * 10.0f;
    float b = rand_float() * 5.0f;
    float eps = 1e-3;
    float rate = 1e-3;
    for (size_t i = 0; i < 6000; i++) {
        float dw = (cost(w + eps, b) - cost(w, b)) / eps;
        float db = (cost(w, b + eps) - cost(w, b)) / eps;
        w -= rate * dw;
        b -= rate * db;
        printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);
    }
    printf("%f\n", w);
    
    return (0);
}