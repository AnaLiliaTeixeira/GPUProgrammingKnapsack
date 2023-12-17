import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import pandas as pd

# Definição do Kernel CUDA
kernel_code = """
#include <stdio.h>

__global__ void score_kernel(float* result, char* line, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        char* evaluated_line = {evaluate_line(line)};
        result[idx] = evaluated_line;
    }
}

"""

def evaluate_line(line) :
    for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        line = line.replace(u, f"np.{u[:-1]}")
    for c in df.columns:
        line = line.replace(f"_{c}_", f"(df[\"{c}\"].values)")
    return line
    
a = eval(line)
b = df["y"]
e = np.square(np.subtract(a, b)).mean()

# Compilação do Kernel CUDA
score_kernel = SourceModule(kernel_code).get_function("score_kernel")

df = pd.read_csv("data.csv")
funs = [ line.strip() for line in open("functions.txt").readlines() ]

def parallel_score(l):

    N = len(df)

    # Alocar memória na GPU
    d_result = cuda.mem_alloc(N * np.float32)  # 4 é o tamanho em bytes para float

    # Transferir dados para a GPU
    #d_data = cuda.to_device(data.astype(np.float32))
    line = cuda.to_device(l.encode())
    #d_y = cuda.to_device(y.astype(np.float32))

    block_size = 1024
    #grid_size = N // block_size
    grid_size = (N + block_size - 1) // block_size

    # Executar o kernel na GPU
    #score_kernel(d_result, d_data, block=(N, 1, 1), grid=(1, 1))
    score_kernel(d_result, line, block=(block_size, 1, 1), grid=(grid_size, 1))

    # Transferir resultados da GPU de volta para a CPU
    result = np.empty(N, dtype=np.float32)
    cuda.memcpy_dtoh(result, d_result)

    # Libertar recursos na GPU
    d_result.free()

    return result

l = funs[0]
print(parallel_score(l) , l)
r = min([ (parallel_score(line), line) for line in funs ])
print(f"{r[0]} {r[1]}")
