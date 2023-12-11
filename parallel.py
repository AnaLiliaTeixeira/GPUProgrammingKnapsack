import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import pandas as pd

# Definição do Kernel CUDA
kernel_code = """
#include <stdio.h>

__global__ void score_kernel(float* result, float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Certificar de que o índice está dentro dos limites do conjunto de dados
    if (idx < N) {
        // Avaliação da expressão para o conjunto de dados
        float a = data[idx];

        //for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        //line = line.replace(u, f"np.{u[:-1]}")

        ////////////////////////////////////////
        //esta versão é sequencial, só pra me basear por aqui
      //  for c in df.columns:
        //    line = line.replace(f"_{c}_", f"(df[\"{c}\"].values)")
        //a = eval(line)
        //b = df["y"]
        //e = np.square(np.subtract(a, b)).mean()
        //////////////////////////////

        // Cálculo da pontuação (erro médio quadrático)
        float expression_result = a;  // Substitua pela lógica da sua expressão
        float error = expression_result;
        result[idx] = error * error;

        // Sincronização para garantir que todos os threads tenham calculado o erro antes da redução
        __syncthreads();

        // Redução do erro usando somas em árvore
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                result[idx] += result[idx + stride];
            }
            // Sincronização a cada iteração do loop
            __syncthreads();
        }

        // O thread 0 de cada bloco armazena o resultado parcial na posição 0 do bloco
        if (threadIdx.x == 0) {
            result[blockIdx.x] = result[idx];
        }
    }
}

"""

# Compilação do Kernel CUDA
mod = SourceModule(kernel_code)
score_kernel = mod.get_function("score_kernel")

df = pd.read_csv("data.csv")

funs = [ line.strip() for line in open("functions.txt").readlines() ]

def parallel_score(data):
    # Tamanho do conjunto de dados
    N = len(data)

    # Alocar memória na GPU
    d_result = cuda.mem_alloc(N * 4)  # Tamanho em bytes para float

    # Transferir dados para a GPU
    #d_data = cuda.to_device(data.astype(np.float32))
    d_data = cuda.to_device(data.encode())
    #d_y = cuda.to_device(y.astype(np.float32))

   # Configuração do lançamento do kernel
   # block_size = 256
   # grid_size = (N + block_size - 1) // block_size

    # Executar o kernel na GPU
    score_kernel(d_result, d_data, block=(N, 1, 1), grid=(1, 1))
    #score_kernel(d_result, d_data, block=(block_size, 1, 1), grid=(grid_size, 1))

    # Transferir resultados da GPU de volta para a CPU
    result = np.empty(N, dtype=np.float32)
    cuda.memcpy_dtoh(result, d_result) # isto n é descessessário já que fazemos o return do result?

    # Libertar recursos na GPU
    d_result.free()

    return result

l = funs[0]
print(parallel_score(l) , l)
r = min([ (parallel_score(line), line) for line in funs ])
print(f"{r[0]} {r[1]}")
