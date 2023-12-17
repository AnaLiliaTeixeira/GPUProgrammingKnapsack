import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")
blocksNumber = 1024

# Definição do Kernel CUDA
def evaluate_line(line):
    for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        line = line.replace(u, f"np.{u[:-1]}")
    for c in df.columns:
        line = line.replace(f"_{c}_", f"(df[\"{c}\"].values)")
        #line = line.replace(f"_{c}_", f"({c}[idx])")

    return eval(line)

kernel_code = f"""
# include <stdio.h>

__global__ void score_kernel(float evaluatedLine, float minimum, float* y) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float linesEvaluated[{blocksNumber}];
    while (idx < {blocksNumber}) {{
        linesEvaluated[idx] = (evaluatedLine-y[idx]) * (evaluatedLine-y[idx]);
        idx += blockDim.x * gridDim.x;
    }}
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (linesEvaluated[threadIdx.x + s] < linesEvaluated[threadIdx.x]) {{
            linesEvaluated[threadIdx.x] = linesEvaluated[threadIdx.x + s];
        }}
        __syncthreads();
    }}

    if (threadIdx.x == 0) {{
        for (int i = 1; i < {blocksNumber}; i++) {{
            minimum += linesEvaluated[i];
        }}
        minimum /= {blocksNumber};
    }}


}}"""

    
# Compilação do Kernel CUDA
score_kernel = SourceModule(kernel_code).get_function("score_kernel")

funs = [ line.strip() for line in open("functions.txt").readlines() ]

def parallel_score(line):

    N = len(df)

    # criar input e output
    y = df["y"]
    y_np = np.array(df["y"].values, dtype=np.float32)
    #y_np = .astype(np.float32)
    #host = line
    #dev = np.zeros(N, np.float32) # output array com a media de todas as funções
    # faço o np.square(np.subtract(a, b)).mean() para cada um no gpu?
    
    # Avaliar a linha no host
    evaluated_line = np.array(evaluate_line(line), dtype=np.float32)
    #evaluated_line = evaluate_line(line)

    # Alocar memória na GPU para o host e para o dev
    #host_gpu = cuda.mem_alloc(4) #alocar o tamanho do input na memoria da gpu # 4 é o tamanho em bytes para float
    #dev_result = cuda.mem_alloc(N * 4) #alocar o tamanho do array output na memoria da gpu # 4 é o tamanho em bytes para float
    
    y_gpu = cuda.mem_alloc(N * 4) #alocar o tamanho do array output na memoria da gpu # 4 é o tamanho em bytes para float
    #host_result = np.empty(N, dtype=np.float32)
    minimum = cuda.mem_alloc(4)

    # Transferir dados para a GPU
    cuda.memcpy_htod(y_gpu, y_np) #copiar o input para a gpu
    #cuda.memcpy_htod(host_gpu, host)

    #grid_size = N // block_size
    gridSize = (N + blocksNumber - 1) // blocksNumber

    # Executar o kernel na GPU
    #score_kernel(d_result, d_data, block=(N, 1, 1), grid=(1, 1))
    score_kernel(evaluated_line, minimum, y_gpu, block=(blocksNumber, 1, 1), grid=(gridSize, 1))

    result = np.empty(1, dtype=np.float32)
    # Transferir resultados da GPU de volta para a CPU
    #cuda.memcpy_dtoh(host_result, dev_result)
    cuda.memcpy_dtoh(result, minimum)

    # Libertar recursos na GPU
    #host_gpu.free()
    #dev_result.free()
    minimum.free()

    return result

#l = funs[0]
#print(parallel_score(l) , l)
#r = min([(parallel_score(evaluate_line(line)), line) for line in funs])
r = [(parallel_score(line), line) for line in funs]
print(f"{r[0]} {r[1]}")
