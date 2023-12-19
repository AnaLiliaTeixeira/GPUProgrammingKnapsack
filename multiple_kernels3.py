import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

N = len(df)
funs = [line.strip() for line in open("functions.txt").readlines()]

NUM_ROWS = len(df)
print(NUM_ROWS)

# Número de threads por bloco
BLOCK_SIZE = 256

# Número de blocos
NUM_BLOCKS = (NUM_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE
print(NUM_BLOCKS)

# Definição do Kernel CUDA
def evaluate_line(line):
    for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        line = line.replace(u, f"np.{u[:-1]}")
    for c in df.columns:
        line = line.replace(f"_{c}_", f"(df[\"{c}\"].values)")
    return eval(line)

kernel_code = f""" 
__global__ void score_kernel(double* evaluatedLineArray, double* y, double* mean) 
{{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double linesEvaluated[{BLOCK_SIZE}];

    // Inicialização da variável compartilhada
    linesEvaluated[threadIdx.x] = 0.0;

    // Cada thread calcula a soma parcial para o bloco
    if (threadIdx.x < blockDim.x) {{
        linesEvaluated[threadIdx.x] += (evaluatedLineArray[threadIdx.x] - y[threadIdx.x]) *
                                       (evaluatedLineArray[threadIdx.x] - y[threadIdx.x]);
    }}
    __syncthreads();

    // Redução dentro do bloco para encontrar a soma para o bloco
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) {{
            linesEvaluated[threadIdx.x] += linesEvaluated[threadIdx.x + s];
        }}
    }}

    // Apenas a primeira thread de todos os blocos realiza a redução final global
    if (idx == 0) {{
        double globalMean = linesEvaluated[idx];

        // Redução entre os valores mínimos de cada bloco
        for (int i = 1; i < gridDim.x; i++) {{
            globalMean += linesEvaluated[i];
        }}

        // O resultado final é armazenado em mean[0]
        mean[0] = globalMean / {NUM_BLOCKS};
    }}

}}
"""

# Compilação do Kernel CUDA
score_kernel = SourceModule(kernel_code).get_function("score_kernel")

def parallel_score(line):
    # Criar host_evaluatedLineArray com tratamento de nan
    host_evaluatedLineArray = np.array(np.nan_to_num(evaluate_line(line)), dtype=np.double)
    host_y = np.array(df["y"], dtype=np.double)
    host_mean = np.zeros(1, dtype=np.double)

    # Alocar memória na GPU
    dev_evaluatedLineArray = cuda.to_device(host_evaluatedLineArray)
    dev_y = cuda.to_device(host_y)
    dev_mean = cuda.to_device(host_mean)

    # Executar o kernel na GPU
    score_kernel(dev_evaluatedLineArray, dev_y, dev_mean, block=(BLOCK_SIZE, 1, 1), grid=(NUM_BLOCKS, 1, 1))
    
    # Transferir resultados da GPU de volta para a CPU
    cuda.memcpy_dtoh(host_mean, dev_mean)
    print(f"Minimum kernel = ", host_mean[0])

    return host_mean[0]
r = min([parallel_score(line) for line in funs])
print(f"Minimum final = {r}")