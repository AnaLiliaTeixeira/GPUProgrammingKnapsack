import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

N = len(df)
funs = [line.strip() for line in open("functions.txt").readlines()]

NUM_ROWS = len(funs)
NUM_COLUMNS = len(df) - 1

# Número de threads por bloco
BLOCK_SIZE = 256

# Número de blocos
NUM_BLOCKS = (NUM_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE

# Definição do Kernel CUDA
kernel_code = f""" 
__global__ void score_kernel(double** evaluatedLineArray, double* y, double* minimum) 
{{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double linesEvaluated[{BLOCK_SIZE}];
    linesEvaluated[threadIdx.x] = 0.0;

    // Cada thread calcula o valor mínimo para o bloco
    if (threadIdx.x < {BLOCK_SIZE}) {{
        for (int j = 0; j < {NUM_COLUMNS}; j++) {{
            linesEvaluated[threadIdx.x] += (evaluatedLineArray[threadIdx.x][j] - y[j]) * (evaluatedLineArray[threadIdx.x][j] - y[j]);
        }}

        linesEvaluated[threadIdx.x] /= {NUM_COLUMNS};
    }}

    __syncthreads();

    // Redução dentro do bloco para encontrar o valor mínimo
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) {{
            if (linesEvaluated[threadIdx.x + s] < linesEvaluated[threadIdx.x]) {{
                linesEvaluated[threadIdx.x] = linesEvaluated[threadIdx.x + s];
            }}
        }}
    }}
    __syncthreads();

    // A primeira thread do bloco armazena o valor mínimo para o bloco
    if (threadIdx.x == 0) {{
        linesEvaluated[blockIdx.x] = linesEvaluated[0];
    }}

    // Apenas a primeira thread de todos os blocos realiza a redução final
    if (idx == 0) {{

        // Redução entre os valores mínimos de cada bloco
        for (int i = 1; i < gridDim.x; i++) {{
            if (minimum[i] < minimum[0]) {{
                minimum[0] = minimum[i];
            }}
        }}
    }}
}}
"""

# Compilação do Kernel CUDA
score_kernel = SourceModule(kernel_code).get_function("score_kernel")

def evaluate_line(line):
    for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        line = line.replace(u, f"np.{u[:-1]}")
    for c in df.columns:
        line = line.replace(f"_{c}_", f"(df[\"{c}\"].values)")
    return eval(line)
    
def parallel_score():
    # Criar host_evaluatedLineArray com tratamento de valores Nan
    host_evaluatedLineArray = np.array(np.nan_to_num([evaluate_line(line) for line in funs]), dtype=np.double)
    host_y = np.array(df["y"], dtype=np.double)
    host_minimum = np.zeros(1, dtype=np.double)

    # Alocar memória na GPU e transferir os dados para a GPU
    dev_evaluatedLineArray = cuda.to_device(host_evaluatedLineArray)
    dev_y = cuda.to_device(host_y)
    dev_minimum = cuda.to_device(host_minimum)

    # Executar o kernel na GPU
    score_kernel(dev_evaluatedLineArray, dev_y, dev_minimum, block=(BLOCK_SIZE, 1, 1), grid=(NUM_BLOCKS, 1, 1))

    # Transferir resultados da GPU de volta para a CPU
    cuda.memcpy_dtoh(host_minimum, dev_minimum)

    return host_minimum[0]

r = parallel_score()
print(f"Minimum of all functions = {r}")
