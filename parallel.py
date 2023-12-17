import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")
blocksNumber = 256
N = len(df)

# Definição do Kernel CUDA
def evaluate_line(line):
    for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        line = line.replace(u, f"np.{u[:-1]}")
    #line = line.replace("sqrtf", f"SAFESQRT")
    for c in df.columns:
        line = line.replace(f"_{c}_", f"(df[\"{c}\"].values)")
        #line = line.replace(f"_{c}_", f"({c}[idx])")

    return eval(line)

#define SAFE_SQRT(x) ((x) >= 0 ? sqrt(x) : 0)

kernel_code = f""" 

__global__ void score_kernel(float** evaluatedLineArray, float* y, float* minimum) 
{{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float linesEvaluated[{blocksNumber}];

    while (idx < {blocksNumber}) {{
        for (int j = 0; j < {N}; j++) {{
            linesEvaluated[idx] += (evaluatedLineArray[idx][j] - y[j]) * (evaluatedLineArray[idx][j] - y[j]);
        }}

        linesEvaluated[idx] /= {N};
        idx += blockDim.x * gridDim.x;  // Atualize idx para o próximo bloco
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
            minimum[0] += linesEvaluated[i];
        }}
        minimum[0] /= {blocksNumber};
    }}
}}
"""

# Compilação do Kernel CUDA
score_kernel = SourceModule(kernel_code).get_function("score_kernel")

funs = [line.strip() for line in open("functions.txt").readlines()]

def parallel_score():

    # Criar host_evaluatedLineArray com tratamento de nan
    host_evaluatedLineArray = np.array(np.nan_to_num([evaluate_line(line) for line in funs], dtype=np.float32))
    host_y = np.array(df["y"], dtype=np.float32)
    host_minimum = np.zeros(1, dtype=np.float32)
    
    # for i in host_evaluatedLineArray:
    #   print(i)
    #print(host_evaluatedLineArray.shape[0])
    #print(host_evaluatedLineArray.shape[1])
    #print(len(host_evaluatedLineArray) * np.float32().nbytes)

    # Alocar memória na GPU

      # Get information about the allocated memory
#     mem_info = cuda.mem_get_info()

#    print("Total free memory before allocating mm::", mem_info.free_memory)
#    print("Free memory in L1 memory pool:", mem_info.l1_free_memory)
#    print("Free memory in L2 memory pool:", mem_info.l2_free_memory)
#    print("Free memory in shared memory pool:", mem_info.shared_memory_free)
#    print("Alignment requirement for float32:", cuda.align_val(np.float32()))


    dev_evaluatedLineArray = cuda.mem_alloc(host_evaluatedLineArray.shape[0] * host_evaluatedLineArray.shape[1] * np.float32(1).nbytes)
    dev_y = cuda.mem_alloc(N * np.float32(1).nbytes) #alocar o tamanho do array output na memoria da gpu # 4 é o tamanho em bytes para float
    dev_minimum = cuda.mem_alloc(np.float32(1).nbytes)

#    mem_info = cuda.mem_get_info()
    
    # Transferir dados para a GPU
    cuda.memcpy_htod(dev_evaluatedLineArray, host_evaluatedLineArray) #copiar o input para a gpu
    cuda.memcpy_htod(dev_y, host_y) #copiar o input para a gpu
    cuda.memcpy_htod(dev_minimum, host_minimum) #copiar o input para a gpu

    #grid_size = N // block_size
    gridSize = (N + blocksNumber - 1) // blocksNumber

    # Executar o kernel na GPU
    score_kernel(dev_evaluatedLineArray, dev_y, dev_minimum, block=(blocksNumber, 1, 1), grid=(gridSize, 1))

    # # Transferir resultados da GPU de volta para a CPU
    cuda.memcpy_dtoh(host_minimum, dev_minimum)

    # # Libertar recursos na GPU
    dev_evaluatedLineArray.free()
    dev_y.free()
    dev_minimum.free()

    return host_minimum[0]

r = parallel_score()
print(f"{r[0]}")
#print(f"{r[0]} {r[1]}")
