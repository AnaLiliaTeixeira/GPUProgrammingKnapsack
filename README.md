#  GPU Programming Knapsack
## Evaluating Expressions
One of the variants of Genetic Algorithms is Genetic Programming, where individuals are programs (expressed as trees) instead of being expressed as an array of integers. Evaluation of programs is more complex, but since individuals can be evaluated in parallel, we can make use of the GPU.

## The Task
Write a GPU version of the program available as sequential.py. The program reads a CSV with input columns (all but the last) and the target output (last column). It also reads a file with all the functions to assess. We want to find the expression that minimizes the Mean Square Error of the prediciton using the candidate solutions (when compared with the correct value in the last column).

In your GPU program you should evaluate Task and Data parallelism, and you should do as much work as possible on the GPU. Consider dynamically generating the kernel code (as a string) from the functions.txt. 

Feel free to modify the generate_inputs.py to increase/decrease the workload.

## Report
You should submit a report up to 3 A4 pages answering the following points:

How did you parallelize your program?<br>
How many kernels and memory copies did you use? <br>
How did you minimize the number of kernels called?<br>
How did you minimize the number of data transfers required?<br>
How did you choose the number of threads and blocks (for each kernel)?<br>
How did you use shared_memory?<br>
How did you take into account branch conflicts in your project?<br>
After which combination of number of functions and number of rows in the CSV does it make sense to use the GPU vs the CPU?<br>
If you were to use this program within a Genetic Programming loop (with evaluation, tournament, crossover, mutation, etc...) several iterations, how would you adapt your code to minimize unnecessary overheads?
