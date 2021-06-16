#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
void bw_test(int min_msg_size, int max_msg_size, int global_rank, int global_size)
{
    int window_size = 10;
    double t_start, t_end, t_total;
    int iterations = 10;
    int skips = 2;
    char *gpu_src_buf, *gpu_dst_buf;
    MPI_Request request_s[global_size];
    MPI_Request request_r[global_size];
    MPI_Status reqstat[global_size];
    MPI_Status stat;
//    if (global_rank == src || global_rank == dst)
//    {
        cudaMalloc(&gpu_src_buf, max_msg_size*global_size);
        cudaMalloc(&gpu_dst_buf, max_msg_size*global_size);
//    }
    for(int size = min_msg_size; size <= max_msg_size; size *= 2)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t_total = 0.0;
        for(int i = 0; i < iterations + skips; ++i)
        {
           if(i >= skips) {
              t_start = MPI_Wtime();
           }
           for(int dst=0;dst<global_size;dst++)
            {
//               printf("Sending %d to %d\n",i,dst);
//                for(int j = 0; j < window_size; j++) {
                    MPI_Isend(&gpu_src_buf[size*dst], size, MPI_CHAR, dst, 100, MPI_COMM_WORLD, &request_s[dst]);
//                }
                MPI_Irecv(&gpu_dst_buf[size*dst], size, MPI_CHAR, dst, 100, MPI_COMM_WORLD, &request_r[dst]);
            }
           MPI_Waitall(global_size, request_r, MPI_STATUSES_IGNORE);
           MPI_Waitall(global_size, request_s, MPI_STATUSES_IGNORE);
           if (i >= skips) {
              t_end = MPI_Wtime();
              t_total += (t_end - t_start);
           }
        }
        MPI_Barrier(MPI_COMM_WORLD);
//        if (global_rank == src) {
            double tmp = size / 1e6 * iterations * window_size;
            double bw = tmp / t_total;
            printf( "%d MB/s  %f\n", size, bw);
//            fflush(stdout);
//        }
    }
        cudaFree(gpu_src_buf);
        cudaFree(gpu_dst_buf);
}
int main(int argc, char** argv)
{
    int global_rank, global_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    cudaSetDevice(global_rank);
    int min_msg_size = 8192;
    int max_msg_size = 2097152;
//    if (global_rank == 0)
//        printf("src %d --> dst %d \n", src, dst);
//    int src=0, dst=1;
    bw_test(min_msg_size, max_msg_size, global_rank, global_size);
    MPI_Barrier(MPI_COMM_WORLD);
//    src=0, dst=2;
//    if (global_rank == 0)
//        printf("src %d --> dst %d \n", src, dst);
    bw_test(min_msg_size, max_msg_size, global_rank, global_size);
    MPI_Finalize();
    return 0;
}
