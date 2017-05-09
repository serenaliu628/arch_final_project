/***********************************************************************************
  COMS 4824 Architecture Final Project
  Implementing Page Walk runing on the GPU using CUDA

  Copyright (c) 2017 Columbia University.
  All rights reserved.

  Created by Serean Liu and Raphael Norwitz.

Usage:

nvcc page_walk1.cu -o page_walk1.out

./page_walk1.out -n <total addresses> <intermediate table parameters>
(eg, ./page_walk1.out -n 400 2 3 2 4)
 ************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_D1 512
#define BLOCK_D2 1
#define BLOCK_D3 1
#define MAX_LEVELS 20

#define cudaCheckErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
					msg, cudaGetErrorString(__err), \
					__FILE__, __LINE__); \
			fprintf(stderr, "*** FAILED - ABORTING\n"); \
			exit(1); \
		} \
	} while (0)


static int max_table;

struct trans_thread {
	void **curr_table;
	unsigned long cuda_result;
	int offset[MAX_LEVELS];
	int curr;
	int max;
};

//function called by CUDA kernel
__host__ __device__ int translate_cpu(struct trans_thread *trans) {
	unsigned long *gpu_ptr = (unsigned long *) trans->curr_table;
	while(trans->curr < trans->max-1) {
		gpu_ptr = (unsigned long *) gpu_ptr[trans->offset[trans->curr]]; 
		trans->curr++;
	}
	return (int) *((int *) gpu_ptr + trans->offset[trans->curr]); // ((void *) trans->curr_table + trans->offset[trans->max-1]);
}

//CPU counterpart 
int translate_cpu2(struct trans_thread *trans) {
	//void **c = trans->curr_table;
	while(trans->curr < trans->max-1) {
		trans->curr_table = (void **) trans->curr_table[trans->offset[trans->curr]];
		trans->curr++;
	}
	// return 0;
	return (int) *((int *) trans->curr_table + trans->offset[trans->curr]); // ((void *) trans->curr_table + trans->offset[trans->max-1]);
}



// CUDA kernel: gpu_run_time<<<gridSize, blockSize>>>(d_new_threads, total_addresses);
__global__ void gpu_run_time(struct trans_thread *trans, int addresses) {
	// note that this assumes no third dimension to the grid
	// id of the block
	int myblock = blockIdx.x + blockIdx.y * gridDim.x;
	// size of each block (within grid of blocks)
	int blocksize = blockDim.x * blockDim.y * blockDim.z;
	// id of thread in a given block
	int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	// assign overall id/index of the thread
	int idx = myblock * blocksize + subthread;
	
	if(idx < addresses) {
		
		translate_cpu(&trans[idx]);
	}
}


// CPU analog for speed comparison
float cpu_run_time(struct trans_thread *trans, int addresses) {
	
	for(int i = 0; i < addresses; i++) {
		translate_cpu2(&trans[i]);	
	}

	return 0;
}

/* --------------------------- host code ------------------------------*/
double read_timer() {

	struct timeval end;
	gettimeofday( &end, NULL );
	return end.tv_sec+1.e-6*end.tv_usec;
}

//sequential phase 1: construct the table 
int construct_table(void *table, int *levels, int num_levels) {
	int i, j, level_size = 1;
	void **table_ptr = (void **) table;
	unsigned long **level_ptr;

	// set intermediate addresses of table
	for(i = 0; i < num_levels-1; i++)
	{
		level_size *= levels[i];

		level_ptr = (unsigned long **) table + level_size + (((unsigned long *)table_ptr - (unsigned long *) table)/(sizeof(unsigned long *)));

		fprintf(stderr, "level_size: %d, level_ptr: %d, table_ptr: %d\n", level_size, (level_ptr- (unsigned long **) table) / sizeof(void *), (unsigned long **) table_ptr -  (unsigned long **) table);

		for(j = 0; j < level_size; j++) {
			table_ptr[j] = level_ptr + ((j)*levels[i+1]);
		}

		table_ptr += level_size;
	}
	assert((intptr_t )table_ptr - (intptr_t )table < max_table);


	// set last level of page table to garbage; 
	for(i = 0; i < level_size * levels[num_levels-1]; i++) {
		*table_ptr = (unsigned long *) i;
		table_ptr++;
	}
	assert((intptr_t )table_ptr - (intptr_t )table == max_table);
	// return number of entries at the lowest level of the
	// page table
	return levels[num_levels-1] * level_size;
}

//sequential phase 2: generate struct trans_thread for each address
struct trans_thread *gen_addresses(int num_addr, int levels, int *level_sizes, void **pgd)
{
	int i,j;
	struct trans_thread *new_threads = (struct trans_thread *)malloc(sizeof(struct trans_thread) * num_addr);
	if (!new_threads){
		fprintf(stderr, "malloc failed: %d\n", strerror(errno));
		exit(1);
	}

	for(i = 0; i < num_addr; i++)
	{
		new_threads[i].curr_table = pgd;
		new_threads[i].max = levels;
		new_threads[i].curr = 0;

		for(j = 0; j < levels; j++) {
			new_threads[i].offset[j] =
				rand() % level_sizes[j];
		}

	}

	return new_threads;
}

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	void **d_pg_table; //device page table

	int i, j, table_size = 0, level_size = 1,
	    total_addresses, table_lowest_addresses,
	    levels = argc-3;

	int level_sizes[levels];
	//struct trans_thread *sample;
	struct trans_thread *h_new_threads; //host
	struct trans_thread *d_new_threads; //device

	cudaError_t cudaStat;

	//start of the program
	printf("===============================================================================\n");
	srand(time(NULL));

	// get number of pointers in contiguous page table
	for(i = 1, j =0; i < argc; i++) {
		if ( !strcmp(argv[i], "-n")) {
			total_addresses = atoi(argv[++i]);
			continue;
		}
		level_size *=  atoi(argv[i]);
		level_sizes[j++] = atoi(argv[i]);
		table_size += level_size;
	}

	//tuning the parameters of the device 
	// fixed block size (512 x1x1 threads)
	const dim3 blockSize(BLOCK_D1, BLOCK_D2, BLOCK_D3);
	// determine number of blocks we need for a given problem size
	int tmp = ceil(pow(total_addresses/BLOCK_D1, 0.5));
	printf("Grid dimension is %i x %i\n", tmp, tmp);
	//grid size
	dim3 gridSize(tmp, tmp, 1);
        //total number of threads 
	int nthreads = BLOCK_D1*BLOCK_D2*BLOCK_D3*tmp*tmp;
	//threads number legitimacy check
	if (nthreads < total_addresses){
		printf("\n============ NOT ENOUGH THREADS TO COVER total addresses=%d ===============\n\n",total_addresses);
	} else {
		printf("Launching %d threads (total_addresses=%d)\n", nthreads, total_addresses);
	}

	// allocate host memory
	max_table = table_size * sizeof(void *); //total size of page table
	void **pg_table = (void **) malloc(sizeof(void *) * table_size);

	if (!pg_table) {
		fprintf(stderr, "host memory allocation failed: %d\n", strerror(errno));
		exit(1);
	}
	else {
		printf ("host memory allocation succeeded.\n");
	}

	// allocate device memory
	cudaStat = cudaMalloc(&d_pg_table, sizeof(void *) * table_size);
	if(cudaStat != cudaSuccess) {
		printf ("device memory allocation failed.\n");
		return EXIT_FAILURE;
	}
	else {
		printf ("device memory allocation succeeded.\n");
	}

	/* --------------------------- sequential code on CPU; phase 1 ------------------------------*/
	//number of entries at the lowest level of the page table
	//number of translatable addresses
	printf ("now construct the page table on the host.\n");
	table_lowest_addresses = construct_table(pg_table, level_sizes, levels);

	fprintf(stderr, "number of translatable addresses: %d\n", table_lowest_addresses);
	fprintf(stderr, "total size of page table: %d\n", max_table);

	cudaDeviceSynchronize();
	double tInit = read_timer();

	/* --------------------------- copy the page table from CPU to the GPU ------------------------------*/
	cudaStat = cudaMemcpy(d_pg_table, pg_table, sizeof(void *) * table_size, cudaMemcpyHostToDevice);

	printf("Memory Copy for page table from Host to Device");
	if (cudaStat != cudaSuccess){
		printf("failed.\n");
		return EXIT_FAILURE;
	} else {
		printf("successful.\n");
	}
	cudaDeviceSynchronize();
	double tTransferToGPU_pgtable = read_timer();

	/* --------------------------- sequential code on CPU; phase 2 ------------------------------*/
	h_new_threads = gen_addresses(total_addresses, levels, level_sizes, pg_table);
	cudaDeviceSynchronize();
	double tInit2 = read_timer();

	/* --------------------------- copy the trans_threads from CPU to the GPU ------------------------------*/
	cudaStat = cudaMalloc( (void**) &d_new_threads, sizeof(struct trans_thread) * total_addresses) ;
	if (cudaStat != cudaSuccess){
		printf("device memory allocation for d_new_threads failed.\n");
		return EXIT_FAILURE;
	} else {
		printf("device memory allocation for d_new_threads succeeded.\n");
	}

	cudaStat = cudaMemcpy( d_new_threads, h_new_threads, sizeof(struct trans_thread) * total_addresses, cudaMemcpyHostToDevice);
	printf("Memory Copy h_new_threads from Host to Device");
	if (cudaStat != cudaSuccess){
		printf("failed.\n");
		return EXIT_FAILURE;
	} else {
		printf(" successful.\n");
	}
	cudaCheckErrors("cudaMemcpy h_new_threads fail");

	cudaDeviceSynchronize();
	double tTransferToGPU_threads = read_timer();
	
        /* --------------------------- parallel code on GPU: kernel ------------------------------*/
	gpu_run_time<<<gridSize, blockSize>>>(d_new_threads, total_addresses);

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr){
		printf("kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
	} else {
		printf("kernel launch success!\n");
	}

	cudaDeviceSynchronize();
	double gpu_time = read_timer();
	printf("GPU done!\n");

        /* --------------------------- sequential code on CPU; phase 3 ------------------------------*/
	printf("now do calculation on CPU for comparison!\n");
	cpu_run_time(h_new_threads, total_addresses);
	double cpu_time = read_timer();
	fprintf(stderr, "The CPU took %lu microseconds to compute %d addresses. ""For a table of depth %d.\n",	cpu_time - gpu_time , total_addresses, levels);

	/* --------------------------- sequential code on CPU; phase 4 ------------------------------*/
	printf("Timing results for n = %d\n", total_addresses);
        //printf("page table Transfer to GPU time: %f\n", tTransferToGPU_pgtable - tInit);
        //printf("threads Transfer to GPU time: %f\n", tTransferToGPU_threads - tInit2);
        printf("Calculation time (GPU): %f\n", gpu_time - tTransferToGPU_threads);
        //printf("Transfer from GPU time: %f\n", tTransferFromGPU - gpu_time);
	printf("Calculation time (CPU): %f\n", cpu_time - gpu_time);


	printf("Freeing memory...\n");
	printf("====================================================\n");
	free(pg_table);
	free(h_new_threads);
	cudaFree(d_pg_table);
	cudaFree(d_new_threads);
	return 0;
}
