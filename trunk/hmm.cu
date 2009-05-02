/*
 * Copyright (c) 2009, Chuan Liu <chuan@cs.jhu.edu>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>


#define handle_error(msg) \
  do { perror(msg); exit(EXIT_FAILURE); } while (0)

#define IDX(i,j,d) (((i)*(d))+(j))

enum {
  BLOCK_SIZE = 16,
  NUM_THREADS = 128,
  IN_DEVICE = 1,
  IN_HOST = 0,
};

int nstates = 0;                /* number of states */
int nobvs = 0;                  /* number of observations */
int nseq = 0;                   /* number of data sequences  */
int length = 0;                 /* data sequencel length */
float *prior = NULL;            /* initial state probabilities */
float *trans = NULL;            /* state transition probabilities */
float *obvs = NULL;             /* output probabilities */
int *data = NULL;               /* observations */
float *transd = NULL;           /* trans in device memory */
float *obvsd = NULL;            /* obvs in device memory */
float *gmmd = NULL;             /* gamma in device memory */
float *xid = NULL;              /* xi in device memory */
float *pid = NULL;              /* pi in device memory */

/* function called in main fuction */
void usage();
void freeall();
void init_count();
float forward_backward(int backward);
void update_prob();

/* utility functions */
float logadd(float, float);
__device__ float logaddd(float, float);
float sumf(float *, int, int);
float logsumf(float *, int, int);


int main(int argc, char *argv[])
{
  char *configfile = NULL;
  FILE *fin, *bin;

  char *linebuf = NULL;
  size_t buflen = 0;

  int iterations = 3;
  int mode = 3;

  int c;
  float d;

  int i, j, k;
  opterr = 0;


  while ((c = getopt(argc, argv, "c:n:hp:")) != -1) {
    switch (c) {
    case 'c':
      configfile = optarg;
      break;
    case 'h':
      usage();
      exit(EXIT_SUCCESS);
    case 'n':
      iterations = atoi(optarg);
      break;
    case 'p':
      mode = atoi(optarg);
      if (mode != 1 && mode != 2 && mode != 3) {
        fprintf(stderr, "illegal mode: %d\n", mode);
        exit(EXIT_FAILURE);
      }
      break;
    case '?':
      fprintf(stderr, "illegal options\n");
      exit(EXIT_FAILURE);
    default:
      abort();
    }
  }

  if (configfile == NULL) {
    fin = stdin;
  } else {
    fin = fopen(configfile, "r");
    if (fin == NULL) {
      handle_error("fopen");
    }
  }
  
  i = 0;
  while ((c = getline(&linebuf, &buflen, fin)) != -1) {
    if (c <= 1 || linebuf[0] == '#')
      continue;
    
    if (i == 0) {
      if (sscanf(linebuf, "%d", &nstates) != 1) {
        fprintf(stderr, "config file format error: %d\n", i);
        freeall();
        exit(EXIT_FAILURE);
      }

      prior = (float *) malloc(sizeof(float) * nstates);
      if (prior == NULL) handle_error("malloc");

      trans = (float *) malloc(sizeof(float) * nstates * nstates);
      if (trans == NULL) handle_error("malloc");

    } else if (i == 1) {
      if (sscanf(linebuf, "%d", &nobvs) != 1) {
        fprintf(stderr, "config file format error: %d\n", i);
        freeall();
        exit(EXIT_FAILURE);
      }

      obvs = (float *) malloc(sizeof(float) * nstates * nobvs);
      if (obvs == NULL) handle_error("malloc");

    } else if (i == 2) {
      /* read initial state probabilities */ 
      bin = fmemopen(linebuf, buflen, "r");
      if (bin == NULL) handle_error("fmemopen");
      for (j = 0; j < nstates; j++) {
        if (fscanf(bin, "%f", &d) != 1) {
          fprintf(stderr, "config file format error: %d\n", i);
          freeall();
          exit(EXIT_FAILURE);
        }
        prior[j] = logf(d);
      }
      fclose(bin);

    } else if (i <= 2 + nstates) {
      /* read state transition  probabilities */ 
      bin = fmemopen(linebuf, buflen, "r");
      if (bin == NULL) handle_error("fmemopen");
      for (j = 0; j < nstates; j++) {
        if (fscanf(bin, "%f", &d) != 1) {
          fprintf(stderr, "config file format error: %d\n", i);
          freeall();
          exit(EXIT_FAILURE);
        }
        trans[IDX((i - 3), j, nstates)] = logf(d);
      }
      fclose(bin);
    } else if (i <= 2 + nstates * 2) {
      /* read output probabilities */
      bin = fmemopen(linebuf, buflen, "r");
      if (bin == NULL) handle_error("fmemopen");
      for (j = 0; j < nobvs; j++) {
        if (fscanf(bin, "%f", &d) != 1) {
          fprintf(stderr, "config file format error: %d\n", i);
          freeall();
          exit(EXIT_FAILURE);
        }
        obvs[IDX((i - 3 - nstates), j, nobvs)] = logf(d);
      }
      fclose(bin);
    } else if (i == 3 + nstates * 2) {
      if (sscanf(linebuf, "%d %d", &nseq, &length) != 2) {
        fprintf(stderr, "config file format error: %d\n", i);
        freeall();
        exit(EXIT_FAILURE);
      }
      data = (int *) malloc (sizeof(int) * nseq * length);
      if (data == NULL) handle_error("malloc");
    } else if (i <= 3 + nstates * 2 + nseq) {
      /* read data */
      bin = fmemopen(linebuf, buflen, "r");
      if (bin == NULL) handle_error("fmemopen");
      for (j = 0; j < length; j++) {
        if (fscanf(bin, "%d", &k) != 1 || k < 0 || k >= nobvs) {
          fprintf(stderr, "config file format error: %d\n", i);
          freeall();
          exit(EXIT_FAILURE);
        }
        data[j * nseq + (i - 4 - nstates * 2)] = k;
      }
      fclose(bin);
    }

    i++;
  }
  fclose(fin);
  if (linebuf) free(linebuf);

  if (i < 4 + nstates * 2 + nseq) {
    fprintf(stderr, "configuration incomplete.\n");
    freeall();
    exit(EXIT_FAILURE);
  }

  
  /* initial cuda device memory */
  c = sizeof(float) * nstates * nstates;
  cudaMalloc((void**)&transd, c);
  cudaMemcpy(transd, trans, c, cudaMemcpyHostToDevice);
  
  c = sizeof(float) * nstates * nobvs;
  cudaMalloc((void**)&obvsd, c);
  cudaMemcpy(obvsd, obvs, c, cudaMemcpyHostToDevice);

  if (mode == 3) {
    /* estimating parameters using Baum-Welch algorithm */
    for (i = 0; i < iterations; i++) {
      init_count();
      d = forward_backward(1);
      update_prob();

      printf("iteration %d log-likelihood: %.4f\n", i + 1, d);
      printf("updated parameters:\n");
      printf("# initial state probability\n");
      for (j = 0; j < nstates; j++) {
        printf(" %.4f", exp(prior[j]));
      }
      printf("\n");
      printf("# state transition probability\n");
      for (j = 0; j < nstates; j++) {
        for (k = 0; k < nstates; k++) {
          printf(" %.4f", exp(trans[IDX(j,k,nstates)]));
        }
        printf("\n");
      }
      printf("# state output probility\n");
      for (j = 0; j < nstates; j++) {
        for (k = 0; k < nobvs; k++) {
          printf(" %.4f", exp(obvs[IDX(j,k,nobvs)]));
        }
        printf("\n");
      }
      printf("\n");
    }

  } else if (mode == 1) {
    /* compute forward probabilities */
    forward_backward(0);
  } else if (mode == 2) {
    /* find most likely path using Viterbi algorithm */
  }
 
  freeall();
  return 0;
}


/* kernel function copied from NVIDIA CUDA SDK */
__global__ void
reduce2(float *g_idata, float *g_odata)
{
  __shared__ float sdata[NUM_THREADS];

  /* load shared mem */
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  /* do reduction in shared mem */
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
      if (tid < s) 
        {
          sdata[tid] += sdata[tid + s];
        }
      __syncthreads();
    }

  /* write result for this block to global mem */
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* calculate the sum of the array of n floats by calling a reduce
   kernel recursively. indevce indicates whether the array points to
   memroy in device or not, i.e. indevice = 0 means the array is in
   main memory. */
float sumf(float *array, int n, int indevice)
{
  int i;
  int num_blocks;
  int remains;
  int size;
  float sum = 0.0;
  float *gin;

  /* NUM_THREADS also serves as CPU threshold */
  if (n < NUM_THREADS) {
    if (indevice == 0) {
      for (i = 0; i < n; i++) {
        sum += array[i];
      }
    } else {
      float gout[n];
      size = sizeof(float) * n;
      cudaMemcpy(gout, array, size, cudaMemcpyDeviceToHost);
      for (i = 0; i < n; i++) {
        sum += gout[i];
      }
    }
  } else {

    num_blocks = n / NUM_THREADS;
    remains = n - num_blocks * NUM_THREADS;

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid(num_blocks);

    if (indevice == 0) {

      size = sizeof(float) * num_blocks * NUM_THREADS;
      cudaMalloc((void**) &gin, size);
      cudaMemcpy(gin, array, size, cudaMemcpyHostToDevice);
    
      reduce2<<<dimGrid, dimBlock>>>(gin, gin);

      sum += sumf(gin, num_blocks, 1);

      if (remains > 0)
        sum += sumf(array + num_blocks * NUM_THREADS, remains, 0);

      cudaFree(gin);

    } else {
      reduce2<<<dimGrid, dimBlock>>>(gin, gin);
      sum += sumf(gin, num_blocks, 1);
      
      if (remains > 0)
        sum += sumf(gin + num_blocks * NUM_THREADS, remains, 1);
    }
  }
  return sum;
}

/* logarithm version of the reduce kernel function */
__global__ void
logreduce2(float *g_idata, float *g_odata)
{
  __shared__ float sdata[NUM_THREADS];

  /* load shared mem */
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  /* do reduction in shared mem */
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = logaddd(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  /* write result for this block to global mem */
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* logarithm version of the sumf function */
float logsumf(float *array, int n, int indevice)
{
  int i;
  int num_blocks;
  int remains;
  int size;
  float sum = - INFINITY;
  float *gin;

  /* NUM_THREADS also serves as CPU threshold */
  if (n < NUM_THREADS) {
    if (indevice == 0) {
      for (i = 0; i < n; i++) {
        sum = logadd(sum, array[i]);
      }
    } else {
      float gout[n];
      size = sizeof(float) * n;
      cudaMemcpy(gout, array, size, cudaMemcpyDeviceToHost);
      for (i = 0; i < n; i++) {
        sum = logadd(sum, gout[i]);
      }
    }
  } else {

    num_blocks = n / NUM_THREADS;
    remains = n - num_blocks * NUM_THREADS;

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid(num_blocks);

    if (indevice == 0) {

      size = sizeof(float) * num_blocks * NUM_THREADS;
      cudaMalloc((void**) &gin, size);
      cudaMemcpy(gin, array, size, cudaMemcpyHostToDevice);
    
      logreduce2<<<dimGrid, dimBlock>>>(gin, gin);

      sum = logadd(sum, logsumf(gin, num_blocks, 1));

      if (remains > 0)
        sum = logadd(sum, logsumf(array + num_blocks * NUM_THREADS, remains, 0));

      cudaFree(gin);

    } else {
      logreduce2<<<dimGrid, dimBlock>>>(gin, gin);
      sum = logadd(sum, logsumf(gin, num_blocks, 1));

      if (remains > 0)
        sum = logadd(sum, logsumf(gin + num_blocks * NUM_THREADS, remains, 1));
    }
  }
  return sum;
}

/* kernel function to initialized xi counts */
__global__ void
initxid(float *xid, int nstates)
{
  int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  xid[IDX(y, x, nstates)] = logf(0);
}

__global__ void
initgmmd(float *gmmd)
{
  int i = blockIdx.x * NUM_THREADS + threadIdx.x;

  gmmd[i] = logf(0);
}

/* initilize counts */
void init_count() {
  int size;
  size_t i;
  float pi[nstates];

  dim3 xiBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 xiGrid(nstates / xiBlock.x, nstates / xiBlock.y);

  size = sizeof(float) * nstates * nstates;
  cudaMalloc((void**)&xid, size);
  initxid<<<xiGrid, xiBlock>>>(xid, nstates);
  
  size = sizeof(float) * nstates * nobvs;
  cudaMalloc((void**)&gmmd, size);
  /* this partition implies if (N = nstate * nobvs) > NUM_THREADS, N
     should be a multiple of NUM_THREADS, e.g. an invalid case ma be:
     NUM_THREADS = 256, nstates = 128 * 3, and nobvs = 3. */
  dim3 gmmBlock(NUM_THREADS);
  dim3 gmmGrid(nstates * nobvs / gmmBlock.x);
  initgmmd<<<gmmGrid, gmmBlock>>>(gmmd);

  for (i = 0; i < nstates; i++)
    pi[i] = - INFINITY;
  size = sizeof(float) * nstates;
  cudaMalloc((void**)&pid, size);
  cudaMemcpy(pid, pi, size, cudaMemcpyHostToDevice);
}

/* add up two logarithm while avoiding overflow */
float logadd(float x, float y) {
  if (y <= x)
    return x + log1pf(expf(y - x));
  else
    return y + log1pf(expf(x - y));
}

/* add up two logarithm while avoiding overflow (device version) */
__device__ float logaddd(float x, float y) {
  if (y <= x)
    return x + log1pf(expf(y - x));
  else
    return y + log1pf(expf(x - y));
}

/* the kernel function for stepfwd */
__global__ void
stepfwdd(float *pre, float *transd, int *O, float *obvsd,
         int nstates, int nobvs, float *A)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = nstates * BLOCK_SIZE * by;
  int aEnd = aBegin + nstates - 1;
  int aStep = BLOCK_SIZE;

  int bBegin = BLOCK_SIZE * bx;
  int bStep = BLOCK_SIZE * nstates;

  float sub = logf(0);

  int a, b, k;

  for (a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Os[BLOCK_SIZE][BLOCK_SIZE];
    
    As[ty][tx] = pre[a + nstates * ty + tx];
    Bs[ty][tx] = transd[b + nstates * ty + tx];
    Os[ty][tx] = obvsd[IDX(BLOCK_SIZE * bx + tx, O[BLOCK_SIZE * by + ty], nobvs)];

    __syncthreads();

    for (k = 0; k < BLOCK_SIZE; ++k) {
      sub = logaddd(sub, As[ty][k] + Bs[k][tx] + Os[ty][tx]);
    }

    __syncthreads();
  }

  a = nstates * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  A[a + nstates * ty + tx] = sub;
}

/* compute one step (n-th) forward probability. the data partition
   follows matrix muptiplication A x B, where the previous forward
   probabilities are A and state transition probabilities are B. */
void stepfwd(float *alpha, size_t n)
{
  int size;
  int *Od;
  float *A;
  float *Ad;

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(nstates / dimBlock.x, nseq / dimBlock.y);

  size = sizeof(int) * nseq;
  cudaMalloc((void**)&Od, size);
  cudaMemcpy(Od, data + n * nseq, size, cudaMemcpyHostToDevice);

  size = sizeof(float) * nstates * nseq;

  cudaMalloc((void**)&A, size);
  cudaMemcpy(A, alpha + (n - 1) * nseq * nstates, size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&Ad, size);

  stepfwdd<<<dimGrid, dimBlock>>>(A, transd, Od, obvsd, nstates, nobvs, Ad);

  cudaMemcpy(alpha + n * nseq * nstates, Ad, size, cudaMemcpyDeviceToHost);

  cudaFree(Od);
  cudaFree(A);
  cudaFree(Ad);
}

/* init first slice of forwad probability matrix. */
void initfwd(float *alpha)
{
  size_t i, j;
  for (i = 0; i < nseq; i++) {
    for (j = 0; j < nstates; j++) {
      alpha[IDX(i, j, nstates)] = prior[j] + obvs[IDX(j, data[i], nobvs)];
    }
  }
}

/* kernel function for initbck() */
__global__ void
initbckd(float *B, int nstates)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  size_t i = bx * BLOCK_SIZE + tx;
  size_t j = by * BLOCK_SIZE + ty;

  B[IDX(j, i, nstates)] = 0;
}

/* initial backward probabilities for the last slice. data partition
   is the same as initfwd(). the values are all initilized to be 0. */
void initbck(float *beta)
{
  int size;
  float *Bd;

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(nstates / dimBlock.x, nseq / dimBlock.y);

  size = sizeof(float) * nstates * nseq;
  cudaMalloc((void**)&Bd, size);

  initbckd<<<dimGrid, dimBlock>>>(Bd, nstates);

  cudaMemcpy(beta, Bd, size, cudaMemcpyDeviceToHost);

  cudaFree(Bd);
}

/* kernel function for updating xi counts */
__global__ void
updatexid(float *A, float *B, float *transd, int *O, float *obvsd,
          int nstates, int nobvs, int nseq, float* loglik, float *xid)
{
  int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  float e;
  int i;

  for (i = 0; i < nseq; i++) {
    e = A[IDX(i, y, nstates)] + transd[IDX(y, x, nstates)]
      + B[IDX(i, x, nstates)] + obvsd[IDX(x, O[i], nobvs)] - loglik[i];
    xid[IDX(y, x, nstates)] = logaddd(xid[IDX(y, x, nstates)], e);
  }
}

/* kernel function for updating gamma counts */
__global__ void
updategmmd(float *A, float *B, int *O, int nstates,
           int nobvs, int nseq, float *loglik, float *gmmd)
{
  int x = blockIdx.x * NUM_THREADS + threadIdx.x;
  float e;
  int i = 0;

  for (i = 0; i < nseq; i++) {
    e = A[IDX(i, x, nstates)] + B[IDX(i, x, nstates)] - loglik[i];
    gmmd[IDX(x, O[i], nobvs)] = logaddd(gmmd[IDX(x, O[i], nobvs)], e);
  }
}

/* kernel function for updateing pi counts */
__global__ void
updatepid(float *A, float *B, int *O, int nstates,
          int nseq, float *loglik, float *pid)
{
  int x = blockIdx.x * NUM_THREADS + threadIdx.x;
  float e;
  int i = 0;

  for (i = 0; i < nseq; i++) {
    e = A[IDX(i, x, nstates)] + B[IDX(i, x, nstates)] - loglik[i];
    pid[x] = logaddd(pid[x], e);
  }
}

/* kernel function for stepbck() */
__global__ void
stepbckd(float *pre, float *transd, int *O, float *obvsd,
         int nstates, int nobvs, float *B)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = nstates * BLOCK_SIZE * by;
  int aEnd = aBegin + nstates - 1;
  int aStep = BLOCK_SIZE;

  int bBegin = nstates * BLOCK_SIZE * bx;
  int bStep = BLOCK_SIZE;

  float sub = logf(0);

  int a, b, k;

  for (a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b+= bStep) {

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    As[ty][tx] = pre[a + IDX(ty, tx, nstates)] +
      obvsd[IDX(a - aBegin + tx, O[by * BLOCK_SIZE + ty], nobvs)];

    Bs[ty][tx] = transd[b + IDX(tx, ty, nstates)];

    __syncthreads();

    for (k = 0; k < BLOCK_SIZE; ++k) {
      sub = logaddd(sub, As[ty][k] + Bs[k][tx]);
    }

    __syncthreads();
  }

  b = nstates * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  B[b + nstates * ty + tx] = sub;
}

/* compute one step backward probability and update counts.

   data partition for computing backward probilities follows forward
   pass. the result of the backward probabities are stored in the
   memory pointed by *beta.

   data partition for updating xi counts carries out in matrix of size
   nstates x nstates.
   
   data partition for gamma couts required nstates be a multiple of
   NUM_THREADS.  */
void stepbck(float *alpha, float *pre, size_t n, float* loglik, float *beta)
{
  int size;
  int *Od;

  float *Bd;
  float *pred;

  float *Ad;

  float *loglikd;

  dim3 xiBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 xiGrid(nstates / xiBlock.x, nstates / xiBlock.y);

  dim3 gmmBlock(NUM_THREADS);
  dim3 gmmGrid(nstates / gmmBlock.x);

  dim3 sBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 sGrid(nstates / sBlock.x, nseq / sBlock.y);

  size = sizeof(int) * nseq;
  cudaMalloc((void**)&Od, size);
  cudaMemcpy(Od, data + (n + 1) * nseq, size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&loglikd, size);
  cudaMemcpy(loglikd, loglik, size, cudaMemcpyHostToDevice);

  size = sizeof(float) * nstates * nseq;

  cudaMalloc((void**)&pred, size);
  cudaMemcpy(pred, pre, size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&Ad, size);
  cudaMemcpy(Ad, alpha + n * nseq * nstates, size, cudaMemcpyHostToDevice);

  /* update counts */
  updatexid<<<xiGrid, xiBlock>>>(Ad, pred, transd, Od, obvsd,
                               nstates, nobvs, nseq, loglikd, xid);

  cudaMemcpy(Ad, alpha + (n + 1) * nseq * nstates, size, cudaMemcpyHostToDevice);
  updategmmd<<<gmmGrid, gmmBlock>>>(Ad, pred, Od, nstates,
                                    nobvs, nseq, loglikd, gmmd);


  /* compute one step beta probabilities */
  cudaMalloc((void**)&Bd, size);
  stepbckd<<<sGrid, sBlock>>>(pred, transd, Od, obvsd, nstates, nobvs, Bd);

  cudaMemcpy(beta, Bd, size, cudaMemcpyDeviceToHost);

  cudaFree(Ad);
  cudaFree(Od);
  cudaFree(Bd);
  cudaFree(pred);
  cudaFree(loglikd);
}

void last_update(float *alpha, float *beta, float *loglik)
{
  int size;
  int *Od;
  float *Bd, *Ad, *loglikd;

  dim3 gmmBlock(NUM_THREADS);
  dim3 gmmGrid(nstates / gmmBlock.x);

  dim3 piBlock(NUM_THREADS);
  dim3 piGrid(nstates / piBlock.x);

  size = sizeof(int) * nseq;
  cudaMalloc((void**)&Od, size);
  cudaMemcpy(Od, data, size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&loglikd, size);
  cudaMemcpy(loglikd, loglik, size, cudaMemcpyHostToDevice);

  size = sizeof(float) * nstates * nseq;
  cudaMalloc((void**)&Bd, size);
  cudaMalloc((void**)&Ad, size);
  cudaMemcpy(Bd, beta, size, cudaMemcpyHostToDevice);
  cudaMemcpy(Ad, alpha, size, cudaMemcpyHostToDevice);

  updategmmd<<<gmmGrid, gmmBlock>>>(Ad, Bd, Od, nstates, nobvs, nseq, loglikd, gmmd);
  updatepid<<<piGrid, piBlock>>>(Ad, Bd, Od, nstates, nseq, loglikd, pid);

  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Od);
  cudaFree(loglikd);
}

/* forwad backward algorithm: running on all sequences in parallel */
float forward_backward(int backward)
{
  float *alpha = NULL;
  float *beta = NULL;
  float *prebeta = NULL;
  size_t i;
  size_t a;
  float *loglik = NULL;
  float p;
  int size;
  float *bckll = NULL;

  /* initial alpha probabilities for all data sequences
     (LARGEST memory allocaltion in the entire program) */
  alpha = (float *) malloc(sizeof(float) * length * nstates * nseq);
  if (alpha == NULL) handle_error("malloc");

  initfwd(alpha);

  /* forward pass */
  for (i = 1; i < length; i++) {
    stepfwd(alpha, i);
  }

  loglik = (float *) malloc(sizeof(float) * nseq);
  if (loglik == NULL) handle_error("malloc");
  for (i = 0, a = (length - 1) * nseq * nstates;
       i < nseq; i++) {
    loglik[i] = logsumf(alpha + a + i * nstates, nstates, 0);
  }
  p = sumf(loglik, nseq, IN_HOST);

  if (! backward) {
    for (i = 0; i < nseq; i++) {
      printf("%.4f\n", loglik[i]);
    }
    printf("total: %.4f\n", p);
    if (loglik) free(loglik);
    if (alpha) free(alpha);
    return p;
  }

  /* initial backward probabilities */
  size = sizeof(float) * nstates * nseq;
  beta = (float *) malloc(size);
  if (beta == NULL) handle_error("malloc");
  prebeta = (float *) malloc(size);
  if (prebeta == NULL) handle_error("malloc");

  initbck(prebeta);

  /* backward pass & update counts at last step */
  for (i = 1; i < length; i++) {
    stepbck(alpha, prebeta, length - 1 - i, loglik, beta);
    memmove(prebeta, beta, size);
  }

  /* update first slice of data */
  last_update(alpha, prebeta, loglik);

#ifdef DEBUG
  /* compute backward prob for verification purpose */
  bckll = (float *) malloc(sizeof(float) * nseq);
  if (bckll == NULL) handle_error("malloc");
  for (i = 0; i < nseq; i++) {
    p = - INFINITY;
    for (j = 0; j < nstates; j++) {
      p = logadd(p, prior[j] + beta[IDX(i,j,nstates)] + obvs[IDX(j, data[i], nobvs)]);
    }
    bckll[i] = p;
  }
  p = sumf(bckll, nseq, IN_HOST);

  for (i = 0; i < nseq; i++)
    if (fabs(bckll[i] - loglik[i]) > 1e-3)
      fprintf(stderr, "Error: forward and backward incompatible: %f, %f\n",
              loglik[i], bckll[i]);
#endif

  if (alpha) free(alpha);
  if (beta) free(beta);
  if (prebeta) free(prebeta);
  if (loglik) free(loglik);
  if (bckll) free(bckll);

  return p;
}

/* update model parameters using estimated counts */
void update_prob()
{
  float pisum;
  float gmmsum[nstates];
  float xisum[nstates];
  float pi[nstates];
  
  float gmm[nstates * nobvs];
  float xi[nstates * nstates];

  size_t i, j;

  cudaMemcpy(xi, xid, nstates * nstates * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(gmm, gmmd, nobvs * nstates * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(pi, pid, nstates * sizeof(float), cudaMemcpyDeviceToHost);

  if (gmmd) cudaFree(gmmd);
  if (xid) cudaFree(xid);
  if (pid) cudaFree(pid);

  pisum = logsumf(pi, nstates, IN_HOST);
  for (i = 0; i < nstates; i++) {
    gmmsum[i] = logsumf(gmm + i * nobvs, nobvs, IN_HOST);
    xisum[i] = logsumf(xi + i * nstates, nstates, IN_HOST);
  }

  for (i = 0; i < nstates; i++) {
    prior[i] = pi[i] - pisum;
  }

  for (i = 0; i < nstates; i++) {
    for (j = 0; j < nstates; j++) {
      trans[IDX(i,j,nstates)] = xi[IDX(i,j,nstates)] - xisum[i];
    }
    for (j = 0; j < nobvs; j++) {
      obvs[IDX(i,j,nobvs)] = gmm[IDX(i,j,nobvs)] - gmmsum[i];
    }
  }

  /* update indevice parameters */
  cudaMemcpy(transd, trans, nstates * nstates * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(obvsd, obvs, nobvs * nstates * sizeof(float), cudaMemcpyHostToDevice);
}

void usage() {
  fprintf(stdout, "chmm [-hnt] [-c config] [-p(1|2|3)]\n");
  fprintf(stdout, "usage:\n");
  fprintf(stdout, "  -h   help\n");
  fprintf(stdout, "  -c   configuration file\n");
  fprintf(stdout, "  -t   output computation time\n");
  fprintf(stdout, "  -p1  compute the probability of the observation sequence\n");
  fprintf(stdout, "  -p2  compute the most probable sequence (Viterbi)\n");
  fprintf(stdout, "  -p3  train hidden Markov mode parameters (Baum-Welch)\n");
  fprintf(stdout, "  -n   number of iterations\n");
}

/* free all memory */
void freeall() {

  if (transd) cudaFree(transd);
  if (obvsd) cudaFree(obvsd);

  if (trans) free(trans);
  if (obvs) free(obvs);
  if (prior) free(prior);
  if (data) free(data);
}
