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

#define handle_error(msg) \
  do { perror(msg); exit(EXIT_FAILURE); } while (0)

#define IDX(i,j,d) (((i)*(d))+(j))


int nstates = 0;                /* number of states */
int nobvs = 0;                  /* number of observations */
int nseq = 0;                   /* number of data sequences  */
int length = 0;                 /* data sequencel length */
float *prior = NULL;            /* initial state probabilities */
float *trans = NULL;            /* state transition probabilities */
float *obvs = NULL;             /* output probabilities */
int *data = NULL;
float *gmm = NULL;              /* gamma */
float *xi = NULL;               /* xi */
float *pi = NULL;               /* pi */

float logadd(float, float);
float sumf(float *, int);
float forward_backward(int *, size_t, int);
void viterbi(int *, size_t);
void init_count();
void update_prob();
void usage();
void freeall();

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
  float *loglik;
  float p;
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

      xi = (float *) malloc(sizeof(float) * nstates * nstates);
      if (xi == NULL) handle_error("malloc");

      pi = (float *) malloc(sizeof(float) * nstates);
      if (pi == NULL) handle_error("malloc");

    } else if (i == 1) {
      if (sscanf(linebuf, "%d", &nobvs) != 1) {
        fprintf(stderr, "config file format error: %d\n", i);
        freeall();
        exit(EXIT_FAILURE);
      }

      obvs = (float *) malloc(sizeof(float) * nstates * nobvs);
      if (obvs == NULL) handle_error("malloc");

      gmm = (float *) malloc(sizeof(float) * nstates * nobvs);
      if (gmm == NULL) handle_error("malloc");

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
        trans[IDX((i - 3),j,nstates)] = logf(d);
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
        obvs[IDX((i - 3 - nstates),j,nobvs)] = logf(d);
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
        data[(i - 4 - nstates * 2) * length + j] = k;
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

  if (mode == 3) {
    loglik = (float *) malloc(sizeof(float) * nseq);
    if (loglik == NULL) handle_error("malloc");

    for (i = 0; i < iterations; i++) {
      init_count();
      for (j = 0; j < nseq; j++) {
        loglik[j] = forward_backward(data + length * j, length, 1);
      }
      p = sumf(loglik, nseq);

      update_prob();

      printf("iteration %d log-likelihood: %.4f\n", i + 1, p);
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
    free(loglik);
  } else if (mode == 2) {
    for (i = 0; i < nseq; i++) {
      viterbi(data + length * i, length);
    }
  } else if (mode == 1) {
    loglik = (float *) malloc(sizeof(float) * nseq);
    if (loglik == NULL) handle_error("malloc");
    for (i = 0; i < nseq; i++) {
      loglik[i] = forward_backward(data + length * i, length, 0);
    }
    p = sumf(loglik, nseq);

    for (i = 0; i < nseq; i++)
      printf("%.4f\n", loglik[i]);
    printf("total: %.4f\n", p);
    free(loglik);
  }

  freeall();
  return 0;
}

/* compute sum of the array using Kahan summation algorithm */
float sumf(float *data, int size)
{
  float sum = data[0];
  int i;
  float y, t;
  float c = 0.0;
  for (i = 1; i < size; i++) {
    y = data[i] - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

/* initilize counts */
void init_count() {
  size_t i;
  for (i = 0; i < nstates * nobvs; i++)
    gmm[i] = - INFINITY;

  for (i = 0; i < nstates * nstates; i++)
    xi[i] = - INFINITY;

  for (i = 0; i < nstates; i++)
    pi[i] = - INFINITY;
}

void update_prob() {
  float pisum = - INFINITY;
  float gmmsum[nstates];
  float xisum[nstates];
  size_t i, j;

  for (i = 0; i < nstates; i++) {
    gmmsum[i] = - INFINITY;
    xisum[i] = - INFINITY;

    pisum = logadd(pi[i], pisum);
  }

  for (i = 0; i < nstates; i++) {
    prior[i] = pi[i] - pisum;
  }

  for (i = 0; i < nstates; i++) {
    for (j = 0; j < nstates; j++) {
      xisum[i] = logadd(xisum[i], xi[IDX(i,j,nstates)]);
    }
    for (j = 0; j < nobvs; j++) {
      gmmsum[i] = logadd(gmmsum[i], gmm[IDX(i,j,nobvs)]);
    }
  }

  for (i = 0; i < nstates; i++) {
    for (j = 0; j < nstates; j++) {
      trans[IDX(i,j,nstates)] = xi[IDX(i,j,nstates)] - xisum[i];
    }
    for (j = 0; j < nobvs; j++) {
      obvs[IDX(i,j,nobvs)] = gmm[IDX(i,j,nobvs)] - gmmsum[i];
    }
  }

}

/* forward backward algoritm: return observation likelihood */
float forward_backward(int *data, size_t len, int backward)
{
  /* construct trellis */
  float alpha[len][nstates];
  float beta[len][nstates];

  size_t i, j, k;
  float p, e;
  float loglik;

  for (i = 0; i < len; i++) {
    for (j = 0; j < nstates; j++) {
      alpha[i][j] = - INFINITY;
      beta[i][j] = - INFINITY;
    }
  }

  /* forward pass */
  for (i = 0; i < nstates; i++) {
    alpha[0][i] = prior[i] + obvs[IDX(i,data[0],nobvs)];
  }
  for (i = 1; i < len; i++) {
    for (j = 0; j < nstates; j++) {
      for (k = 0; k < nstates; k++) {
        p = alpha[i-1][k] + trans[IDX(k,j,nstates)] + obvs[IDX(j,data[i],nobvs)];
        alpha[i][j] = logadd(alpha[i][j], p);
      }
    }
  }
  loglik = -INFINITY;
  for (i = 0; i < nstates; i++) {
    loglik = logadd(loglik, alpha[len-1][i]);
  }

  if (! backward)
    return loglik;

  /* backward pass & update counts */
  for (i = 0; i < nstates; i++) {
    beta[len-1][i] = 0;         /* 0 = log (1.0) */
  }
  for (i = 1; i < len; i++) {
    for (j = 0; j < nstates; j++) {

      e = alpha[len-i][j] + beta[len-i][j] - loglik;
      gmm[IDX(j,data[len-i],nobvs)] = logadd(gmm[IDX(j,data[len-i],nobvs)], e);

      for (k = 0; k < nstates; k++) {
        p = beta[len-i][k] + trans[IDX(j,k,nstates)] + obvs[IDX(k,data[len-i],nobvs)];
        beta[len-1-i][j] = logadd(beta[len-1-i][j], p);

        e = alpha[len-1-i][j] + beta[len-i][k]
          + trans[IDX(j,k,nstates)] + obvs[IDX(k,data[len-i],nobvs)] - loglik;
        xi[IDX(j,k,nstates)] = logadd(xi[IDX(j,k,nstates)], e);
      }
    }
  }
  p = -INFINITY;
  for (i = 0; i < nstates; i++) {
    p = logadd(p, prior[i] + beta[0][i] + obvs[IDX(i,data[0],nobvs)]);

    e = alpha[0][i] + beta[0][i] - loglik;
    gmm[IDX(i,data[0],nobvs)] = logadd(gmm[IDX(i,data[0],nobvs)], e);

    pi[i] = logadd(pi[i], e);
  }

#ifdef DEBUG
  /* verify if forward prob == backward prob */
  if (fabs(p - loglik) > 1e-3) {
    fprintf(stderr, "Error: forward and backward incompatible: %f, %f\n", loglik, p);
  }
#endif

  return loglik;
}

/* find the most probable sequence */
void viterbi(int *data, size_t len)
{
  float lambda[len][nstates];
  int backtrace[len][nstates];
  int stack[len];

  size_t i, j, k;
  float p;

  for (i = 0; i < len; i++) {
    for (j = 0; j < nstates; j++) {
      lambda[i][j] = - INFINITY;
    }
  }

  for (i = 0; i < nstates; i++) {
    lambda[0][i] = prior[i] + obvs[IDX(i,data[0],nobvs)];
    backtrace[0][i] = -1;       /* -1 is starting point */
  }
  for (i = 1; i < len; i++) {
    for (j = 0; j < nstates; j++) {
      for (k = 0; k < nstates; k++) {
        p = lambda[i-1][k] + trans[IDX(k,j,nstates)] + obvs[IDX(j,data[i],nobvs)];
        if (p > lambda[i][j]) {
          lambda[i][j] = p;
          backtrace[i][j] = k;
        }
      }
    }
  }

  /* backtrace */
  for (i = 0; i < nstates; i++) {
    if (i == 0 || lambda[len-1][i] > p) {
      p = lambda[len-1][i];
      k = i;
    }
  }
  stack[len - 1] = k;
  for (i = 1; i < len; i++) {
    stack[len - 1 - i] = backtrace[len - i][stack[len - i]];
  }
  for (i = 0; i < len; i++) {
    printf("%d ", stack[i]);
  }
  printf("\n");
}

float logadd(float x, float y) {
  if (y <= x)
    return x + log1pf(expf(y - x));
  else
    return y + log1pf(expf(x - y));
}

void usage() {
  fprintf(stdout, "hmm [-hnt] [-c config] [-p(1|2|3)]\n");
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
  if (trans) free(trans);
  if (obvs) free(obvs);
  if (prior) free(prior);
  if (data) free(data);
  if (gmm) free(gmm);
  if (xi) free(xi);
  if (pi) free(pi);
}
