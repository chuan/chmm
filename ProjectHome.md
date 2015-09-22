# Introduction #

This is an implementation of hidden Markov model (HMM) training and classification for NVIDIA CUDA platform. A serial implementation in C is also included for comparison.

The implementation of HMM follows the tutorial paper by Rabiner. The three problem for HMM defined in the paper are:
  1. compute the probability of the observation sequence
  1. compute the most probable sequence
  1. train hidden Markov mode parameters
This implementation supports all the three problems. However there is no support for continuous densities.




# Usage #

The command line usage is as follows.

```
$ ./hmm -h
hmm [-hnt] [-c config] [-p(1|2|3)]
usage:
  -h   help
  -c   configuration file
  -t   output computation time
  -p1  compute the probability of the observation sequence
  -p2  compute the most probable sequence (Viterbi)
  -p3  train hidden Markov mode parameters (Baum-Welch)
  -n   number of iterations
```




# Configuration #

The configuration format is same for all the three problems. For problem 1, the forward probabilities for all input sequences are computed from the given model parameters. For problem 2, decoding is performed for all sequences based on the given parameters. For problem 3, the given input are used as training data.

The following example shows a 16 states HMM with 2 observation symbols and 32 input sequences. Empty line and line begins with # will be ignored. The order of parameters follows the example.

```
# a HMM model configuration for testing purpose

# number of states
16

# number of output
2

# initial state probability
0.04 0.02 0.06 0.04 0.11 0.11 0.01 0.09 0.03 0.05 0.06 0.11 0.05 0.11 0.03 0.08 

# state transition probability
0.08 0.02 0.10 0.05 0.07 0.08 0.07 0.04 0.08 0.10 0.07 0.02 0.01 0.10 0.09 0.01 
0.06 0.10 0.11 0.01 0.04 0.11 0.04 0.07 0.08 0.10 0.08 0.02 0.09 0.05 0.02 0.02 
0.08 0.07 0.08 0.07 0.01 0.03 0.10 0.02 0.07 0.03 0.06 0.08 0.03 0.10 0.10 0.08 
0.08 0.04 0.04 0.05 0.07 0.08 0.01 0.08 0.10 0.07 0.11 0.01 0.05 0.04 0.11 0.06 
0.03 0.03 0.08 0.10 0.11 0.04 0.06 0.03 0.03 0.08 0.03 0.07 0.10 0.11 0.07 0.03 
0.02 0.05 0.01 0.09 0.05 0.09 0.05 0.12 0.09 0.07 0.01 0.07 0.05 0.05 0.11 0.06 
0.11 0.05 0.10 0.07 0.01 0.08 0.05 0.03 0.03 0.10 0.01 0.10 0.08 0.09 0.07 0.02 
0.03 0.02 0.16 0.01 0.05 0.01 0.14 0.14 0.02 0.05 0.01 0.09 0.07 0.14 0.03 0.01 
0.01 0.09 0.13 0.01 0.02 0.04 0.05 0.03 0.10 0.05 0.06 0.06 0.11 0.06 0.03 0.14 
0.09 0.03 0.04 0.05 0.04 0.03 0.12 0.04 0.07 0.02 0.07 0.10 0.11 0.03 0.06 0.09 
0.09 0.04 0.06 0.06 0.05 0.07 0.05 0.01 0.05 0.10 0.04 0.08 0.05 0.08 0.08 0.10 
0.07 0.06 0.01 0.07 0.06 0.09 0.01 0.06 0.07 0.07 0.08 0.06 0.01 0.11 0.09 0.05 
0.03 0.04 0.06 0.06 0.06 0.05 0.02 0.10 0.11 0.07 0.09 0.05 0.05 0.05 0.11 0.08 
0.04 0.03 0.04 0.09 0.10 0.09 0.08 0.06 0.04 0.07 0.09 0.02 0.05 0.08 0.04 0.09 
0.05 0.07 0.02 0.08 0.06 0.08 0.05 0.05 0.07 0.06 0.10 0.07 0.03 0.05 0.06 0.10 
0.11 0.03 0.02 0.11 0.11 0.01 0.02 0.08 0.05 0.08 0.11 0.03 0.02 0.10 0.01 0.11 

# state output probability
0.01 0.99 
0.58 0.42 
0.48 0.52 
0.58 0.42 
0.37 0.63 
0.33 0.67 
0.51 0.49 
0.28 0.72 
0.35 0.65 
0.61 0.39 
0.97 0.03 
0.87 0.13 
0.46 0.54 
0.55 0.45 
0.23 0.77 
0.76 0.24 

# data size
32 10

# data
0 0 0 0 0 0 1 0 1 1 
1 1 0 0 1 1 1 0 0 0 
1 1 0 1 0 0 0 1 0 1 
1 1 1 1 1 0 1 1 1 0 
0 1 0 1 1 0 1 1 1 1 
1 0 1 1 0 1 0 1 1 1 
1 0 1 1 1 1 0 0 1 1 
0 1 0 1 1 1 0 0 0 0 
0 1 1 0 0 0 1 1 1 1 
0 1 1 0 0 0 0 1 1 0 
1 1 1 1 1 0 1 1 0 0 
0 0 0 0 1 1 0 1 1 1 
1 0 1 0 1 1 1 1 1 0 
1 0 0 1 1 1 0 0 0 0 
0 0 1 1 1 0 0 0 0 1 
1 0 1 1 0 1 0 1 0 0 
1 0 1 0 1 0 0 1 0 1 
0 0 0 1 0 0 0 1 0 1 
1 1 1 0 0 0 0 1 0 0 
0 1 0 1 1 1 1 1 1 1 
0 0 0 0 1 1 1 0 1 0 
0 1 1 1 0 1 0 1 0 0 
1 1 0 1 1 0 0 0 0 1 
0 0 0 0 1 1 0 0 1 1 
0 1 0 1 1 1 1 1 0 0 
0 1 1 1 0 1 1 0 1 1 
1 1 1 1 0 0 1 1 0 0 
1 1 0 1 1 0 0 0 0 0 
0 1 0 0 0 0 0 0 0 1 
1 0 0 1 0 1 0 0 1 1 
0 1 0 1 0 0 1 1 0 0 
0 0 1 0 1 1 1 1 0 0 

```

# Further Information #

For more detailed information, please refer to the report at
http://liuchuan.org/pub/cuHMM.pdf