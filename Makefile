INC = -I$(CUDA_SDK_PATH)/samples/common/inc
LIB = -L$(CUDA_SDK_PATH)/lib64
PG = -Xcompiler -DPROFILE_PG
PGPU = -Xcompiler -DPROFILE_GPU

all: cuhmm hmm fhmm

cuhmm: hmm.cu
	nvcc $(INC) $(LIB) hmm.cu -o cuhmm

hmm: hmm.c
	gcc hmm.c -Wall -lm -o hmm

fhmm: fhmm.c
	gcc fhmm.c -Wall -lm -o fhmm

clean: hmm
	rm -f hmm cuhmm fhmm
