INC = -I$(CUDA_SDK_PATH)/common/inc
LIB = -L$(CUDA_SDK_PATH)/lib -lcutil
PG = -Xcompiler -DPROFILE_PG
PGPU = -Xcompiler -DPROFILE_GPU

all: cuhmm hmm fhmm

cuhmm: hmm.cu
	nvcc $(INC) $(LIB) hmm.cu -o cuhmm

hmm: hmm.c
	gcc -Wall -lm hmm.c -o hmm

fhmm: fhmm.c
	gcc -Wall -lm fhmm.c -o fhmm

clean: hmm
	rm -f hmm cuhmm fhmm
