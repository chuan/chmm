INC = -I$(CUDA_SDK_PATH)/common/inc
LIB = -L$(CUDA_SDK_PATH)/lib -lcutil
PG = -Xcompiler -DPROFILE_PG
PGPU = -Xcompiler -DPROFILE_GPU

all: cuhmm hmm fhmm

cuhmm: hmm.cu
	nvcc $(INC) $(LIB) $(PGPU) $(PG) hmm.cu -o cuhmm

hmm: hmm.c
	gcc -Wall -lm hmm.c -o hmm

fhmm: fhmm.c
	gcc -Wall -DPROFILE -lm fhmm.c -o fhmm

test: hmm fhmm
	./hmm -p1 -c config.0
	./hmm -p2 -c config.0
	./hmm -p3 -c config.0

	./fhmm -p1 -c config.0
	./fhmm -p2 -c config.0
	./fhmm -p3 -c config.0

clean: hmm
	rm -f hmm cuhmm fhmm
