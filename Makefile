# INC=-I$$CUDA_SDK_PATH/common/inc

all: cuhmm hmm fhmm

cuhmm: hmm.cu
	nvcc hmm.cu -o cuhmm

hmm: hmm.c
	gcc -Wall -pg -lm hmm.c -o hmm

fhmm: fhmm.c
	gcc -Wall -pg -lm fhmm.c -o fhmm

test: hmm fhmm
	./hmm -p1 -c config.0
	./hmm -p2 -c config.0
	./hmm -p3 -c config.0

	./fhmm -p1 -c config.0
	./fhmm -p2 -c config.0
	./fhmm -p3 -c config.0

clean: hmm
	rm -f hmm cuhmm fhmm
