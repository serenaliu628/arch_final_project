all: page_walk

.PHONY: clean
clean:
	rm -f page_walk page_walk.out

page_walk: page_walk.c
	gcc -o page_walk page_walk.c

gpu: page_walk.cu
	nvcc page_walk.cu -o page_walk.out
