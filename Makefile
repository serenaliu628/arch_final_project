all: page_walk

.PHONY: clean
clean:
	rm -f page_walk

page_walk: page_walk.c
	gcc -o page_walk page_walk.c
