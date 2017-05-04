#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

struct translation_thread {
	void *current_table;
	int *offset;
	int current;
};

int main(int argc, char** argv) {
	void *pg_table;
	int i,j, table_size=1, total_addresses,
	    levels = argc-3;
	int level_sizes[levels];
	
	// get number of pointers in contiguous page table
	for(i = 1, j =0; i < argc; i++) {
		if ( !strcmp(argv[i], "-n")) {
			total_addresses = atoi(argv[++i]);
			continue;
		}
		table_size *=  atoi(argv[i]);
		level_sizes[j++] = atoi(argv[i]);
	}

	fprintf(stderr, "size of page table: %d\n", table_size);
	fprintf(stderr, "number of addresses to \
			test with: %d\n", total_addresses);
	for (i = 0; i < levels; i++)
		fprintf(stderr, "level %d: %d\n", i, level_sizes[i]);

}
