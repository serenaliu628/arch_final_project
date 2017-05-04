#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

static int max_table;

struct translation_thread {
	void *current_table;
	int *offset;
	int current;
};

int construct_table(void *table, int *levels, int num_levels) {
	int i, j, level_size = 1;
        void **table_ptr = (void **) table,
	    *level_ptr = (void *) table;

	// set intermediate addresses of table
	for(i = 0; i < num_levels-1; i++)
	{
		level_size *= levels[i];
		level_ptr = (void *) table_ptr + level_size;

		for(j = 0; j < level_size; j++) {
			table_ptr[j] = level_ptr + levels[i+1];
			level_ptr += levels[i+1];
		}
		

		table_ptr += level_size;
	}


	// set last level of page table to garbage; for our purposes
	// it doesn't matter
	for(i = 0; i < levels[num_levels-1] * level_size; i++) {
		*(table_ptr++) = (int *) i;
	}

	assert((void *) table_ptr - table == max_table);
	
	// return number of entries at the lowest level of the
	// page table
	return levels[num_levels-1] * level_size;
}

int main(int argc, char** argv) {
	void *pg_table;
	int i, j, table_size = 0, level_size = 1, 
	    total_addresses, table_lowest_addresses, 
	    levels = argc-3;
	int level_sizes[levels];
	
	// get number of pointers in contiguous page table
	for(i = 1, j =0; i < argc; i++) {
		if ( !strcmp(argv[i], "-n")) {
			total_addresses = atoi(argv[++i]);
			continue;
		}
		level_size *=  atoi(argv[i]);
		level_sizes[j++] = atoi(argv[i]);
		table_size += level_size;
	}

	max_table = table_size * sizeof(void *);

	for (i = 0; i < levels; i++)
		fprintf(stderr, "level %d: %d\n", i, level_sizes[i]);

	pg_table = (void *) malloc(sizeof(void *) * table_size);
	
	if (!pg_table) {
		fprintf(stderr, "malloc failed: %d\n", strerror(errno));
		exit(1);
	}

	table_lowest_addresses = construct_table(pg_table, level_sizes, 
			levels);

	fprintf(stderr, "number of translatable addresses: %d\n", table_lowest_addresses);
	fprintf(stderr, "total size of page table: %d\n", max_table);

}
