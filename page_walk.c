#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <sys/time.h>

#define MAX_LEVELS 20

static int max_table;

struct trans_thread {
	void **curr_table;
	int offset[MAX_LEVELS];
	int curr;
	int max;
};

int translate_cpu(struct trans_thread *trans) {
	void **c = trans->curr_table;
	while(trans->curr < trans->max-1) {
		trans->curr_table = (void **) trans->curr_table[trans->offset[trans->curr]];
		trans->curr++;
		fprintf(stderr, "offset: %d\n", trans->offset[trans->curr]);
	}

	return (int) *(trans->curr_table + trans->offset[trans->curr]); // ((void *) trans->curr_table + trans->offset[trans->max-1]);
}

float cpu_run_time(struct trans_thread *trans, int addresses) {
	int i;
       	struct timeval start_time, stop_time;
	
	gettimeofday(&start_time, NULL);

	for(i = 0; i < addresses; i++) {
		translate_cpu(&trans[i]);
	}

	gettimeofday(&stop_time, NULL);

	long time_diff = ((stop_time.tv_sec * 1000000) + stop_time.tv_usec)
		-((start_time.tv_sec * 1000000) + start_time.tv_usec);	
	return time_diff / 1000000.0;
}



int construct_table(void *table, int *levels, int num_levels) {
	int i, j, level_size = 1;
        void **table_ptr = (void **) table,
	    **level_ptr;

	// set intermediate addresses of table
	for(i = 0; i < num_levels-1; i++)
	{
		level_size *= levels[i];

		// hideous but best way I could find to get next level
		level_ptr = (void **) table + (level_size)+ (((void *)table_ptr - table)/(sizeof(void*)));
		// helpufl check: fprintf(stderr, "level_size: %d, level_ptr: %d, table_ptr: %d\n", level_size, (level_ptr- (void **) table) / sizeof(void *), table_ptr -  (void **) table);

		for(j = 0; j < level_size; j++) {
			table_ptr[j] = level_ptr + ((j)*levels[i+1]);
		}	

		table_ptr += level_size;
	}
	assert((void *) table_ptr - table < max_table);


	// set last level of page table to garbage; for our purposes
	// it doesn't matter
	for(i = 0; i < level_size * levels[num_levels-1]; i++) {
		*table_ptr = (void *) i;
		table_ptr++;
	}


	assert((void *) table_ptr - table == max_table);
	
	// return number of entries at the lowest level of the
	// page table
	return levels[num_levels-1] * level_size;
}

int main(int argc, char** argv) {
	void **pg_table;
	int i, j, table_size = 0, level_size = 1, 
	    total_addresses, table_lowest_addresses, 
	    levels = argc-3;
	int level_sizes[levels];
	struct trans_thread *sample;
	
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

	/*for(i = 0; i < table_size; i++) {
		fprintf(stderr, "address number: %d, address intermediate value: %d, final value: %d, address: %d\n", i,  ((void **) pg_table[i] - pg_table) / sizeof(void *), pg_table[i], pg_table+i);
	}

	for (i = 0; i <  table_size - table_lowest_addresses; i++)
	{
		fprintf(stderr, "address %d, points to %d\n", pg_table + i, pg_table[i]);
	}*/

	fprintf(stderr, "number of translatable addresses: %d\n", table_lowest_addresses);
	fprintf(stderr, "total size of page table: %d\n", max_table);
	
	sample = (struct trans_thread *) malloc(sizeof(struct trans_thread));

	sample->curr_table = pg_table;
	sample->offset[0] = 1;
	sample->offset[1] = 1;
	sample->offset[2] = 1;
	sample->offset[3] = 1;
	sample->curr = 0;
	sample->max = 4;

	int sample_test = translate_cpu(sample);
	fprintf(stderr, "translated address: %d\n", sample_test );



}
