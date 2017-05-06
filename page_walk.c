#include <stdio.h>
#include <sdtlib.h>
#include <unistd.h>

struct translation_thread {
	void *current_table;
	int *offset;
	int current;
}

int main(int argc, char** argv) {
	void *pg_table;
	int i, total_addresses, size prior;
	void [levels] = {0};
	
	// get size of address space
	for(i = 1; i < argc - 1; i++) {
		if ( !strcmp(argv[i], "-n")) {
			total_addresses = atoi(argv[i]);
			continue;
		}
		count *=  atoi(argv[i]);
	}
	


}
