#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include "common.h"

using namespace std;

int main(int argc, char **argv) {
	// move this code to FIB class constructor. 
	std::string fib_fname = argv[1];
	std::cout << "filename: " << fib_fname << std::endl;
	std::ifstream ff(fib_fname);

	if (ff.is_open())
	{
		// each line has IP_arr NextHop Port : all strings.

	}
}

// given keys, values, nn, M : find optimal BF sizes based on OPT problem.
template<int VL, class Val>
std::vector<float> get_opt_bf_sizes(vector<Key> &keys, vector<Val> &values, uint64_t nn, uint64_t memory)
{
	// find #keys for each value, find top [5], 
	std::map<Val, uint64_t> per_val_key_ct;
}
