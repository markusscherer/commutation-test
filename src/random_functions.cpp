#include <iostream>
#include <string>
#include <cstdint>
#include <sstream>

#include "constants.hpp"
#include "code_generators.hpp"

const uint64_t MAX_DOMAIN_SIZE = 4;
const uint64_t MAX_ARITY = 4;

void usage() {
    std::cout
            << "usage: random_functions <domain_size> <arity> <filename> <count>"
            << std::endl;
    std::cout << std::endl;
    std::cout << "Maximum arity for this executable: " << MAX_ARITY << std::endl;
    std::cout << "Maximum domain size for this executable: " << MAX_DOMAIN_SIZE
              << std::endl;
}

std::string filename;

template <uint64_t D, uint64_t A> struct print_command {
    static void run(uint64_t count) {
        int bs = space_per_function<D, A, uint8_t>::of_type;
        std::cout << "dd if=/dev/urandom of=" << filename << "." << A << "." << D
                  << " count=" << count << " bs=" << bs << std::endl;
    }
};

int main(int argc, char **argv) {
    if (argc != 5) {
        usage();
        return 1;
    }

    std::stringstream argparse;

    for (int i = 1; i < argc; ++i) {
        argparse << argv[i] << " ";
    }

    uint64_t domain_size;
    uint64_t arity;
    int count;

    safe_read(domain_size, argparse);
    safe_read(arity, argparse);
    argparse >> filename;
    safe_read(count, argparse);

    if (arity > MAX_ARITY || domain_size > MAX_DOMAIN_SIZE) {
        usage();
        return 1;
    }

    domain_size_select<MAX_DOMAIN_SIZE, MAX_ARITY, print_command>::select(
        domain_size, arity, count);
}
