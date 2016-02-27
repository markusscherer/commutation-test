#ifndef ARRAY_FUNCTION_HPP_MSDA
#define ARRAY_FUNCTION_HPP_MSDA

#include <cstdint>
#include <array>
#include <fstream>

#include "constants.hpp"
#include "bitset_function.hpp"

template <uint64_t D, uint64_t A, typename ElementType> class array_function {
public:
    typedef ElementType element_type;
    static const uint64_t domain_size = D;
    static const uint64_t arity = A;
    static const uint64_t storage_elements =
        space_per_function<D, A, element_type>::of_type;

    std::array<ElementType, storage_elements> storage;

    array_function() {};

    template <typename T>
    explicit array_function(bitset_function<D, A, T>& in)
        : storage(bitset_function_to_array(in)) {
    }
};

template <uint64_t D, uint64_t A>
std::vector<array_function<D, A, uint8_t>>
read_functions(std::string filename) {
    const uint64_t array_size = space_per_function<D, A, uint8_t>::of_type;
    std::vector<array_function<D, A, uint8_t>> vec;

    std::ifstream in;
    in.open(filename, std::ifstream::ate);
    std::streampos size = in.tellg();
    in.seekg(std::ios_base::beg);

    if (size % array_size != 0) {
        std::cerr << "File contains incomplete function." << std::endl;
    }

    vec.resize(size / array_size);

    uint64_t counter = 0;

    while (!in.eof()) {
        in.read(reinterpret_cast<char *>(vec[counter].storage.data()), array_size);

        if (in.gcount() == 0) {
            break;
        }

        if (in.fail()) {
            std::cerr << "Failed to extract function number " << vec.size() << "."
                      << std::endl;
            exit(1);
        }

        ++counter;
    }

    return vec;
}
#endif
