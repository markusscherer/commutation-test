#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>

#include "bitset_function.hpp"
#include "code_generators.hpp"
#include "misc_tools.hpp"

const uint64_t MAX_DOMAIN_SIZE = 4;
const uint64_t MAX_ARITY = 4;

enum mode { PACK, PRINT, UNSET };

template <uint64_t D, uint64_t A> struct parse_and_write {
    static void run(uint64_t count) {
        for (uint64_t c = 0; c < count; ++c) {
            bitset_function<D, A> f;
            f.storage.set();

            typedef typename bitset_function<D, A>::element_type element_type;

            std::array<element_type, A> args;
            element_type res;

            for (uint64_t i = 0; i < pow(D, A); ++i) {
                for (uint64_t p = 0; p < A; ++p) {
                    try_read(args[p]);
                    check(args[p] < D, "Argument value " + to_string(args[p]) +
                          " at position " + to_string(p) +
                          " lies outside of domain " + to_string(D) +
                          ".");
                }

                if (try_read(res)) {
                    check(res < D, "Result value " + to_string(res) +
                          " lies outside of domain " + to_string(D) + ".");
                    f.set(args, res);
                } else {
                    expect_string("end");
                    break;
                }
            }

            auto arr = bitset_function_to_array<D, A, uint8_t>(f);
            std::cout.write(reinterpret_cast<const char *>(arr.data()), arr.size());
        }
    }
};

template <uint64_t D, uint64_t A> struct parse_and_write_compact {
    static void run(uint64_t count) {
        typedef typename bitset_function<D, A>::element_type element_type;
        std::array<char, D> valid_values;

        for (uint64_t i = 0; i < D; ++i) {
            valid_values[i] = '0' + i;
        }

        for (uint64_t c = 0; c < count; ++c) {
            bitset_function<D, A> f;
            f.storage.set();
            std::array<element_type, A> args;
            args.fill(0);

            std::string line;
            std::cin >> line;

            if (line.length() == 0) {
                continue;
            }

            check(line.length() == cpow(D, A),
                  "Line length should be " + to_string(cpow(D, A)) + " but is " +
                  to_string(line.length()) + ".");

            for (uint64_t i = 0; i < pow(D, A); ++i) {
                check(contains(valid_values, line[i]),
                      "Result value " + std::string(1, line[i]) +
                      " lies outside of domain " + to_string(D) + ".");
                f.set(args, line[i] - '0');
                increment_array<D, A, element_type>(args);
            }

            auto arr = bitset_function_to_array<D, A, uint8_t>(f);
            std::cout.write(reinterpret_cast<const char *>(arr.data()), arr.size());
        }
    }
};

template <uint64_t D, uint64_t A> struct read_and_print {
    static void run(uint64_t) {
        const uint64_t array_size = space_per_function<D, A, uint8_t>::of_type;
        std::vector<bitset_function<D, A>> vec;
        typedef typename bitset_function<D, A>::element_type element_type;
        std::array<uint8_t, array_size> arr;

        while (!std::cin.eof()) {
            std::cin.read(reinterpret_cast<char *>(arr.data()), array_size);

            if (std::cin.gcount() == 0) {
                break;
            }

            if (std::cin.fail()) {
                std::cerr << "Failed to extract function number " << vec.size() << "."
                          << std::endl;
                exit(1);
            }

            vec.push_back(array_to_bitset_function<D, A, uint8_t>(arr));
        }

        std::cout << "count " << vec.size() << std::endl;
        std::cout << "domain_size " << D << std::endl;
        std::cout << "arity " << A << std::endl;

        std::array<element_type, A> args;
        args.fill(0);

        for (auto f = vec.begin(); f != vec.end(); ++f) {
            for (uint64_t l = 0; l < pow(D, A); ++l) {
                for (uint64_t i = 0; i < A; ++i) {
                    std::cout << args[i] << " ";
                }

                std::cout << f->eval(args) << std::endl;
                increment_array<D, A, element_type>(args);
            }

            std::cout << std::endl;
        }
    }
};

template <uint64_t D, uint64_t A> struct read_and_print_compact {
    static void run(uint64_t) {
        const uint64_t array_size = space_per_function<D, A, uint8_t>::of_type;
        std::vector<bitset_function<D, A>> vec;
        typedef typename bitset_function<D, A>::element_type element_type;
        std::array<uint8_t, array_size> arr;

        while (!std::cin.eof()) {
            std::cin.read(reinterpret_cast<char *>(arr.data()), array_size);

            if (std::cin.gcount() == 0) {
                break;
            }

            if (std::cin.fail()) {
                std::cerr << "Failed to extract function number " << vec.size() << "."
                          << std::endl;
                exit(1);
            }

            vec.push_back(array_to_bitset_function<D, A, uint8_t>(arr));
        }

        std::cout << "count " << vec.size() << std::endl;
        std::cout << "domain_size " << D << std::endl;
        std::cout << "arity " << A << std::endl;

        std::array<element_type, A> args;
        args.fill(0);

        for (auto f = vec.begin(); f != vec.end(); ++f) {
            for (uint64_t l = 0; l < pow(D, A); ++l) {
                std::cout << f->eval(args);
                increment_array<D, A, element_type>(args);
            }

            std::cout << std::endl;
        }
    }
};

void usage() {
    std::cout << "usage: packer -k | --pack [-i | --in-file FILENAME] [-o | "
              << "--out-file FILENAME]" << std::endl;
    std::cout
            << "       packer -t | --print -a | --arity INT -d | --domain-size INT "
            << std::endl
            << "                     "
            << "[-i | --in-file FILENAME] [-o | --out-file FILENAME]" << std::endl;
    std::cout << "       packer -h | --help" << std::endl;
    std::cout << std::endl;
    std::cout << "Read function in human-readable format from in-file and and "
              << "print it in packed format to out-file." << std::endl
              << std::endl;
    std::cout << "Read function in packed format from in-file and and "
              << "print it in human-readable to out-file." << std::endl
              << std::endl;
    std::cout << "Print this help" << std::endl << std::endl;
    std::cout << "If no filenames are given, stdin and stdout are assumed."
              << std::endl
              << std::endl;
    std::cout << "Maximum arity for this executable: " << MAX_ARITY << std::endl;
    std::cout << "Maximum domain size for this executable: " << MAX_DOMAIN_SIZE
              << std::endl;
}

int main(int argc, char **argv) {
    std::stringstream argparse;

    for (int i = 1; i < argc; ++i) {
        argparse << argv[i] << " ";
    }

    std::string s;
    std::string infilename;
    std::string outfilename;
    uint64_t count = -1;
    uint64_t domain_size = -1;
    uint64_t arity = -1;
    mode m = UNSET;

    std::ofstream out;
    std::ifstream in;
    std::streambuf *cinbuf = std::cin.rdbuf();
    std::streambuf *coutbuf = std::cout.rdbuf();
    bool compact = false;

    while (!argparse.eof()) {
        argparse >> s;

        if (argparse.bad() || argparse.eof()) {
            break;
        }

        if (s == "--help" || s == "-h") {
            usage();
            return 0;
        } else if (s == "--pack" || s == "-k") {
            if (m == UNSET) {
                m = PACK;
            } else {
                std::cerr << "More than one operation specified." << std::endl;
                usage();
                return 0;
            }
        } else if (s == "--print" || s == "-t") {
            if (m == UNSET) {
                m = PRINT;
            } else {
                std::cerr << "More than one operation specified." << std::endl;
                usage();
                return 0;
            }
        } else if (s == "--arity" || s == "-a") {
            safe_read(arity, argparse);
        } else if (s == "--domain-size" || s == "-d") {
            safe_read(domain_size, argparse);
        } else if (s == "--in-file" || s == "-i") {
            argparse >> infilename;
        } else if (s == "--out-file" || s == "-o") {
            argparse >> outfilename;
        } else if (s == "-c" || s == "--compact") {
            compact = true;
        }
    }

    if (m == UNSET) {
        std::cerr << "Specify mode with --pack/--print" << std::endl;
        usage();
        return 0;
    } else if (m == PRINT) {
        if (arity > MAX_ARITY) {
            std::cerr << "Arity to large or unspecified" << std::endl;
            usage();
            return 0;
        }

        if (domain_size > MAX_DOMAIN_SIZE) {
            std::cerr << "Domain size to large or unspecified" << std::endl;
            usage();
            return 0;
        }
    }

    if (!infilename.empty()) {
        in.open(infilename);
        std::cin.rdbuf(in.rdbuf());
    }

    if (!outfilename.empty()) {
        out.open(outfilename);
        std::cout.rdbuf(out.rdbuf());
    }

    switch (m) {
        case PACK:
            expect_string("count");
            safe_read(count);

            expect_string("domain_size");
            safe_read(domain_size);

            expect_string("arity");
            safe_read(arity);

            if (compact) {
                domain_size_select<MAX_DOMAIN_SIZE, MAX_ARITY,
                                   parse_and_write_compact>::select(domain_size, arity,
                                                                    count);
            } else {
                domain_size_select<MAX_DOMAIN_SIZE, MAX_ARITY, parse_and_write>::select(
                    domain_size, arity, count);
            }

            break;

        case PRINT:
            if (compact) {
                domain_size_select<MAX_DOMAIN_SIZE, MAX_ARITY,
                                   read_and_print_compact>::select(domain_size, arity,
                                                                   count);
            } else {
                domain_size_select<MAX_DOMAIN_SIZE, MAX_ARITY, read_and_print>::select(
                    domain_size, arity, count);
            }

            break;

        default:
            check(false, "Mode is not set. This is a programming error.");
    }

    std::cin.rdbuf(cinbuf);
    std::cout.rdbuf(coutbuf);

    return 0;
}
