#include <array>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <string>
#include <sstream>
#include <limits>
#include <algorithm>
#include "array_function.hpp"
#include "constants.hpp"
#include "bitset_function.hpp"
#include "misc_tools.hpp"

#include "comtest.config.hpp"

std::map<std::string, std::set<std::string>> matches;

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

struct range {
    uint64_t startA, endA, startB, endB;
};

template <uint64_t D, uint64_t A1, uint64_t A2>
void commutation_test(std::vector<array_function<D, A1, uint8_t>>& vec1,
                      std::vector<array_function<D, A2, uint8_t>>& vec2) {
    #pragma omp parallel for

    for (uint64_t i = 0; i < vec1.size(); ++i) {
        for (uint64_t j = 0; j < vec2.size(); ++j) {
            /* if(A1 != A2 && j < i) { */
            /*   break; */
            /* } */
            if (solver<D, A1, A2>::commutes(vec1[i], vec2[j])) {
                #pragma omp critical
                {
                    std::string id1 = to_string(i) + "/" + to_string(A1);
                    std::string id2 = to_string(j) + "/" + to_string(A2);

                    matches[id1].insert(id2);
                    matches[id2].insert(id1);
                }
            }
        }
    }
}

template <uint64_t D, uint64_t A1, uint64_t A2>
void commutation_test(std::vector<array_function<D, A1, uint8_t>>& vec1,
                      std::vector<array_function<D, A2, uint8_t>>& vec2,
                      const range& r) {
    range ar;
    ar.startA = std::min(r.startA, vec1.size());
    ar.startB = std::min(r.startB, vec2.size());
    ar.endA = std::min(r.endA, vec2.size());
    ar.endB = std::min(r.endB, vec2.size());

    for (uint64_t i = ar.startA; i < ar.endA; ++i) {
        for (uint64_t j = A1 != A2 ? ar.startB : std::max(i, ar.startB);
                j < ar.endB; ++j) {
            std::string id1 = to_string(i) + "/" + to_string(A1);
            std::string id2 = to_string(j) + "/" + to_string(A2);

            if (solver<D, A1, A2>::commutes(vec1[i], vec2[j])) {
                matches[id1].insert(id2);
                matches[id2].insert(id1);
            }
        }
    }
}

const uint64_t MAX_ARITY = 4;

bool parse_range_component(uint64_t& val, uint64_t defaultval, std::string s) {
    if (s.empty()) {
        val = defaultval;
    } else {
        if (!try_read(val, s)) {
            std::cerr << "Failed to range parse argument. Parameter ignored."
                      << std::endl;
            return false;
        }
    }

    return true;
}

bool parse_range_component(uint64_t& arity, uint64_t& leftval,
                           uint64_t& rightval, std::string s) {
    size_t slash = s.find("/");

    if (slash == std::string::npos) {
        std::cerr << "Both sides have to contain '/'. Parameter ignored."
                  << std::endl;
        return false;
    }

    if (!try_read(arity, s.substr(slash + 1))) {
        std::cerr << "Failed to parse arities. Parameter ignored." << std::endl;
        return false;
    }

    std::string range = s.substr(0, slash);
    size_t dash = range.find("-");

    std::string left = range.substr(0, dash);
    std::string right = range.substr(dash + 1);

    bool ret = parse_range_component(leftval, 0, left) &&
               parse_range_component(rightval,
                                     std::numeric_limits<uint64_t>::max(), right);
    return ret;
}

void parse_range(std::map<std::pair<uint64_t, uint64_t>, range>& ranges,
                 std::string s) {
    size_t middle = s.find("#");

    if (middle == std::string::npos) {
        std::cerr << "Range has to contain '#'. Parameter ignored." << std::endl;
        return;
    }

    std::string left = s.substr(0, middle);
    std::string right = s.substr(middle + 1);

    uint64_t leftarity;
    uint64_t rightarity;
    range r;

    bool ret = parse_range_component(leftarity, r.startA, r.endA, left) &&
               parse_range_component(rightarity, r.startB, r.endB, right);

    if (!ret) {
        return;
    }

    std::pair<uint64_t, uint64_t> key;
    key.first = std::min(leftarity, rightarity);
    key.second = std::min(leftarity, rightarity);

    if (key.first == leftarity) {
        ranges[key] = r;
    } else {
        std::swap(r.startA, r.startB);
        std::swap(r.endA, r.endB);
        ranges[key] = r;
    }
}

template <class A, class B, class C, class D>
void test_all(A v1, B v2, C v3, D v4) {

    if (!v1.empty()) {
        commutation_test(v1, v1);

        if (!v2.empty()) {
            commutation_test(v1, v2);
        }

        if (!v3.empty()) {
            commutation_test(v1, v3);
        }

        if (!v4.empty()) {
            commutation_test(v1, v4);
        }
    }

    if (!v2.empty()) {
        commutation_test(v2, v2);

        if (!v3.empty()) {
            commutation_test(v2, v3);
        }

        if (!v4.empty()) {
            commutation_test(v2, v4);
        }
    }

    if (!v3.empty()) {
        commutation_test(v3, v3);

        if (!v4.empty()) {
            commutation_test(v3, v4);
        }
    }

    if (!v4.empty()) {
        commutation_test(v4, v4);
    }
}

int main(int argc, char **argv) {
    const uint64_t D = 4;

    if (argc < 2) {
        return 1;
    }

    std::stringstream argparse;

    for (int i = 1; i < argc; ++i) {
        argparse << argv[i] << " ";
    }

    std::string s;
    std::string infilenames[MAX_ARITY];
    std::map<std::pair<uint64_t, uint64_t>, range> ranges;

    while (!argparse.eof()) {
        argparse >> s;

        if (argparse.bad() || argparse.eof()) {
            break;
        }

        if (s == "--functions-1" || s == "-1") {
            if (infilenames[0].empty()) {
                argparse >> infilenames[0];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--functions-2" || s == "-2") {
            if (infilenames[1].empty()) {
                argparse >> infilenames[1];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--functions-3" || s == "-3") {
            if (infilenames[2].empty()) {
                argparse >> infilenames[2];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--functions-4" || s == "-4") {
            if (infilenames[3].empty()) {
                argparse >> infilenames[3];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--range" || s == "-r") {
            argparse >> s;
            parse_range(ranges, s);
        }
    }

    std::vector<array_function<D, 1, uint8_t>> v1;
    std::vector<array_function<D, 2, uint8_t>> v2;
    std::vector<array_function<D, 3, uint8_t>> v3;
    std::vector<array_function<D, 4, uint8_t>> v4;

    if (!infilenames[0].empty()) {
        v1 = read_functions<4, 1>(infilenames[0]);
    }

    if (!infilenames[1].empty()) {
        v2 = read_functions<4, 2>(infilenames[1]);
    }

    if (!infilenames[2].empty()) {
        v3 = read_functions<4, 3>(infilenames[2]);
    }

    if (!infilenames[3].empty()) {
        v4 = read_functions<4, 4>(infilenames[3]);
    }

    if (ranges.empty()) {
        test_all(v1, v2, v3, v4);
    } else {
        for (const auto& r : ranges) {
            const uint64_t a1 = r.first.first;
            const uint64_t a2 = r.first.second;

            if (a1 == 1) {
                if (a2 == 1) {
                    commutation_test(v1, v1, r.second);
                } else if (a2 == 2) {
                    commutation_test(v1, v2, r.second);
                } else if (a2 == 3) {
                    commutation_test(v1, v3, r.second);
                } else if (a2 == 4) {
                    commutation_test(v1, v4, r.second);
                }
            } else if (a1 == 2) {
                if (a2 == 2) {
                    commutation_test(v2, v2, r.second);
                } else if (a2 == 3) {
                    commutation_test(v2, v3, r.second);
                } else if (a2 == 4) {
                    commutation_test(v2, v4, r.second);
                }
            } else if (a1 == 3) {
                if (a2 == 3) {
                    commutation_test(v3, v3, r.second);
                } else if (a2 == 4) {
                    commutation_test(v3, v4, r.second);
                }
            } else if (a1 == 4 && a2 == 4) {
                commutation_test(v4, v4, r.second);
            }
        }
    }

    std::cout << matches.size() << std::endl;

    for (auto& k : matches) {
        std::cout << k.first;
        std::cout << ":";

        for (auto& c : k.second) {
            std::cout << c << " ";
        }

        std::cout << std::endl;
    }

    return 0;
}
