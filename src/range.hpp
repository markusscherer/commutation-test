#pragma once

#include <map>
#include <string>
#include <cstdint>
#include <iostream>
#include <algorithm>

#include "misc_tools.hpp"

struct range {
    uint64_t startA, endA, startB, endB;
};

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

// Takes a string that conforms in this format: [x]-[y]/a#[u]-[v]/b
// If s conforms to this format, a range object is added to ranges where the
// A-values correspond to the range specifier of lower arity
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
    key.second = std::max(leftarity, rightarity);

    if (key.first == leftarity) {
        ranges[key] = r;
    } else {
        std::swap(r.startA, r.startB);
        std::swap(r.endA, r.endB);
        ranges[key] = r;
    }
}
