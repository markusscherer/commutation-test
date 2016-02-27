#pragma once

#include <cstdint>
#include <set>
#include <map>
#include <string>

struct match_map {
    typedef std::pair<uint64_t, uint64_t> function_identifier;
    std::map<function_identifier, std::set<function_identifier>> storage;

    void add(uint64_t x, uint64_t y, uint64_t a1, uint64_t a2) {
        auto a = function_identifier(x, a1);
        auto b = function_identifier(y, a2);

        storage[a].insert(b);
        storage[b].insert(a);
    }
};

match_map join_matches(const match_map& x, const match_map& y) {
    match_map ret(x);

    for (const auto& m : y.storage) {
        for (const auto& mm : m.second) {
            ret.storage[m.first].insert(mm);
        }
    }

    return ret;
}
