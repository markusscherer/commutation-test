#ifndef MISC_TOOLS_HPP_MSDA
#define MISC_TOOLS_HPP_MSDA

template <typename T> std::string to_string(T t) {
    std::stringstream s;
    s << t;
    return s.str();
}

void check(bool b, std::string s) {
    if (!b) {
        std::cerr << "Critical Error: " << s << std::endl;
        exit(1);
    }
}

void expect_string(std::string expected) {
    std::string s;
    std::cin >> s;
    check(expected == s,
          "Expected \"" + expected + "\", got \"" + s + "\" instead.");
}

template <typename T> bool safe_read(T &x, std::istream &in = std::cin) {
    in >> x;
    check(in.good(), "In safe_read (most likely expected numeral).");
    return true;
}

template <typename T> bool try_read(T &x) {
    std::cin >> x;

    if (!std::cin.good()) {
        std::cin.clear();
        return false;
    }

    return true;
}

#endif
