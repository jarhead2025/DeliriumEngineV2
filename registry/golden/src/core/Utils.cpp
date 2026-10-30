// FILEPATH: src/core/Utils.cpp
#include "Utils.h"
#include <algorithm>
#include <cctype>

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return ""; // No content

    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

bool isNumber(const std::string& str) {
    for (size_t i = 0; i < str.size(); ++i) {
        if (!std::isdigit(str[i])) return false;
    }
    return !str.empty();
}
