// FILEPATH: src/io/PSFFile.cpp
#include "PSFFile.h"
#include <fstream>
#include <sstream>

bool PSFFile::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Load the PSF file contents here
    // Example:
    std::stringstream buffer;
    buffer << file.rdbuf();
    content = buffer.str();

    file.close();
    return true;
}
