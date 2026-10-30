// FILEPATH: src/core/Settings.h
#ifndef SETTINGS_H
#define SETTINGS_H

#include <string>

struct Settings {
    std::string configurationFile;
    int resolutionWidth;
    int resolutionHeight;

    Settings(const std::string& configFile, int width, int height)
        : configurationFile(configFile), resolutionWidth(width), resolutionHeight(height) {}
};

#endif // SETTINGS_H
