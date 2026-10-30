// FILEPATH: src/core/Settings.cpp
#include "Settings.h"

Settings loadSettings() {
    return Settings("settings.cfg", 1920, 1080);
}
