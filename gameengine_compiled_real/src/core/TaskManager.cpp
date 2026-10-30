// FILEPATH: src/core/TaskManager.cpp
#include "TaskManager.h"

void TaskManager::enqueueTask(std::function<void()> task) {
    tasks.push_back(task);
}

void TaskManager::executeAllTasks() {
    for (auto& task : tasks) {
        task();
    }
    tasks.clear();
}
