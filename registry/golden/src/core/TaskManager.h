// FILEPATH: src/core/TaskManager.h
#ifndef TASKMANAGER_H
#define TASKMANAGER_H

#include <vector>
#include <functional>

class TaskManager {
public:
    void enqueueTask(std::function<void()> task);
    void executeAllTasks();

private:
    std::vector<std::function<void()>> tasks;
};

#endif // TASKMANAGER_H
