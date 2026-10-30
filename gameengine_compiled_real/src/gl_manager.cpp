// FILEPATH: src/gl_manager.cpp

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

void initOpenGL() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return;
    }

    GLFWwindow* window = glfwCreateWindow(1024, 768, "Compute Shader Example", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return;
    }
    
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        glfwDestroyWindow(window);
       glfwTerminate();
        return;
    }

    // Create buffer for PSF data
    GLuint psfBuffer;
    glGenBuffers(1, &psfBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, psfBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 262144 * sizeof(float), nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psfBuffer);
}

#include <CL/cl.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>

GLuint psfBuffer;

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenCL and OpenGL Example", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Create a buffer object
    glGenBuffers(1, &psfBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, psfBuffer);
    float* data = new float[262144];
    for (int i = 0; i < 262144; ++i) {
        data[i] = static_cast<float>(i % 513); // Example initialization
    }
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * 262144, data, GL_DYNAMIC_DRAW);
    delete[] data;

    initOpenGL();

    // Load shader program
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    const char* source = R"(
#version 450 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer PSFBuffer {
    float psf_data[];
};

void main() {
    uvec2 global_id = gl_GlobalInvocationID.xy;
    if (global_id.x >= 512 || global_id.y >= 512) return;

    int index = global_id.y * 512 + global_id.x;
    float sum = 0.0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (global_id.x + i >= 0 && global_id.x + i < 512 &&
                global_id.y + j >= 0 && global_id.y + j < 512) {
                sum += psf_data[(global_id.y + j) * 512 + (global_id.x + i)];
            }
        }
    }

    atomicAdd(psf_data[index], sum);
}
)";

    glShaderSource(computeShader, 1, &source, nullptr);
    glCompileShader(computeShader);

    // Check for compilation errors
    GLint success;
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(computeShader, 512, nullptr, infoLog);
        std::cerr << "Error in shader compilation: " << infoLog << std::endl;
        return -1;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, computeShader);
    glLinkProgram(program);

    // Check for linking errors
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Error in shader linking: " << infoLog << std::endl;
        return -1;
    }

    // Use the compute program
    glUseProgram(program);

    // Dispatch compute shader workgroups
    glDispatchCompute(32, 32, 1); // Adjust based on actual data size

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, psfBuffer);
    float* result = static_cast<float*>(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY));
    
    // Output results or handle them as needed
    for (int i = 0; i < 262144; ++i) {
        std::cout << "Result[" << i << "] = " << result[i] << std::endl;
    }

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    
    // Clean up
    glDeleteShader(computeShader);
    glDeleteProgram(program);

    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
