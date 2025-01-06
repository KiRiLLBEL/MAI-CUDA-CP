#pragma once

#include <string>
#include <fstream>
#include <vector>

namespace render
{

#define EPS 1e-6

enum Device
{
    GPU,
    CPU,
    DEFAULT
};

struct Timer
{
    float elapsedTimeMs;
    cudaEvent_t eventStart, eventStop;

    Timer()
    {
        cudaEventCreate(&eventStart);
        cudaEventCreate(&eventStop);
    }

    void begin()
    {
        cudaEventRecord(eventStart);
    }

    void finish()
    {
        cudaDeviceSynchronize();
        cudaEventRecord(eventStop);
        cudaEventSynchronize(eventStop);
    }

    float elapsed()
    {
        cudaEventElapsedTime(&elapsedTimeMs, eventStart, eventStop);
        return elapsedTimeMs;
    }

    ~Timer()
    {
        cudaEventDestroy(eventStart);
        cudaEventDestroy(eventStop);
    }
};

Device ProcessArgs(int argc, char* argv[]);

int SplitString(const std::string& str, const char delimiter);

uchar4* LoadFile(const std::string& file, int& w, int& h);

void SaveFile(const std::string& file, std::vector<uchar4>& data, int w, int h, int number);
};