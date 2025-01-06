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

Device ProcessArgs(int argc, char* argv[]);

int SplitString(const std::string& str, const char delimiter);

uchar4* LoadFile(const std::string& file, int& w, int& h);

void SaveFile(const std::string& file, std::vector<uchar4>& data, int w, int h, int number);
};