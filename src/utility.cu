#include <fstream>
#include <utility.cuh>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace render
{
    
Device ProcessArgs(int argc, char* argv[])
{
    if (argc > 2)
    {
        throw std::runtime_error{"too many argument"};
    }
    else if(argc == 2)
    {
        if(std::string(argv[1]) == "--cpu")
        {
            return Device::CPU;
        }
        else if (std::string(argv[1]) == "--gpu")
        {
            return Device::GPU;
        }
        else if (std::string(argv[1]) == "--default")
        {
            return Device::DEFAULT;
        }
        else
        {
            throw std::runtime_error{"Unknown argument\nUsage:\nprogram --cpu\nprogram --gpu\nprogram --default\nprogram"};
        }
    }

    return Device::GPU;
}

int SplitString(const std::string& str, const char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    std::getline(ss, token, delimiter);
    return std::stoi(token);
}

uchar4* LoadFile(const std::string& file, int& w, int& h)
{
    std::ifstream fin(file, std::ios::binary);
    if (!fin)
    {
        throw std::runtime_error{"file not found"};
    }

    fin.read(reinterpret_cast<char*>(&w), sizeof(int));
    fin.read(reinterpret_cast<char*>(&h), sizeof(int));

    uchar4* data = new uchar4[h * w];
    fin.read(reinterpret_cast<char*>(data), sizeof(uchar4) * w * h);
    fin.close();
    return data;
}

std::string replacePlaceholderWithNumber(const std::string& input, int number) {
    std::string numberStr = std::to_string(number);
    std::string result = input;
    size_t pos = 0;
    while ((pos = result.find("%d", pos)) != std::string::npos) {
        result.replace(pos, 2, numberStr);
        pos += numberStr.length();
    }
    return result;
}

void SaveFile(const std::string& file, std::vector<uchar4>& data, int w, int h, int number)
{
    auto filename = replacePlaceholderWithNumber(file, number);
    std::ofstream fout(filename, std::ios::binary);
    fout.write(reinterpret_cast<char*>(&w), sizeof(int));
    fout.write(reinterpret_cast<char*>(&h), sizeof(int));
    fout.write(reinterpret_cast<char*>(data.data()), sizeof(uchar4) * w * h);
}

}