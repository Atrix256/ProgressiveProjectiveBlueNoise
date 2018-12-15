#pragma once

#include <vector>
#include <stdio.h>
#include <stdint.h>

// default values recommended by http://isthe.com/chongo/tech/comp/fnv/
static const uint32_t Prime = 0x01000193; //   16777619
static const uint32_t Seed = 0x811C9DC5; // 2166136261
/// hash a single byte
static inline uint32_t fnv1a(unsigned char oneByte, uint32_t hash = Seed)
{
    return (oneByte ^ hash) * Prime;
}

/// hash a block of memory
static inline uint32_t fnv1a(const void* data, size_t numBytes, uint32_t hash = Seed)
{
    const unsigned char* ptr = (const unsigned char*)data;
    while (numBytes--)
        hash = fnv1a(*ptr++, hash);
    return hash;
}

template <typename LAMDBA, typename T>
void MakeDataCached(const LAMDBA& lambda, const T& params, std::vector<unsigned char>& buffer)
{
    // get the hash of the data
    uint32_t hash = fnv1a(&params, sizeof(T));

    // load it from the cache if it exists
    char fileName[256];
    sprintf_s(fileName, "cache/%x", hash);
    FILE* file = nullptr;
    fopen_s(&file, fileName, "rb");
    if (file)
    {
        fseek(file, 0, SEEK_END);
        buffer.resize(ftell(file));
        fseek(file, 0, SEEK_SET);
        fread(buffer.data(), buffer.size(), 1, file);
        fclose(file);
        return;
    }

    // else generate it
    lambda(params, buffer);

    // now save it to the cache
    fopen_s(&file, fileName, "wb");
    if (file)
    {
        fwrite(buffer.data(), buffer.size(), 1, file);
        fclose(file);
    }
}