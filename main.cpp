#include <stdio.h>
#include <vector>
#include <array>
#include <random>
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

#define TEST_IMAGE_SIZE() 128 // in pixels, on each axis

static const float c_pi = 3.14159265359f;

struct Vec2
{
    float x, y;
};

typedef uint8_t uint8;

typedef void (*GeneratePoints) (std::vector<Vec2>& points, size_t numPoints);
typedef float (*SampleImage) (const Vec2& samplePos);

std::mt19937& RNG()
{
    static std::random_device rd;
    static std::mt19937 rng(rd());
    return rng;
}

inline float Clamp(float x, float min, float max)
{
    if (x <= min)
        return min;
    else if (x >= max)
        return max;
    else
        return x;
}

inline float Lerp(float A, float B, float t)
{
    return A * (1.0f - t) + B * t;
}

void GeneratePoints_WhiteNoise(std::vector<Vec2>& points, size_t numPoints)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    points.resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        points[i].x = dist(RNG());
        points[i].y = dist(RNG());
    }
}

// reference value is 0.5
float SampleImage_Disk(const Vec2& samplePos)
{
    return ((samplePos.x * samplePos.x + samplePos.y * samplePos.y) < 2.0f / c_pi) ? 1.0f : 0.0f;
}

// reference value is 0.5
float SampleImage_Triangle(const Vec2& samplePos)
{
    return (samplePos.y > samplePos.x) ? 1.0f : 0.0f;
}

// reference value is 1/pi
float SampleImage_Step(const Vec2& samplePos)
{
    return (samplePos.x < 1.0f / c_pi) ? 1.0f : 0.0f;
}

// reference value is pi/4 * erf^2(1)
float SampleImage_Gaussian(const Vec2& samplePos)
{
    return expf(-(samplePos.x*samplePos.x)-(samplePos.y*samplePos.y));
}

// reference value is 0.25
float SampleImage_Bilinear(const Vec2& samplePos)
{
    return samplePos.x * samplePos.y;
}

template <typename T>
void MakeSampleImage(const T& SampleImage, const char* fileName)
{
    std::vector<uint8> image;
    image.resize(TEST_IMAGE_SIZE()*TEST_IMAGE_SIZE());
    uint8* pixel = image.data();

    float halfPixel = 0.5f / float(TEST_IMAGE_SIZE());

    Vec2 samplePos;

    for (size_t iy = 0; iy < TEST_IMAGE_SIZE(); ++iy)
    {
        samplePos.y = halfPixel + float(iy) / float(TEST_IMAGE_SIZE());

        for (size_t ix = 0; ix < TEST_IMAGE_SIZE(); ++ix)
        {
            samplePos.x = halfPixel + float(ix) / float(TEST_IMAGE_SIZE());

            *pixel = uint8(Clamp(0.5f + SampleImage(samplePos) * 255.0f, 0.0f, 255.0f));

            ++pixel;
        }
    }

    stbi_write_png(fileName, TEST_IMAGE_SIZE(), TEST_IMAGE_SIZE(), 1, image.data(), 0);
}

template <typename T>
void Integrate(const T& SampleImage, const std::vector<Vec2>& points, const char* name)
{
    float result = 0.0f;
    for (size_t i = 0; i < points.size(); ++i)
    {
        float sample = SampleImage(points[i]);
        result = Lerp(result, sample, 1.0f / float(i + 1));
    }
    printf("%s: %f\n", name, result);
}

void DoTest2D (const GeneratePoints& generatePoints)
{
    std::vector<Vec2> points;
    generatePoints(points, 100000);

    Integrate(SampleImage_Disk, points, "Disk");
    Integrate(SampleImage_Triangle, points, "Triangle");
    Integrate(SampleImage_Step, points, "Step");
    Integrate(SampleImage_Gaussian, points, "Gaussian");
    Integrate(SampleImage_Bilinear, points, "Bilinear");
}

int main(int argc, char **argv)
{
    MakeSampleImage(SampleImage_Disk, "out/disk.png");
    MakeSampleImage(SampleImage_Triangle, "out/triangle.png");
    MakeSampleImage(SampleImage_Step, "out/step.png");
    MakeSampleImage(SampleImage_Gaussian, "out/gaussian.png");
    MakeSampleImage(SampleImage_Bilinear, "out/bilinear.png");

    DoTest2D(GeneratePoints_WhiteNoise);

    system("pause");

    return 0;
}