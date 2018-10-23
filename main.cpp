#include <stdio.h>
#include <vector>
#include <array>
#include <random>
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

#define TEST_IMAGE_SIZE() 128 // in pixels, on each axis
#define NUM_SAMPLES() 100

static const float c_pi = 3.14159265359f;

static const float c_referenceValue_Disk = 0.5f;
static const float c_referenceValue_Triangle = 0.5f;
static const float c_referenceValue_Step = 1.0f / c_pi;
static const float c_referenceValue_Gaussian = c_pi / 4.0f * (float)erf(1.0) * (float)erf(1.0);
static const float c_referenceValue_Bilinear = 0.25f;

// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
static const float c_goldenRatio2 = 1.32471795724474602596f;

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

void GeneratePoints_GoldenRatio(std::vector<Vec2>& points, size_t numPoints)
{
    static const float a1 = 1.0f / c_goldenRatio2;
    static const float a2 = 1.0f / (c_goldenRatio2 * c_goldenRatio2);

    points.resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        points[i].x = fmodf(0.5f + a1 * float(i), 1.0f);
        points[i].y = fmodf(0.5f + a2 * float(i), 1.0f);
    }
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
void Integrate(const T& SampleImage, const std::vector<Vec2>& points, float reference, const char* samplesName, const char* testName, FILE* csvFile)
{
    float result = 0.0f;
    fprintf(csvFile, "\"%s %s\"\n", samplesName, testName);
    for (int i = 0; i < points.size(); ++i)
    {
        float sample = SampleImage(points[i]);
        result = Lerp(result, sample, 1.0f / float(i + 1));

        fprintf(csvFile, "\"%i\",\"%f\",\"%f\",\"%f\"\n", i, log10f(float(i+1)), result, fabsf(result-reference));
    }
    printf("  %s: %f (error = %f)\n", testName, result, fabsf(result - reference));
    fprintf(csvFile, "\n");
}

void DoTest2D (const GeneratePoints& generatePoints, const char* samplesName, FILE* csvFile)
{
    std::vector<Vec2> points;
    generatePoints(points, NUM_SAMPLES());

    printf("%s\n", samplesName);

    Integrate(SampleImage_Disk, points, c_referenceValue_Disk, samplesName, "Disk", csvFile);
    //Integrate(SampleImage_Triangle, points, c_referenceValue_Triangle, samplesName, "Triangle", csvFile);
    //Integrate(SampleImage_Step, points, c_referenceValue_Step, samplesName, "Step", csvFile);
    //Integrate(SampleImage_Gaussian, points, c_referenceValue_Gaussian, samplesName, "Gaussian", csvFile);
    //Integrate(SampleImage_Bilinear, points, c_referenceValue_Bilinear, samplesName, "Bilinear", csvFile);
}

int main(int argc, char **argv)
{
    FILE* csvFile = nullptr;
    fopen_s(&csvFile, "out/error.csv", "w+t");

    MakeSampleImage(SampleImage_Disk, "out/disk.png");
    MakeSampleImage(SampleImage_Triangle, "out/triangle.png");
    MakeSampleImage(SampleImage_Step, "out/step.png");
    MakeSampleImage(SampleImage_Gaussian, "out/gaussian.png");
    MakeSampleImage(SampleImage_Bilinear, "out/bilinear.png");

    DoTest2D(GeneratePoints_WhiteNoise, "WhiteNoise", csvFile);
    DoTest2D(GeneratePoints_GoldenRatio, "GoldenRatio", csvFile);

    fclose(csvFile);

    system("pause");

    return 0;
}

/*

----- Good Candidate algorithm -----

co-author with brandon!

Motivation:
  Dart throwing and best candidate make progressive sequences, but best candidate gives better results.
  Actually, dart throwing isn't progressive out of the box really! You have a radius!
  Anyways, not as high quality as voronoi relaxation method, but it's progressive which is neat. Maybe eg 1024 points as constants in a shader, use however many you want, subset of that.

Do the tests from the multijittered sampling: disk, step, gaussian, bilinear.

compare to... ??
* whtie noise
* blue noise
* owen scrambled sobol
? projective blue noise sampling via other methods?
? low discrepancy blue noise?

other links:
renderman: https://graphics.pixar.com/library/RendermanTog2018/paper.pdf
projective blue noise article: http://resources.mpi-inf.mpg.de/ProjectiveBlueNoise/
multijittered sampling: https://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/
generalized golden ratio: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/


*/