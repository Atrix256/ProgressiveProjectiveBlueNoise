#include <stdio.h>
#include <vector>
#include <array>
#include <random>
#include <string>
#include <stdint.h>
#include <stdarg.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

#include "image.h"

#define TEST_IMAGE_SIZE() 128 // in pixels, on each axis
#define SAMPLE_IMAGE_SIZE() 1024
#define NUM_SAMPLES() 100

static const float c_referenceValue_Disk = 0.5f;
static const float c_referenceValue_Triangle = 0.5f;
static const float c_referenceValue_Step = 1.0f / c_pi;
static const float c_referenceValue_Gaussian = c_pi / 4.0f * (float)erf(1.0) * (float)erf(1.0);
static const float c_referenceValue_Bilinear = 0.25f;

// generalized golden ratio, for making 2d low discrepancy sequences
// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
static const float c_goldenRatio2 = 1.32471795724474602596f;

typedef std::array<float, 2> Vec2;

typedef std::array<std::vector<std::string>, 5> Log;

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
    return ((samplePos[0] * samplePos[0] + samplePos[1] * samplePos[1]) < 2.0f / c_pi) ? 1.0f : 0.0f;
}

// reference value is 0.5
float SampleImage_Triangle(const Vec2& samplePos)
{
    return (samplePos[1] > samplePos[0]) ? 1.0f : 0.0f;
}

// reference value is 1/pi
float SampleImage_Step(const Vec2& samplePos)
{
    return (samplePos[0] < 1.0f / c_pi) ? 1.0f : 0.0f;
}

// reference value is pi/4 * erf^2(1)
float SampleImage_Gaussian(const Vec2& samplePos)
{
    return expf(-(samplePos[0] *samplePos[0])-(samplePos[1] *samplePos[1]));
}

// reference value is 0.25
float SampleImage_Bilinear(const Vec2& samplePos)
{
    return samplePos[0] * samplePos[1];
}

template <typename T, typename LAMBDA1, typename LAMBDA2>
void MitchelsBestCandidateAlgorithm (std::vector<T>& results, size_t desiredItemCount, int candidateMultiplier, const LAMBDA2& GenerateRandomCandidate, const LAMBDA1& DifferenceScoreCalculator)
{
    results.resize(desiredItemCount);

    // for each item we need to fill in
    for (int itemIndex = 0; itemIndex < desiredItemCount; ++itemIndex)
    {
        // calculate how many candidates we want to generate for this item
        int candidateCount = itemIndex * candidateMultiplier + 1;

        T bestCandidate;
        float bestCandidateMinimumDifferenceScore;

        // for each candidate
        for (int candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
        {
            // make a randomized candidate
            T candidate = GenerateRandomCandidate();
            float minimumDifferenceScore = FLT_MAX;

            // the score of this candidate is the minimum difference from all existing items
            for (int checkItemIndex = 0; checkItemIndex < itemIndex; ++checkItemIndex)
            {
                float differenceScore = DifferenceScoreCalculator(candidate, results[checkItemIndex]);
                minimumDifferenceScore = std::min(minimumDifferenceScore, differenceScore);
            }

            // the candidate with the largest minimum distance is the one we want to keep
            if (candidateIndex == 0 || minimumDifferenceScore > bestCandidateMinimumDifferenceScore)
            {
                bestCandidate = candidate;
                bestCandidateMinimumDifferenceScore = minimumDifferenceScore;
            }
        }

        // keep the winning candidate
        results[itemIndex] = bestCandidate;
    }
}

template <typename T, typename LAMBDA1, typename LAMBDA2>
void GoodCandidateAlgorithm(std::vector<T>& results, size_t desiredItemCount, int candidateMultiplier, const LAMBDA2& GenerateRandomCandidate, const LAMBDA1& DifferenceScoreCalculator)
{
    static const size_t DIMENSION = 2;

    // map candidate index to score
    struct CandidateScore
    {
        size_t index;
        float score;
    };
    typedef std::vector<CandidateScore> CandidateScores;
    static const size_t c_numScores = (1 << DIMENSION) - 1;  // 2^(dimension)-1

    // make space for the results
    results.resize(desiredItemCount);

    // for each item we need to fill in
    for (int itemIndex = 0; itemIndex < desiredItemCount; ++itemIndex)
    {
        // calculate how many candidates we want to generate for this item
        int candidateCount = itemIndex * candidateMultiplier + 1;

        // generate the candidates
        std::vector<T> candidates;
        candidates.resize(candidateCount);
        for (T& candidate : candidates)
            candidate = GenerateRandomCandidate();

        // initialize the overall scores
        CandidateScores overallScores;
        overallScores.resize(candidateCount);
        for (int candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
        {
            overallScores[candidateIndex].index = candidateIndex;
            overallScores[candidateIndex].score = 0.0f;
        }

        // allocate space for the individual scores
        CandidateScores scores;
        scores.resize(candidateCount);

        // score the candidates by each measure of scoring
        for (size_t scoreIndex = 0; scoreIndex < c_numScores; ++scoreIndex)
        {
            // for each candidate in this score index...
            for (size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
            {
                const T& candidate = candidates[candidateIndex];

                // calculate the score of the candidate.
                // the score is the minimum distance to any other points
                float minimumDifferenceScore = FLT_MAX;
                for (int checkItemIndex = 0; checkItemIndex < itemIndex; ++checkItemIndex)
                {
                    float differenceScore = DifferenceScoreCalculator(candidate, results[checkItemIndex], scoreIndex);
                    minimumDifferenceScore = std::min(minimumDifferenceScore, differenceScore);
                }

                scores[candidateIndex].index = candidateIndex;
                scores[candidateIndex].score = minimumDifferenceScore;
            }

            // sort the scores from high to low
            std::sort(
                scores.begin(),
                scores.end(),
                [] (const CandidateScore& A, const CandidateScore& B)
                {
                    return A.score > B.score;
                }
            );

            // add the rank of this score a score for each candidate
            for (size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
                overallScores[scores[candidateIndex].index].score += float(candidateIndex);
        }

        // sort the overall scores from low to high
        std::sort(
            overallScores.begin(),
            overallScores.end(),
            [] (const CandidateScore& A, const CandidateScore& B)
            {
                return A.score < B.score;
            }
        );

        // keep the point that had the lowest summed rank
        results[itemIndex] = candidates[overallScores[0].index];
    }
}

void GeneratePoints_WhiteNoise(std::vector<Vec2>& points, size_t numPoints)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    points.resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        points[i][0] = dist(RNG());
        points[i][1] = dist(RNG());
    }
}

void GeneratePoints_GoldenRatio(std::vector<Vec2>& points, size_t numPoints)
{
    static const float a1 = 1.0f / c_goldenRatio2;
    static const float a2 = 1.0f / (c_goldenRatio2 * c_goldenRatio2);

    points.resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        points[i][0] = fmodf(0.5f + a1 * float(i), 1.0f);
        points[i][1] = fmodf(0.5f + a2 * float(i), 1.0f);
    }
}

void GeneratePoints_BlueNoise(std::vector<Vec2>& points, size_t numPoints)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    MitchelsBestCandidateAlgorithm(
        points,
        numPoints,
        5,
        [&]()
        {
            Vec2 ret;
            ret[0] = dist(RNG());
            ret[1] = dist(RNG());
            return ret;
        },
        [](const Vec2& A, const Vec2& B)
        {
            float distSq = 0.0f;
            for (int i = 0; i < 2; ++i)
            {
                float diff = fabsf(B[i] - A[i]);
                if (diff > 0.5f)
                    diff = 1.0f - diff;
                distSq += diff * diff;
            }
            return distSq;
        }
    );
}

void GeneratePoints_ProjectiveBlueNoise(std::vector<Vec2>& points, size_t numPoints)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    static const size_t c_numScores = (1 << 2) - 1;
    Vec2 axisMasks[c_numScores];
    for (size_t i = 1; i <= c_numScores; ++i)
    {
        Vec2& axisMask = axisMasks[i - 1];

        for (size_t j = 0; j < 2; ++j)
        {
            if (i & (size_t(1) << j))
                axisMask[j] = 1.0f;
            else
                axisMask[j] = 0.0f;
        }
    }

    GoodCandidateAlgorithm(
        points,
        numPoints,
        20,
        [&]()
        {
            Vec2 ret;
            ret[0] = dist(RNG());
            ret[1] = dist(RNG());
            return ret;
        },
        [&](const Vec2& A, const Vec2& B, size_t scoreIndex)
        {
            const Vec2& axisMask = axisMasks[scoreIndex];

            float distSq = 0.0f;
            for (int i = 0; i < 2; ++i)
            {
                float diff = fabsf(B[i] - A[i]) * axisMask[i];
                if (diff > 0.5f)
                    diff = 1.0f - diff;
                distSq += diff * diff;
            }
            return sqrtf(distSq);
        }
    );
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
        samplePos[1] = halfPixel + float(iy) / float(TEST_IMAGE_SIZE());

        for (size_t ix = 0; ix < TEST_IMAGE_SIZE(); ++ix)
        {
            samplePos[0] = halfPixel + float(ix) / float(TEST_IMAGE_SIZE());

            *pixel = uint8(Clamp(0.5f + SampleImage(samplePos) * 255.0f, 0.0f, 255.0f));

            ++pixel;
        }
    }

    stbi_write_png(fileName, TEST_IMAGE_SIZE(), TEST_IMAGE_SIZE(), 1, image.data(), 0);
}

void LogLineAppend(std::vector<std::string>& logLines, int lineNumber, const char* fmt, ...)
{
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsprintf_s(buffer, fmt, args);
    va_end(args);

    if (lineNumber + 1 > logLines.size())
        logLines.resize(lineNumber + 1);

    logLines[lineNumber] += buffer;
}

template <typename T>
void Integrate(const T& SampleImage, const std::vector<Vec2>& points, float reference, std::vector<std::string>& logLines)
{
    float result = 0.0f;
    for (int i = 0; i < points.size(); ++i)
    {
        float sample = SampleImage(points[i]);
        result = Lerp(result, sample, 1.0f / float(i + 1));

        // TODO: more data, and log error / index?
        LogLineAppend(logLines, i + 1, ",\"%f\"", fabsf(result - reference));
        //fprintf(csvFile, "\"%i\",\"%f\",\"%f\",\"%f\"\n", i, log10f(float(i+1)), result, fabsf(result-reference));
    }
}

void DoTest2D (const GeneratePoints& generatePoints, Log& log, const char* label)
{
    // generate the sample points
    std::vector<Vec2> points;
    generatePoints(points, NUM_SAMPLES());

    // TODO: move to a function
    // TODO: rename things to make sense, like "imageBottom" == ??
    // TODO: DFT point samples
    // Save the sample points as an image
    {
        // draw the footer - show the angle and offset distribution on a number line
        Image imageBottom(SAMPLE_IMAGE_SIZE(), SAMPLE_IMAGE_SIZE());
        Image imageSamples(50, 50);  // TODO: make this be a #define at the top for how large it is
        ImageComplex imageSamplesDFTComplex(imageSamples.m_width, imageSamples.m_height);

        int graphSize = int(float(SAMPLE_IMAGE_SIZE()) * 0.7f);
        int padding = (SAMPLE_IMAGE_SIZE() - graphSize) / 2;

        DrawLine(imageBottom, padding, padding, padding + graphSize, padding, 128, 128, 255);
        DrawLine(imageBottom, padding, padding + graphSize, padding + graphSize, padding + graphSize, 128, 128, 255);

        DrawLine(imageBottom, padding, padding, padding, padding + graphSize, 128, 128, 255);
        DrawLine(imageBottom, padding + graphSize, padding, padding + graphSize, padding + graphSize, 128, 128, 255);

        for (int sampleIndex = 0; sampleIndex < NUM_SAMPLES(); ++sampleIndex)
        {
            const Vec2& v = points[sampleIndex];
            float percent = float(sampleIndex) / float(NUM_SAMPLES() - 1);
            uint8 color = uint8(255.0f * percent);

            int dotX = padding + int(v[0] * float(graphSize));

            int dotY = padding + int(v[1] * float(graphSize));

            // draw the 2d dots
            DrawCircle(imageBottom, dotX, dotY, 2, 0, color, 0);

            // draw the 1d angle dots to the right
            DrawCircle(imageBottom, padding + graphSize + padding / 2, dotY, 2, 0, color, 0);

            // draw the 1d offset dots below
            DrawCircle(imageBottom, dotX, padding + graphSize + padding / 2, 2, 0, color, 0);

            // make the sample image
            {
                int x = int(v[0] * float(imageSamples.m_width - 1) + 0.5f);
                int y = int(v[1] * float(imageSamples.m_height - 1) + 0.5f);
                uint8* imageSamplePixel = &imageSamples.m_pixels[(y*imageSamples.m_width + x) * 4];
                imageSamplePixel[0] = 0;
                imageSamplePixel[1] = 0;
                imageSamplePixel[2] = 0;
                imageSamplePixel[3] = 255;
            }
        }

        // save the images
        char fileName[256];
        sprintf_s(fileName, "out/samples_%s.png", label);
        SaveImage(fileName, imageBottom);
    }

    // test the sample points for integration
    Integrate(SampleImage_Disk, points, c_referenceValue_Disk, log[0]);
    Integrate(SampleImage_Triangle, points, c_referenceValue_Triangle, log[1]);
    Integrate(SampleImage_Step, points, c_referenceValue_Step, log[2]);
    Integrate(SampleImage_Gaussian, points, c_referenceValue_Gaussian, log[3]);
    Integrate(SampleImage_Bilinear, points, c_referenceValue_Bilinear, log[4]);
}

void WriteLog(std::vector<std::string>& log, const char* fileName)
{
    FILE* csvFile = nullptr;
    fopen_s(&csvFile, fileName, "w+t");
    for (const std::string& s : log)
        fprintf(csvFile, "%s\n", s.c_str());
    fprintf(csvFile, "\n");
    fclose(csvFile);
}

int main(int argc, char **argv)
{
    // set up the logs
    Log log;
    for (auto& l : log)
    {
        LogLineAppend(l, 0, "\"Sample\",\"White Noise\",\"Golden Ratio\",\"Blue Noise\",\"Projective Blue Noise\"");
        for (int i = NUM_SAMPLES(); i > 0; --i)
            LogLineAppend(l, i, "\"%i\"", i);
    }

    // make images of the functions we are integrating
    MakeSampleImage(SampleImage_Disk, "out/int_disk.png");
    MakeSampleImage(SampleImage_Triangle, "out/int_triangle.png");
    MakeSampleImage(SampleImage_Step, "out/int_step.png");
    MakeSampleImage(SampleImage_Gaussian, "out/int_gaussian.png");
    MakeSampleImage(SampleImage_Bilinear, "out/int_bilinear.png");

    // do the tests for each type of sampling
    printf("White Noise...\n");
    DoTest2D(GeneratePoints_WhiteNoise, log, "white");
    printf("Golden Ratio...\n");
    DoTest2D(GeneratePoints_GoldenRatio, log, "golden");
    printf("Blue Noise...\n");
    DoTest2D(GeneratePoints_BlueNoise, log, "blue");
    printf("Projective Blue Noise...\n");
    DoTest2D(GeneratePoints_ProjectiveBlueNoise, log, "projblue");

    // write out the logs
    WriteLog(log[0], "out/data_disk.csv");
    WriteLog(log[1], "out/data_triangle.csv");
    WriteLog(log[2], "out/data_step.csv");
    WriteLog(log[3], "out/data_gaussian.csv");
    WriteLog(log[4], "out/data_bilinear.csv");

    return 0;
}

/*

TODO:

* expand the sampling box out a little bit so points don't overlap it!
* dft sampling patterns - how?


* the multijitter paper did 10,000 samples. that is going to be super slow for blue noise and projective blue noise.
 * could maybe do a grid or go multithreaded or something?



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