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
#define GRAPH_IMAGE_SIZE() 1024
#define NUM_SAMPLES() 500
#define DO_SLOW_SAMPLES() true

#define BLUENOISE_CANDIDATE_MULTIPLIER() 5

#define PROJBLUENOISE_CANDIDATE_MULTIPLIER() 100
#define PROJBLUENOISE_PARTITIONS() 10

#define DO_DFT() true
#define DFT_IMAGE_SIZE() 256

static const float c_referenceValue_Disk = 0.5f;
static const float c_referenceValue_Triangle = 0.5f;
static const float c_referenceValue_Step = 1.0f / c_pi;
static const float c_referenceValue_Gaussian = c_pi / 4.0f * (float)erf(1.0) * (float)erf(1.0);
static const float c_referenceValue_Bilinear = 0.25f;

// generalized golden ratio, for making 2d low discrepancy sequences
// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
static const float c_goldenRatio2 = 1.32471795724474602596f;

typedef std::array<float, 2> Vec2;

typedef void(*GeneratePointsFN)(std::vector<Vec2>& points, size_t numPoints);

void GeneratePoints_WhiteNoise(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_GoldenRatio(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_GoldenRatio2(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_BlueNoise(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_ProjectiveBlueNoise(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_ProjectiveBlueNoise2(std::vector<Vec2>& points, size_t numPoints);

struct SamplingPattern
{
    const char* nameHuman;
    const char* nameFile;
    GeneratePointsFN generatePoints;
    bool enable;
};

static const SamplingPattern g_samplingPatterns[] =
{
    {"White Noise", "white", GeneratePoints_WhiteNoise, true},
    {"Golden Ratio", "golden", GeneratePoints_GoldenRatio, true},
    {"Golden Ratio2", "golden2", GeneratePoints_GoldenRatio2, true},
    {"Blue Noise", "blue", GeneratePoints_BlueNoise, DO_SLOW_SAMPLES()},
    {"Projective Blue Noise", "projblue", GeneratePoints_ProjectiveBlueNoise, DO_SLOW_SAMPLES()},
    {"Projective Blue Noise 2", "projblue2", GeneratePoints_ProjectiveBlueNoise2, DO_SLOW_SAMPLES()},
};
static const size_t c_numSamplingPatterns = sizeof(g_samplingPatterns) / sizeof(g_samplingPatterns[0]);

struct Log
{
    std::array<std::vector<std::string>, 5> logs;
    std::array<std::array<std::vector<float>, 5>, c_numSamplingPatterns+3> errors;  // indexed by: [sampleType][test]
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

template <size_t DIMENSION>
void MitchelsBestCandidateAlgorithm (std::vector< std::array<float, DIMENSION>>& results, size_t desiredItemCount, int candidateMultiplier)
{
    typedef std::array<float, DIMENSION> T;

    static std::random_device rd;
    static std::mt19937 rng(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

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
            T candidate;
            for (size_t i = 0; i < DIMENSION; ++i)
                candidate[i] = dist(rng);

            float minimumDifferenceScore = FLT_MAX;

            // the score of this candidate is the minimum difference from all existing items
            for (int checkItemIndex = 0; checkItemIndex < itemIndex; ++checkItemIndex)
            {
                float distSq = 0.0f;
                for (int i = 0; i < DIMENSION; ++i)
                {
                    float diff = fabsf(results[checkItemIndex][i] - candidate[i]);
                    if (diff > 0.5f)
                        diff = 1.0f - diff;
                    distSq += diff * diff;
                }
                minimumDifferenceScore = std::min(minimumDifferenceScore, distSq);
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

template <size_t DIMENSION, size_t PARTITIONS>
struct GoodCandidateSubspace
{
    typedef std::array<float, DIMENSION> T;

    void Init(size_t scoreIndex)
    {
        size_t partitionCount = 1;
        int multiplier = 1;
        for (size_t dimensionIndex = 0; dimensionIndex < DIMENSION; ++dimensionIndex)
        {
            axisMask[dimensionIndex] = (scoreIndex & (size_t(1) << dimensionIndex)) ? false : true;

            if (!axisMask[dimensionIndex])
            {
                axisPartitionOffset[dimensionIndex] = 0;
                continue;
            }

            axisPartitionOffset[dimensionIndex] = multiplier;
            partitionCount *= PARTITIONS;
            multiplier *= PARTITIONS;
        }

        partitionedPoints.resize(partitionCount);
        partitionChecked.resize(partitionCount);
    }

    int PartitionForPoint(const T& point) const
    {
        int subspacePartition = 0;
        int multiplier = 1;
        for (size_t dimensionIndex = 0; dimensionIndex < DIMENSION; ++dimensionIndex)
        {
            if (!axisMask[dimensionIndex])
                continue;

            int axisPartition = int(point[dimensionIndex] * float(PARTITIONS));
            axisPartition = std::min(axisPartition, int(PARTITIONS-1));
            subspacePartition += axisPartition * multiplier;
            multiplier *= PARTITIONS;
        }

        // TODO: use axisPartitionOffset[] instead of duplicating multiplier logic!

        return subspacePartition;
    }

    void PartitionCoordinatesForPoint(const T& point, std::array<int, DIMENSION>& partitionCoordinates) const
    {
        int partition = PartitionForPoint(point);

        for (size_t dimensionIndexPlusOne = DIMENSION; dimensionIndexPlusOne > 0; --dimensionIndexPlusOne)
        {
            size_t dimensionIndex = dimensionIndexPlusOne - 1;

            if (!axisMask[dimensionIndex])
            {
                partitionCoordinates[dimensionIndex] = 0;
                continue;
            }

            partitionCoordinates[dimensionIndex] = partition / axisPartitionOffset[dimensionIndex];
            partition = partition % axisPartitionOffset[dimensionIndex];
        }
    }

    float SquaredDistanceToClosestPointRecursive(const T& point, std::array<int, DIMENSION> partitionCoordinates, int radius, size_t dimensionIndex)
    {
        // if we have run out of dimensions, it's time to search a partition
        if (dimensionIndex == DIMENSION)
        {
            int subspacePartition = 0;
            for (int i = 0; i < DIMENSION; ++i)
            {
                if (!axisMask[i])
                    continue;
                subspacePartition += partitionCoordinates[i] * axisPartitionOffset[i];
            }

            // if we've already checked this partition, nothing to do.
            // otherwise, mark is as checked so it isn't checked again.
            if (partitionChecked[subspacePartition])
                return FLT_MAX;
            partitionChecked[subspacePartition] = true;

            float minDistSq = FLT_MAX;
            for (auto& p : partitionedPoints[subspacePartition])
            {
                float distSq = 0.0f;
                for (size_t i = 0; i < DIMENSION; ++i)
                {
                    if (!axisMask[i])
                        continue;

                    float diff = fabsf(p[i] - point[i]);
                    if (diff > 0.5f)
                        diff = 1.0f - diff;
                    distSq += diff * diff;
                }
                minDistSq = std::min(minDistSq, distSq);
            }

            return minDistSq;
        }

        // if radius 0, or this axis doesn't participate, do a pass through!
        if (radius == 0 || !axisMask[dimensionIndex])
            return SquaredDistanceToClosestPointRecursive(point, partitionCoordinates, radius, dimensionIndex + 1);

        // loop through this axis radius and return the smallest value we've found
        float ret = FLT_MAX;
        std::array<int, DIMENSION> searchPartitionCoordinates = partitionCoordinates;
        for (int axisOffset = -radius; axisOffset <= radius; ++axisOffset)
        {
            searchPartitionCoordinates[dimensionIndex] = (partitionCoordinates[dimensionIndex] + axisOffset + PARTITIONS) % PARTITIONS;
            ret = std::min(ret, SquaredDistanceToClosestPointRecursive(point, searchPartitionCoordinates, radius, dimensionIndex + 1));
        }

        return ret;
    }

    float SquaredDistanceToClosestPoint(const T& point)
    {
        // mark all partitions as having not been checked yet
        std::fill(partitionChecked.begin(), partitionChecked.end(), false);

        // get the partition coordinate this point is in
        std::array<int, DIMENSION> partitionCoordinates;
        PartitionCoordinatesForPoint(point, partitionCoordinates);

        // Loop through increasingly larger rectangular rings until we find a ring that has at least one point.
        // return the distance to the closest point in that ring, and the next ring out.
        // We need to do an extra ring to get the correct answer.
        int maxRadius = int(PARTITIONS / 2);
        bool foundInnerRing = false;
        float minDist = FLT_MAX;
        for (int radius = 0; radius <= maxRadius; ++radius)
        {
            float distance = SquaredDistanceToClosestPointRecursive(point, partitionCoordinates, radius, 0);
            minDist = std::min(minDist, distance);
            if (minDist < FLT_MAX)
            {
                if (foundInnerRing)
                    return minDist;
                else
                    foundInnerRing = true;
            }
        }
        return minDist;
    }
    
    void Insert(const T& point)
    {
        int subspacePartition = PartitionForPoint(point);
        partitionedPoints[subspacePartition].push_back(point);
    }

    std::vector<std::vector<T>> partitionedPoints;
    std::vector<bool> partitionChecked;
    std::array<bool, DIMENSION> axisMask;
    std::array<int, DIMENSION> axisPartitionOffset;
};

template <size_t DIMENSION>
void GoodCandidateAlgorithm(std::vector< std::array<float, DIMENSION>>& results, size_t desiredItemCount, int candidateMultiplier, bool reportProgress)
{
    typedef std::array<float, DIMENSION> T;

    static std::random_device rd;
    static std::mt19937 rng(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    GoodCandidateSubspace accelerationStructure;
    accelerationStructure.Init(0);

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

    int lastPercent = -1;

    // for each item we need to fill in
    for (int itemIndex = 0; itemIndex < desiredItemCount; ++itemIndex)
    {
        // calculate how many candidates we want to generate for this item
        int candidateCount = itemIndex * candidateMultiplier + 1;

        // generate the candidates
        std::vector<T> candidates;
        candidates.resize(candidateCount);
        for (T& candidate : candidates)
        {
            for (size_t i = 0; i < DIMENSION; ++i)
                candidate[i] = dist(rng);
        }

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
            // make the axis mask for this score index. we are scoring within a specific subspace.
            std::array<bool, DIMENSION> axisMask;
            for (size_t i = 0; i < DIMENSION; ++i)
                axisMask[i] = (scoreIndex & (size_t(1) << i)) ? false : true;

            // for each candidate in this score index...
            for (size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
            {
                const T& candidate = candidates[candidateIndex];

                // calculate the score of the candidate.
                // the score is the minimum distance to any other points
                float minimumDifferenceScore = FLT_MAX;
                for (int checkItemIndex = 0; checkItemIndex < itemIndex; ++checkItemIndex)
                {
                    float distSq = 0.0f;
                    for (int i = 0; i < DIMENSION; ++i)
                    {
                        if (!axisMask[i])
                            continue;

                        float diff = fabsf(results[checkItemIndex][i] - candidate[i]);
                        if (diff > 0.5f)
                            diff = 1.0f - diff;
                        distSq += diff * diff;
                    }
                    minimumDifferenceScore = std::min(minimumDifferenceScore, distSq);
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

        if (reportProgress)
        {
            int percent = int(100.0f * float(itemIndex) / float(desiredItemCount));
            if (lastPercent != percent)
            {
                lastPercent = percent;
                printf("\rMaking Points: %i%%", lastPercent);
            }
        }
    }

    if (reportProgress)
    {
        printf("\rMaking Points: 100%%\n");
    }
}

template <size_t DIMENSION, size_t PARTITIONS, bool EXTRAPENALTY>
void GoodCandidateAlgorithmAccell(std::vector< std::array<float, DIMENSION>>& results, size_t desiredItemCount, int candidateMultiplier, bool reportProgress)
{
    typedef std::array<float, DIMENSION> T;

    static std::random_device rd;
    static std::mt19937 rng(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // map candidate index to score
    struct CandidateScore
    {
        size_t index;
        float score;
    };
    typedef std::vector<CandidateScore> CandidateScores;
    static const size_t c_numScores = (1 << DIMENSION) - 1;  // 2^(dimension)-1

    // setup the acceleration structures
    std::array<GoodCandidateSubspace<DIMENSION, PARTITIONS>, c_numScores> subspaces;
    for (size_t scoreIndex = 0; scoreIndex < c_numScores; ++scoreIndex)
        subspaces[scoreIndex].Init(scoreIndex);

    // make space for the results
    results.resize(desiredItemCount);

    int lastPercent = -1;

    // for each item we need to fill in
    for (int itemIndex = 0; itemIndex < desiredItemCount; ++itemIndex)
    {
        // calculate how many candidates we want to generate for this item
        int candidateCount = itemIndex * candidateMultiplier + 1;

        // generate the candidates
        std::vector<T> candidates;
        candidates.resize(candidateCount);
        for (T& candidate : candidates)
        {
            for (size_t i = 0; i < DIMENSION; ++i)
                candidate[i] = dist(rng);
        }

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
            // get the subspace we are working with
            auto& subspace = subspaces[scoreIndex];

            // for each candidate in this score index...
            for (size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
            {
                // calculate the score of the candidate.
                // the score is the minimum distance to any other points
                scores[candidateIndex].index = candidateIndex;
                scores[candidateIndex].score = subspace.SquaredDistanceToClosestPoint(candidates[candidateIndex]);
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
                overallScores[scores[candidateIndex].index].score += EXTRAPENALTY ? float(candidateIndex) * float(candidateIndex) : float(candidateIndex);
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

        // insert this point into the acceleration structures
        for (size_t scoreIndex = 0; scoreIndex < c_numScores; ++scoreIndex)
            subspaces[scoreIndex].Insert(results[itemIndex]);

        // report our percentage done if we should
        if (reportProgress)
        {
            int percent = int(100.0f * float(itemIndex) / float(desiredItemCount));
            if (lastPercent != percent)
            {
                lastPercent = percent;
                printf("\rMaking Points: %i%%", lastPercent);
            }
        }
    }

    if (reportProgress)
        printf("\rMaking Points: 100%%\n");
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

void GeneratePoints_GoldenRatio2(std::vector<Vec2>& points, size_t numPoints)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    static const float a1 = 1.0f / c_goldenRatio2;
    static const float a2 = 1.0f / (c_goldenRatio2 * c_goldenRatio2);

    static const float c_magicNumber = 0.732f;

    points.resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        points[i][0] = fmodf(dist(RNG()) * c_magicNumber + a1 * float(i) / sqrtf(float(i + 1)), 1.0f);
        points[i][1] = fmodf(dist(RNG()) * c_magicNumber + a2 * float(i) / sqrtf(float(i + 1)), 1.0f);
    }
}

void GeneratePoints_BlueNoise(std::vector<Vec2>& points, size_t numPoints)
{
    MitchelsBestCandidateAlgorithm(points, numPoints, BLUENOISE_CANDIDATE_MULTIPLIER());
}

void GeneratePoints_ProjectiveBlueNoise(std::vector<Vec2>& points, size_t numPoints)
{
    GoodCandidateAlgorithmAccell<2, PROJBLUENOISE_PARTITIONS(), false>(points, numPoints, PROJBLUENOISE_CANDIDATE_MULTIPLIER(), true);
}

void GeneratePoints_ProjectiveBlueNoise2(std::vector<Vec2>& points, size_t numPoints)
{
    GoodCandidateAlgorithmAccell<2, PROJBLUENOISE_PARTITIONS(), true>(points, numPoints, PROJBLUENOISE_CANDIDATE_MULTIPLIER(), true);
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
void Integrate(const T& SampleImage, const std::vector<Vec2>& points, float reference, std::vector<std::string>& logLines, std::vector<float>& errors)
{
    float result = 0.0f;
    for (int i = 0; i < points.size(); ++i)
    {
        float sample = SampleImage(points[i]);
        result = Lerp(result, sample, 1.0f / float(i + 1));
        LogLineAppend(logLines, i + 1, ",\"%f\"", fabsf(result - reference));
        errors.push_back(fabsf(result - reference));
    }
}

void MakeSamplesImage(std::vector<Vec2>& points, const char* label)
{
    // draw the footer - show the angle and offset distribution on a number line
    Image image(SAMPLE_IMAGE_SIZE(), SAMPLE_IMAGE_SIZE());
    Image imageSamples(DFT_IMAGE_SIZE(), DFT_IMAGE_SIZE());

    int graphSize = int(float(SAMPLE_IMAGE_SIZE()) * 0.7f);
    int padding = (SAMPLE_IMAGE_SIZE() - graphSize) / 2;

    DrawLine(image, padding, padding, padding + graphSize, padding, 128, 128, 255);
    DrawLine(image, padding, padding + graphSize, padding + graphSize, padding + graphSize, 128, 128, 255);

    DrawLine(image, padding, padding, padding, padding + graphSize, 128, 128, 255);
    DrawLine(image, padding + graphSize, padding, padding + graphSize, padding + graphSize, 128, 128, 255);

    for (int sampleIndex = 0; sampleIndex < NUM_SAMPLES(); ++sampleIndex)
    {
        const Vec2& v = points[sampleIndex];
        float percent = float(sampleIndex) / float(NUM_SAMPLES() - 1);
        uint8 color = uint8(255.0f * percent);

        int dotX = padding + int(v[0] * float(graphSize));

        int dotY = padding + int(v[1] * float(graphSize));

        // draw the 2d dots
        DrawCircle(image, dotX, dotY, 2, 0, color, 0);

        // draw the 1d angle dots to the right
        DrawCircle(image, padding + graphSize + padding / 2, dotY, 2, 0, color, 0);

        // draw the 1d offset dots below
        DrawCircle(image, dotX, padding + graphSize + padding / 2, 2, 0, color, 0);

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
    SaveImage(fileName, image);

    // also write the samples out as a csv
    {
        FILE* file = nullptr;
        sprintf_s(fileName, "out/samplescsv_%s.csv", label);
        fopen_s(&file, fileName, "w+t");
        for (const Vec2& v : points)
            fprintf(file, "\"%f\",\"%f\"\n", v[0], v[1]);
        fclose(file);
    }

    // DFT the samples image if we should
    if (DO_DFT())
    {
        ImageComplex imageSamplesDFTComplex(imageSamples.m_width, imageSamples.m_height);
        DFTImage(imageSamples, imageSamplesDFTComplex, true);
        Image imageSamplesDFT(imageSamples.m_width, imageSamples.m_height);
        GetMagnitudeData(imageSamplesDFTComplex, imageSamplesDFT);
        sprintf_s(fileName, "out/samplesdft_%s.png", label);
        SaveImage(fileName, imageSamplesDFT);
        sprintf_s(fileName, "out/samplesdftraw_%s.png", label);
        SaveImage(fileName, imageSamples);
    }
}

void DoTest2D (const GeneratePoints& generatePoints, Log& log, const char* label, int noiseType)
{
    // generate the sample points and save them as an image
    std::vector<Vec2> points;
    generatePoints(points, NUM_SAMPLES());
    MakeSamplesImage(points, label);

    // test the sample points for integration
    Integrate(SampleImage_Disk, points, c_referenceValue_Disk, log.logs[0], log.errors[noiseType][0]);
    Integrate(SampleImage_Triangle, points, c_referenceValue_Triangle, log.logs[1], log.errors[noiseType][1]);
    Integrate(SampleImage_Step, points, c_referenceValue_Step, log.logs[2], log.errors[noiseType][2]);
    Integrate(SampleImage_Gaussian, points, c_referenceValue_Gaussian, log.logs[3], log.errors[noiseType][3]);
    Integrate(SampleImage_Bilinear, points, c_referenceValue_Bilinear, log.logs[4], log.errors[noiseType][4]);
}

void MakeErrorGraph(const Log& log, int test, const char* fileName)
{
    Image image(GRAPH_IMAGE_SIZE(), GRAPH_IMAGE_SIZE());
    ClearImage(image, 224, 224, 224);

    // get the x axis min and max
    float xAxisMin = log10f(1.0f);
    float xAxisMax = ceilf(log10f(float(NUM_SAMPLES())));

    // get the y axis min and max
    float yAxisMin = log.errors[0][test][0];
    float yAxisMax = yAxisMin;
    for (auto& a : log.errors)
    {
        for (auto& b : a[test])
        {
            yAxisMin = std::min(yAxisMin, b);
            yAxisMax = std::max(yAxisMax, b);
        }
    }
    yAxisMin = std::max(yAxisMin, 0.00001f);
    yAxisMin = log10f(yAxisMin);
    yAxisMax = log10f(yAxisMax);

    // TODO: use golden ratio to make colors

    // draw the graph
    uint8 colors[9][3] =
    {
        {255, 0, 0},
        {0, 255, 0},
        {0, 0, 255},
        {0, 255, 255},
        {255, 0, 255},
        {255, 255, 0},
        {128, 0, 0},
        {0, 128, 0},
        {0, 0, 128},
    };

    int colorIndex = 0;
    for (int sampleType = 0; sampleType < log.errors.size(); ++sampleType)
    {
        if (sampleType < 3)
            continue;

        bool firstPoint = true;
        Vec2 lastUV;
        for (int sampleIndex = 0; sampleIndex < NUM_SAMPLES(); ++sampleIndex)
        {
            float logSample = log10f(float(sampleIndex+1));
            float logError = log.errors[sampleType][test][sampleIndex] > 0.0f ? log10f(log.errors[sampleType][test][sampleIndex]) : yAxisMin;

            Vec2 uv;
            uv[0] = (logSample - xAxisMin) / (xAxisMax - xAxisMin);
            uv[1] = (logError - yAxisMin) / (yAxisMax - yAxisMin);

            // we want y=0 to be at the bottom of the image
            uv[1] = 1.0f - uv[1];

            if (firstPoint)
            {
                firstPoint = false;
            }
            else
            {
                // TODO: make the drawing functions work in uv's?
                int x1 = int(0.5f + lastUV[0] * float(GRAPH_IMAGE_SIZE()));
                int y1 = int(0.5f + lastUV[1] * float(GRAPH_IMAGE_SIZE()));
                int x2 = int(0.5f + uv[0] * float(GRAPH_IMAGE_SIZE()));
                int y2 = int(0.5f + uv[1] * float(GRAPH_IMAGE_SIZE()));

                DrawLine(image, x1, y1, x2, y2, colors[colorIndex][0], colors[colorIndex][1], colors[colorIndex][2]);
            }
            lastUV = uv;
        }
        ++colorIndex;
    }

    // TODO: make legend somehow. maybe a source image that gets loaded and slapped on/

    // TODO: draw axis loglines?

    // TODO: use golden ratio to come up with colors for each noise type based on noise index?

    SaveImage(fileName, image);
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
    for (auto& l : log.logs)
    {
        LogLineAppend(l, 0, "\"Sample\",\"N^-0.5\",\"N^-0.75\",\"N^-1\",\"White Noise\",\"Golden Ratio\",\"Blue Noise\",\"Projective Blue Noise\"");
        for (int i = 1; i <= NUM_SAMPLES(); ++i)
            LogLineAppend(l, i, "\"%i\",\"%f\",\"%f\"", i, powf(float(i), -0.5f), powf(float(i), -0.75f));
    }
    for (auto& e : log.errors[c_numSamplingPatterns+0])
    {
        for (int i = 1; i <= NUM_SAMPLES(); ++i)
            e.push_back(powf(float(i), -0.5f));
    }
    for (auto& e : log.errors[c_numSamplingPatterns+1])
    {
        for (int i = 1; i <= NUM_SAMPLES(); ++i)
            e.push_back(powf(float(i), -0.75f));
    }
    for (auto& e : log.errors[c_numSamplingPatterns+2])
    {
        for (int i = 1; i <= NUM_SAMPLES(); ++i)
            e.push_back(powf(float(i), -1.0f));
    }

    // make images of the functions we are integrating
    MakeSampleImage(SampleImage_Disk, "out/int_disk.png");
    MakeSampleImage(SampleImage_Triangle, "out/int_triangle.png");
    MakeSampleImage(SampleImage_Step, "out/int_step.png");
    MakeSampleImage(SampleImage_Gaussian, "out/int_gaussian.png");
    MakeSampleImage(SampleImage_Bilinear, "out/int_bilinear.png");

    // do the tests for each type of sampling
    for (size_t samplingPattern = 0; samplingPattern < c_numSamplingPatterns; ++samplingPattern)
    {
        const SamplingPattern& pattern = g_samplingPatterns[samplingPattern];
        if (!pattern.enable)
            continue;

        printf("%s...\n", pattern.nameHuman);
        DoTest2D(pattern.generatePoints, log, pattern.nameFile, (int)samplingPattern);
    }

    // make error graphs
    printf("Making Graphs...\n");
    MakeErrorGraph(log, 0, "out/error_disk.png");
    MakeErrorGraph(log, 1, "out/error_triangle.png");
    MakeErrorGraph(log, 2, "out/error_step.png");
    MakeErrorGraph(log, 3, "out/error_gaussian.png");
    MakeErrorGraph(log, 4, "out/error_bilinear.png");

    // write out the logs
    printf("Writing CSVs...\n");
    WriteLog(log.logs[0], "out/data_disk.csv");
    WriteLog(log.logs[1], "out/data_triangle.csv");
    WriteLog(log.logs[2], "out/data_step.csv");
    WriteLog(log.logs[3], "out/data_gaussian.csv");
    WriteLog(log.logs[4], "out/data_bilinear.csv");

    return 0;
}

/*
TODO:

* need to figure out how to calculate: spectrum radial average, spectrum anisotropy, 1d power specutrum (vs the 2d power spectrum)

* soft shadow of a sphere on a plane

* for integration, average multiple runs

* use acceleration structure for blue noise too.

* how to compare projblue and projblue2?
* need an anisotropy thingy from the DFT for one
* need a histogram of projected points i think.

* there's a crash when some of the sampling is disabled. (eg white noise)

* formalize the tests too

* in good candidate algorithm, rename scores terminology to subspaces

* maybe have regular blue noise use acceleration structure too? unsure if needed.

* test projective blue noise with lower sample count but higher multiplier
 * find a good value for projective blue noise, even if it's restrictively high?
 * should be better after acceleration structures are added

* acceleration structure for blue noise and projective blue noise

* flatten the best / good candidate so it doesn't take lambdas, and knows how to do comparisons / generation internally for vector dimensions.

* make the tests and sample types be structs so fewer magic numbers

* get stuff from email.
 * like the other golden ratio pattern w/ randomization.
 * also do something graphical (soft shadow?)

* make the graph have a border and tick marks / axis marks and a legend somehow.
* look through comments and get rid of anything that was copy/paste

* todos
* generalized golden ratio isn't doing that well for integration seemingly. why not, are you doing it right?
* need more sampling patterns. especially LDS if golden ratio is going to suck it up.
 * like owen scrambled sobol
* it could be nice if you made the graphs automatically. they are kind of a pain in open office, and don't look like I want.
* expand the sampling box out a little bit so points don't overlap it!
* dft sampling patterns - how?

* show results to Martin on twitter, see what he says

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