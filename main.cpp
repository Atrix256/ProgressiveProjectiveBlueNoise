#include <stdio.h>
#include <vector>
#include <array>
#include <random>
#include <string>
#include <stdint.h>
#include <stdarg.h>

#include "image.h"
#include "raytrace.h"
#include "AO.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TEST_IMAGE_SIZE() 128 // in pixels, on each axis
#define SAMPLE_IMAGE_SIZE() 1024
#define GRAPH_IMAGE_SIZE() 1024
#define NUM_SAMPLES() 1024
#define NUM_SAMPLES_SMALL() 32
#define DO_SLOW_SAMPLES() true

#define BLUENOISE_CANDIDATE_MULTIPLIER() 5

#define PROJBLUENOISE_CANDIDATE_MULTIPLIER() 100
#define PROJBLUENOISE_PARTITIONS() 10

#define DO_RAYTRACING() false
#define DO_GROUND_TRUTH_RAYTRACE() false
#define DO_AO_RAYTRACE() false
#define GROUND_TRUTH_SAMPLES() 10000
#define RAYTRACE_IMAGE_SIZE() 512

#define DO_DFT() false
#define DFT_IMAGE_SIZE() 256

#define DO_BLUR_TEST() false

#define DO_SAMPLING_TEST() false
#define SAMPLING_IMAGE_SIZE() 256

#define DO_SAMPLING_ZONEPLATE_TEST() true

static const float c_referenceValue_Disk = 0.5f;
static const float c_referenceValue_Triangle = 0.5f;
static const float c_referenceValue_Step = 1.0f / c_pi;
static const float c_referenceValue_Gaussian = c_pi / 4.0f * (float)erf(1.0) * (float)erf(1.0);
static const float c_referenceValue_Bilinear = 0.25f;

// generalized golden ratio, for making 2d low discrepancy sequences
// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
static const float c_goldenRatio = 1.61803398875f;
static const float c_goldenRatio2 = 1.32471795724474602596f;

typedef std::array<float, 2> Vec2;
typedef std::array<float, 3> Vec3;

typedef void(*GeneratePointsFN)(std::vector<Vec2>& points, size_t numPoints);

void GeneratePoints_GoldenRatio(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_GoldenRatio_Spiral(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_R2(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_R2_Spiral(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_R2_Spiral2(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_R2_Spiral3(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_R2_Jittered(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_WhiteNoise(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_Hammersley(std::vector<Vec2>& points, size_t numPoints);
void GeneratePoints_Sobol(std::vector<Vec2>& points, size_t numPoints);
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
    {"Golden Ratio", "golden", GeneratePoints_GoldenRatio, true},
    {"Golden Ratio Spiral", "goldens", GeneratePoints_GoldenRatio_Spiral, true},
    {"R2", "r2", GeneratePoints_R2, true},
    {"R2 Spiral2", "r2s2", GeneratePoints_R2_Spiral2, true},
    {"R2 Spiral3", "r2s3", GeneratePoints_R2_Spiral3, true},
    {"R2 Spiral", "r2s", GeneratePoints_R2_Spiral, true},
    {"R2 Jittered", "r2j", GeneratePoints_R2_Jittered, true},
    {"White Noise", "white", GeneratePoints_WhiteNoise, true},
    {"Hammersley", "hammersley", GeneratePoints_Hammersley, true},
    {"Sobol", "sobol", GeneratePoints_Sobol, true},
    {"Blue Noise", "blue", GeneratePoints_BlueNoise, DO_SLOW_SAMPLES()},
    {"Projective Blue Noise", "projblue", GeneratePoints_ProjectiveBlueNoise, DO_SLOW_SAMPLES()},
    {"Projective Blue Noise 2", "projblue2", GeneratePoints_ProjectiveBlueNoise2, DO_SLOW_SAMPLES()},
};
static const size_t c_numSamplingPatterns = sizeof(g_samplingPatterns) / sizeof(g_samplingPatterns[0]);

struct Log
{
    std::array<std::vector<std::string>, 5> logs;
    std::array<std::array<std::vector<float>, 5>, c_numSamplingPatterns+3> errors;  // indexed by: [sampleType][test]
    std::array<std::vector<Vec2>, c_numSamplingPatterns> points;
    std::array<std::vector<Vec2>, c_numSamplingPatterns> pointsSmall;

    std::array<std::vector<float>, c_numSamplingPatterns> samplingRMSE;
    std::vector<size_t> samplingRMSE_SampleCounts;

    std::array<std::vector<float>, c_numSamplingPatterns> samplingRMSEZP;
    std::vector<size_t> samplingRMSEZP_SampleCounts;
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

inline size_t Ruler(size_t n)
{
    size_t ret = 0;
    while (n != 0 && (n & 1) == 0)
    {
        n /= 2;
        ++ret;
    }
    return ret;
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

void GeneratePoints_R2(std::vector<Vec2>& points, size_t numPoints)
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

void GeneratePoints_R2_Jittered(std::vector<Vec2>& points, size_t numPoints)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    static const float a1 = 1.0f / c_goldenRatio2;
    static const float a2 = 1.0f / (c_goldenRatio2 * c_goldenRatio2);

    static const float c_magicNumber = 0.732f;

    points.resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        points[i][0] = fmodf(dist(RNG()) * c_magicNumber / sqrtf(float(i + 1)) + a1 * float(i), 1.0f);
        points[i][1] = fmodf(dist(RNG()) * c_magicNumber / sqrtf(float(i + 1)) + a2 * float(i), 1.0f);
    }
}

void GeneratePoints_GoldenRatio(std::vector<Vec2>& points, size_t numPoints)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    points.resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        points[i][0] = fmodf(0.5f + float(i) / float(numPoints), 1.0f);
        points[i][1] = fmodf(0.5f + float(i) / c_goldenRatio, 1.0f);
    }
}

void GeneratePoints_GoldenRatio_Spiral(std::vector<Vec2>& points, size_t numPoints)
{
    /*
    Math Notes:
        * The golden spiral creates points in a circle.
        * We only want to keep the points inside of the square in that circle.
        * That means we reject some number of points.
        * The percentage of points rejected is (area of circle - area of square) / (area of circle)
        * with a circle of radius sqrt(2), that circle has area of 2pi
        * square has side length of 2, so area of 4.
        * percentage rejected = (2pi - 4) / (2pi) = ~0.363
        * So, in our r1 value, we want to increase the "numpoints" by that ratio i think...

        ! However: changing the numpoints changes the location of the points, making different points get rejected so this helps the problem but doesn't fix it.
          I multiplied this by hand tuned numbers til i came up with 1.55.
    */

    // Note: this "doubles up" on points because it discards some of the points.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float angleOffset = dist(RNG());
    float radiusOffset = 0.0f;// dist(RNG());

    static const float c_sqrt2 = sqrtf(2.0f);

    float numPointsMultiplier = 1.0f + (2.0f * c_pi - 4.0f) / (2.0f * c_pi) * 1.55f;

    points.resize(numPoints);
    int candidateIndex = 1;
    int pointCount = 0;
    while (pointCount < numPoints)
    {
        float r1 = fmodf(0.5f + float(candidateIndex) / (float(numPoints)*numPointsMultiplier), 1.0f);
        float r2 = fmodf(0.5f + float(candidateIndex) / c_goldenRatio, 1.0f);
        ++candidateIndex;

        float angle = std::fmodf(r1 + angleOffset, 1.0f) * 2.0f * c_pi;
        float radius = sqrtf(std::fmodf(r2 + radiusOffset, 1.0f)) * c_sqrt2;

        points[pointCount][0] = radius * cosf(angle) * 0.5f + 0.5f;
        points[pointCount][1] = radius * sinf(angle) * 0.5f + 0.5f;

        if (points[pointCount][0] < 0.0f ||
            points[pointCount][0] >= 1.0f ||
            points[pointCount][1] < 0.0f ||
            points[pointCount][1] >= 1.0f)
        {
            continue;
        }

        ++pointCount;
    }
}

void GeneratePoints_R2_Spiral(std::vector<Vec2>& points, size_t numPoints)
{
    static const float a1 = 1.0f / c_goldenRatio2;
    static const float a2 = 1.0f / (c_goldenRatio2 * c_goldenRatio2);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float angleOffset = dist(RNG());
    float radiusOffset = 0.0f;// dist(RNG());

    static const float c_sqrt2 = sqrtf(2.0f);

    points.resize(numPoints);
    int candidateIndex = 1;
    int pointCount = 0;
    while (pointCount < numPoints)
    {
        float r1 = fmodf(0.5f + a1 * float(candidateIndex), 1.0f);
        float r2 = fmodf(0.5f + a2 * float(candidateIndex), 1.0f);
        ++candidateIndex;

        float angle = std::fmodf(r1 + angleOffset, 1.0f) * 2.0f * c_pi;
        float radius = sqrtf(std::fmodf(r2 + radiusOffset, 1.0f)) * c_sqrt2;

        points[pointCount][0] = radius * cosf(angle) * 0.5f + 0.5f;
        points[pointCount][1] = radius * sinf(angle) * 0.5f + 0.5f;

        if (points[pointCount][0] < 0.0f ||
            points[pointCount][0] >= 1.0f ||
            points[pointCount][1] < 0.0f ||
            points[pointCount][1] >= 1.0f)
        {
            continue;
        }

        ++pointCount;
    }
}

// "A Low Distortion Map Between Disk and Square"
// https://pdfs.semanticscholar.org/4322/6a3916a85025acbb3a58c17f6dc0756b35ac.pdf
Vec2 ToUnitDisk(const Vec2& onSquare)
{
    float phi, r, u, v;
    float a = 2.0f * onSquare[0] - 1.0f; // (a,b) is now on [-1,1]ˆ2
    float b = 2.0f * onSquare[1] - 1.0f;
    if (a > -b) // region 1 or 2
    {
        if (a > b) // region 1, also |a| > |b|
        {
            r = a;
            phi = (c_pi / 4.0f) * (b / a);
        }
        else // region 2, also |b| > |a|
        {
            r = b;
            phi = (c_pi / 4.0f) * (2.0f - (a / b));
        }
    }
    else // region 3 or 4
    {
        if (a < b) // region 3, also |a| >= |b|, a != 0
        {
            r = -a;
            phi = (c_pi / 4.0f) * (4.0f + (b / a));
        }
        else // region 4, |b| >= |a|, but a==0 and b==0 could occur.
        {
            r = -b;
            if (b != 0.0f)
                phi = (c_pi / 4.0f) * (6.0f - (a / b));
            else
                phi = 0.0f;
        }
    }

    u = r * cos(phi);
    v = r * sin(phi);
    return { u, v };
}

void GeneratePoints_R2_Spiral2(std::vector<Vec2>& points, size_t numPoints)
{
    // Got this from Martin https://twitter.com/TechSparx
    //(x,y) = (x * sqrt(1-y*y/2), y * sqrt(1-x*x/2)

    static const float a1 = 1.0f / c_goldenRatio2;
    static const float a2 = 1.0f / (c_goldenRatio2 * c_goldenRatio2);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    static const float c_sqrt2 = sqrtf(2.0f);

    points.resize(numPoints);
    int candidateIndex = 1;
    int pointCount = 0;
    while (pointCount < numPoints)
    {
        float r1 = fmodf(0.5f + a1 * float(candidateIndex), 1.0f) * 2.0f - 1.0f;
        float r2 = fmodf(0.5f + a2 * float(candidateIndex), 1.0f) * 2.0f - 1.0f;
        ++candidateIndex;

        points[pointCount][0] = r1 * sqrtf(1.0f - r2 * r2 / 2.0f) * c_sqrt2 * 0.5f + 0.5f;
        points[pointCount][1] = r2 * sqrtf(1.0f - r1 * r1 / 2.0f) * c_sqrt2 * 0.5f + 0.5f;

        if (points[pointCount][0] < 0.0f ||
            points[pointCount][0] >= 1.0f ||
            points[pointCount][1] < 0.0f ||
            points[pointCount][1] >= 1.0f)
        {
            continue;
        }

        ++pointCount;
    }
}

void GeneratePoints_R2_Spiral3(std::vector<Vec2>& points, size_t numPoints)
{
    // Got this from "A Low Distortion Map Between Disk and Square"
    //https://pdfs.semanticscholar.org/4322/6a3916a85025acbb3a58c17f6dc0756b35ac.pdf

    static const float a1 = 1.0f / c_goldenRatio2;
    static const float a2 = 1.0f / (c_goldenRatio2 * c_goldenRatio2);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    static const float c_sqrt2 = sqrtf(2.0f);

    points.resize(numPoints);
    int candidateIndex = 1;
    int pointCount = 0;
    while (pointCount < numPoints)
    {
        float r1 = fmodf(0.5f + a1 * float(candidateIndex), 1.0f);// *2.0f - 1.0f;
        float r2 = fmodf(0.5f + a2 * float(candidateIndex), 1.0f);// *2.0f - 1.0f;
        ++candidateIndex;

        points[pointCount] = ToUnitDisk({ r1, r2 });
        points[pointCount][0] = points[pointCount][0] * c_sqrt2 * 0.5f + 0.5f;
        points[pointCount][1] = points[pointCount][1] * c_sqrt2 * 0.5f + 0.5f;

        if (points[pointCount][0] < 0.0f ||
            points[pointCount][0] >= 1.0f ||
            points[pointCount][1] < 0.0f ||
            points[pointCount][1] >= 1.0f)
        {
            continue;
        }

        ++pointCount;
    }
}

void GeneratePoints_Hammersley(std::vector<Vec2>& points, size_t numPoints)
{
    // figure out how many bits we are working in.
    size_t value = 1;
    size_t numBits = 0;
    while (value < numPoints)
    {
        value *= 2;
        ++numBits;
    }

    size_t truncateBits = 0;

    // calculate the sample points
    points.resize(numPoints);
    size_t sampleInt = 0;
    for (size_t i = 0; i < numPoints; ++i)
    {
        // x axis
        points[i][0] = 0.0f;
        {
            size_t n = i >> truncateBits;
            float base = 1.0f / 2.0f;
            while (n)
            {
                if (n & 1)
                    points[i][0] += base;
                n /= 2;
                base /= 2.0f;
            }
        }

        // y axis
        points[i][1] = 0.0f;
        {
            size_t n = i >> truncateBits;
            size_t mask = size_t(1) << (numBits - 1 - truncateBits);

            float base = 1.0f / 2.0f;
            while (mask)
            {
                if (n & mask)
                    points[i][1] += base;
                mask /= 2;
                base /= 2.0f;
            }
        }
    }
}

void GeneratePoints_Sobol(std::vector<Vec2>& points, size_t numPoints)
{
    // x axis
    points.resize(numPoints);
    size_t sampleInt = 0;
    for (size_t i = 0; i < numPoints; ++i)
    {
        size_t ruler = Ruler(i + 1);
        size_t direction = size_t(size_t(1) << size_t(31 - ruler));
        sampleInt = sampleInt ^ direction;
        points[i][0] = float(sampleInt) / std::pow(2.0f, 32.0f);
    }

    // y axis
    // Code adapted from http://web.maths.unsw.edu.au/~fkuo/sobol/
    // uses numbers: new-joe-kuo-6.21201

    // Direction numbers
    std::vector<size_t> V;
    V.resize((size_t)ceil(log((double)numPoints+1) / log(2.0)));
    V[0] = size_t(1) << size_t(31);
    for (size_t i = 1; i < V.size(); ++i)
        V[i] = V[i - 1] ^ (V[i - 1] >> 1);

    // Samples
    sampleInt = 0;
    for (size_t i = 0; i < numPoints; ++i) {
        size_t ruler = Ruler(i + 1);
        sampleInt = sampleInt ^ V[ruler];
        points[i][1] = float(sampleInt) / std::pow(2.0f, 32.0f);
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

void MakeSamplesImage(std::vector<Vec2>& points, const char* label, bool small)
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

    for (int sampleIndex = 0; sampleIndex < points.size(); ++sampleIndex)
    {
        const Vec2& v = points[sampleIndex];
        float percent = float(sampleIndex) / float(points.size() - 1);
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
    sprintf_s(fileName, "out/samples_%s%s.png", label, small ? "_sm" : "");
    SaveImage(fileName, image);

    // also write the samples out as a csv
    {
        FILE* file = nullptr;
        sprintf_s(fileName, "out/samplescsv_%s%s.csv", label, small ? "_sm" : "");
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

        Image dft;
        AppendImageVertical(dft, imageSamples, imageSamplesDFT);
        sprintf_s(fileName, "out/samplesdft_%s%s.png", label, small ? "_sm" : "");
        SaveImage(fileName, dft);
    }
}

void GroundTruthRaytrace()
{
#if DO_GROUND_TRUTH_RAYTRACE() == false
    return;
#endif

    std::vector<Vec2> points;
    points.resize(GROUND_TRUTH_SAMPLES());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : points)
    {
        v[0] = dist(RNG());
        v[1] = dist(RNG());
    }

    // make a white noise random number per pixel for Cranley Patterson Rotation.
    static std::vector<Vec2> whiteNoise;
    if (whiteNoise.size() == 0)
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        whiteNoise.resize(RAYTRACE_IMAGE_SIZE() * RAYTRACE_IMAGE_SIZE());
        for (Vec2& v : whiteNoise)
        {
            v[0] = dist(RNG());
            v[1] = dist(RNG());
        }
    }

    ImageFloat resultFloat(RAYTRACE_IMAGE_SIZE(), RAYTRACE_IMAGE_SIZE());
    Image result;

    RaytraceTest(resultFloat, 0, GROUND_TRUTH_SAMPLES(), points, whiteNoise, false);
    ImageFloatToImage(resultFloat, result);
    SaveImage("out/raytrace/__truth.png", result);
    SaveImage("out/raytrace_correlated/__truth.png", result);
}

void DoTestRaytrace(const std::vector<Vec2>& points, const char* label)
{
#if DO_RAYTRACING() == false
    return;
#endif

    // make a white noise random number per pixel for Cranley Patterson Rotation.
    static std::vector<Vec2> whiteNoise;
    if (whiteNoise.size() == 0)
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        whiteNoise.resize(RAYTRACE_IMAGE_SIZE() * RAYTRACE_IMAGE_SIZE());
        for (Vec2& v : whiteNoise)
        {
            v[0] = dist(RNG());
            v[1] = dist(RNG());
        }
    }

    ImageFloat resultFloat(RAYTRACE_IMAGE_SIZE(), RAYTRACE_IMAGE_SIZE());
    Image result;
    char fileName[256];

    static const size_t c_sampleCounts[] =
    {
        0, 1, 16, 128, 256, 512, 1024
    };

    for (size_t index = 0; index < sizeof(c_sampleCounts) / sizeof(c_sampleCounts[0]) - 1; ++index)
    {
        RaytraceTest(resultFloat, c_sampleCounts[index], c_sampleCounts[index + 1], points, whiteNoise, true);
        ImageFloatToImage(resultFloat, result);
        sprintf_s(fileName, "out/raytrace/%s_%zu.png", label, c_sampleCounts[index+1]);
        SaveImage(fileName, result);
    }
    for (size_t index = 0; index < sizeof(c_sampleCounts) / sizeof(c_sampleCounts[0]) - 1; ++index)
    {
        RaytraceTest(resultFloat, c_sampleCounts[index], c_sampleCounts[index + 1], points, whiteNoise, false);
        ImageFloatToImage(resultFloat, result);
        sprintf_s(fileName, "out/raytrace_correlated/%s_%zu.png", label, c_sampleCounts[index + 1]);
        SaveImage(fileName, result);
    }
}

void DoTestAO(const std::vector<Vec2>& points, const char* label)
{
#if DO_AO_RAYTRACE() == false
    return;
#endif

    // make a white noise random number per pixel for Cranley Patterson Rotation.
    static std::vector<Vec2> whiteNoise;
    if (whiteNoise.size() == 0)
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        whiteNoise.resize(RAYTRACE_IMAGE_SIZE() * RAYTRACE_IMAGE_SIZE());
        for (Vec2& v : whiteNoise)
        {
            v[0] = dist(RNG());
            v[1] = dist(RNG());
        }
    }

    ImageFloat resultFloat(RAYTRACE_IMAGE_SIZE(), RAYTRACE_IMAGE_SIZE());
    Image result;
    char fileName[256];

    static const size_t c_sampleCounts[] =
    {
        0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    };

    for (size_t index = 0; index < sizeof(c_sampleCounts) / sizeof(c_sampleCounts[0]) - 1; ++index)
    {
        AOTest(resultFloat, c_sampleCounts[index], c_sampleCounts[index + 1], points, whiteNoise, true);
        ImageFloatToImage(resultFloat, result);
        sprintf_s(fileName, "out/AO/%s_%zu.png", label, c_sampleCounts[index+1]);
        SaveImage(fileName, result);
    }
    for (size_t index = 0; index < sizeof(c_sampleCounts) / sizeof(c_sampleCounts[0]) - 1; ++index)
    {
        AOTest(resultFloat, c_sampleCounts[index], c_sampleCounts[index + 1], points, whiteNoise, false);
        ImageFloatToImage(resultFloat, result);
        sprintf_s(fileName, "out/AO_correlated/%s_%zu.png", label, c_sampleCounts[index + 1]);
        SaveImage(fileName, result);
    }
}

void DoTestBlur(const std::vector<Vec2>& points, const char* label)
{
#if DO_BLUR_TEST() == false
    return;
#endif

    static bool imageLoaded = false;
    static ImageFloat srcImage;
    if (!imageLoaded)
    {
        imageLoaded = true;
        int width, height, components;
        float* pixels = stbi_loadf("scenery.png", &width, &height, &components, 4);

        srcImage = ImageFloat(width, height);

        memcpy(srcImage.m_pixels.data(), pixels, width*height * 4 * sizeof(float));

        stbi_image_free(pixels);        
    }

    ImageFloat resultFloat(srcImage.m_width, srcImage.m_height);
    Image result;
    char fileName[256];

    // diameter = c_kernelRadius * 2 + 1
    static const size_t c_kernelRadius = 7; // 15x15 = 225 samples
    static const size_t c_sampleSteps = 10; // this many steps from 0 to 100%

    static const size_t c_sampleDiameter = (c_kernelRadius * 2 + 1);
    static const size_t c_totalSampleCount = c_sampleDiameter * c_sampleDiameter;

    size_t samplesTaken = 0;

    // TODO: try to get samplecount in the inner loop so it's better for the cache

    for (size_t index = 0; index < c_sampleSteps + 1; ++index)
    {
        size_t sampleCount;
        if (index == 0)
            sampleCount = 1;
        else
            sampleCount = size_t(float(c_totalSampleCount) * float((index - 1) + 1) / float(c_sampleSteps));

        for (; samplesTaken < sampleCount; ++samplesTaken)
        {
            float lerpAmount = 1.0f / float(samplesTaken + 1);

            for (int y = 0; y < srcImage.m_height; ++y)
            {
                for (int x = 0; x < srcImage.m_width; ++x)
                {
                    const Vec2& sampleFloat = points[samplesTaken];

                    int sampleOffsetX = int(Clamp((sampleFloat[0] * float(c_sampleDiameter)), 0.0f, c_sampleDiameter - 1)) - c_kernelRadius;
                    int sampleOffsetY = int(Clamp((sampleFloat[1] * float(c_sampleDiameter)), 0.0f, c_sampleDiameter - 1)) - c_kernelRadius;

                    int sampleX = x + sampleOffsetX;
                    int sampleY = y + sampleOffsetY;

                    if (sampleX < 0)
                        sampleX = 0;
                    if (sampleX > srcImage.m_width - 1)
                        sampleX = srcImage.m_width - 1;

                    if (sampleY < 0)
                        sampleY = 0;
                    if (sampleY > srcImage.m_height - 1)
                        sampleY = srcImage.m_height - 1;

                    // TODO: set this to start of image and increment each loop?
                    float* destPixel = &resultFloat.m_pixels[(y*srcImage.m_width + x) * 4];
                    
                    const float* srcPixel = &srcImage.m_pixels[(sampleY*srcImage.m_width + sampleX) * 4];

                    for (int i = 0; i < 3; ++i)
                        destPixel[i] = Lerp(destPixel[i], srcPixel[i], lerpAmount);

                    destPixel[3] = 1.0f;
                }
            }
        }

        ImageFloatToImage(resultFloat, result);
        sprintf_s(fileName, "out/Blur/%s_%zu.png", label, sampleCount);
        SaveImage(fileName, result);
    }

    // TODO: make a ground truth version that is exhaustive! nmaybe do that in the !imageLoaded block.
}

// This function is from Nathan Reed
// Python Code: https://gist.github.com/Reedbeta/893b63390160e33ddb3c#file-antialias-test-py-L36-L44
// Blog Post: http://reedbeta.com/blog/antialiasing-to-splat-or-not/
float SampleFreqFunc(float x, float y)
{
    float minPeriod = 2e-5f;
    float maxPeriod = 0.2f;
    float period = minPeriod + (maxPeriod - minPeriod) * y*y;
    float phase = x / period;
    phase -= floorf(phase);
    return roundf(phase);
}

void DoTestSampling(const std::vector<Vec2>& points, std::vector<float>& RMSE, std::vector<size_t>& sampleCountArray, const char* label)
{
    #if DO_SAMPLING_TEST() == false
    return;
    #endif

    static const float pixelSizeUV = 1.0f / float(SAMPLING_IMAGE_SIZE());

    // make a ground truth, the first time we call this function
    static ImageFloat groundTruth(SAMPLING_IMAGE_SIZE(), SAMPLING_IMAGE_SIZE());
    static bool firstCall = true;
    bool thisIsTheFirstCall = firstCall;
    if (firstCall)
    {
        firstCall = false;
        static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float* pixel = groundTruth.m_pixels.data();
        for (size_t y = 0; y < SAMPLING_IMAGE_SIZE(); ++y)
        {
            for (size_t x = 0; x < SAMPLING_IMAGE_SIZE(); ++x)
            {
                for (size_t sampleIndex = 1; sampleIndex <= GROUND_TRUTH_SAMPLES(); ++sampleIndex)
                {
                    Vec2 jitteredUV;
                    jitteredUV[0] = (float(x) + dist(RNG())) * pixelSizeUV;
                    jitteredUV[1] = (float(y) + dist(RNG())) * pixelSizeUV;

                    float value = SampleFreqFunc(jitteredUV[0], jitteredUV[1]);
                    pixel[0] = Lerp(pixel[0], value, 1.0f / float(sampleIndex + 1));
                }

                pixel[1] = pixel[2] = pixel[0];
                pixel[3] = 1.0f;
                pixel += 4;
            }
        }

        Image result;
        char fileName[256];
        ImageFloatToImage(groundTruth, result);
        sprintf_s(fileName, "out/Sampling/__truth.png");
        SaveImage(fileName, result);
    }

    size_t sampleCounts[] =
    {
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024
    };

    if (thisIsTheFirstCall)
    {
        sampleCountArray.resize(sizeof(sampleCounts) / sizeof(sampleCounts[0]) - 1);
        for (size_t i = 0; i < sampleCountArray.size(); ++i)
            sampleCountArray[i] = sampleCounts[i + 1];
    }

    ImageFloat resultFloat(SAMPLING_IMAGE_SIZE(), SAMPLING_IMAGE_SIZE());
    for (size_t sampleCountIndex = 1; sampleCountIndex < sizeof(sampleCounts) / sizeof(sampleCounts[0]); ++sampleCountIndex)
    {
        float* pixel = resultFloat.m_pixels.data();
        for (size_t y = 0; y < SAMPLING_IMAGE_SIZE(); ++y)
        {
            for (size_t x = 0; x < SAMPLING_IMAGE_SIZE(); ++x)
            {
                for (size_t sampleIndex = sampleCounts[sampleCountIndex-1]; sampleIndex < sampleCounts[sampleCountIndex]; ++sampleIndex)
                {
                    Vec2 jitteredUV;
                    jitteredUV[0] = (float(x) + points[sampleIndex][0]) * pixelSizeUV;
                    jitteredUV[1] = (float(y) + points[sampleIndex][1]) * pixelSizeUV;

                    float value = SampleFreqFunc(jitteredUV[0], jitteredUV[1]);
                    pixel[0] = Lerp(pixel[0], value, 1.0f / float(sampleIndex + 1));
                }

                pixel[1] = pixel[2] = pixel[0];
                pixel[3] = 1.0f;
                pixel += 4;
            }
        }

        char fileName[1024];
        Image result;
        ImageFloatToImage(resultFloat, result);
        sprintf_s(fileName, "out/Sampling/%s_%zu.png", label, sampleCounts[sampleCountIndex]);
        SaveImage(fileName, result);

        float rootMeanSquaredError = sqrtf(MeanSquaredError(resultFloat, groundTruth));
        RMSE.push_back(rootMeanSquaredError);
    }
}

// Looking at aliasing vs noise when sampling sine waves of different frequencies using different types of noise
// Making a zone plane: https://blogs.mathworks.com/steve/2011/07/19/jahne-test-pattern-take-3/ and https://blogs.mathworks.com/steve/2011/07/22/filtering-fun/
template <bool ANTIALIAS, bool ONEQUADRANT, size_t SIZE>
float SampleZonePlate(float x, float y, float frequencyMultiplier)
{
    double km = 0.8 * c_pi;
    double rm = (ONEQUADRANT ? double(SIZE) : double(SIZE/2)) / frequencyMultiplier;
    double w = rm / 10.0;

    double r;
    if (ONEQUADRANT)
        r = std::hypot(double(x), double(y));
    else
        r = std::hypot(double(x-SIZE/2), double(y-SIZE/2));

    double term1 = sin((km * r * r) / (2.0 * rm));
    double term2 = 0.5 * tanh((rm - r) / w) + 0.5;

    // with anti aliasing via the tanh function
    if (ANTIALIAS)
        return float((term1 * term2 + 1.0) / 2.0);
    // without anti aliasing
    else
        return float((term1 + 1.0) / 2.0);
}

// TODO: 2d zone plate and 1d next to it!
// TODO: maybe do higher frequencies so it aliases?

void DoTestSamplingZonePlate(const std::vector<Vec2>& points, std::vector<float>& RMSE, std::vector<size_t>& sampleCountArray, const char* label)
{
    #if DO_SAMPLING_ZONEPLATE_TEST() == false
    return;
    #endif

    static const float pixelSizeUV = 1.0f / float(SAMPLING_IMAGE_SIZE());

    // make a ground truth, the first time we call this function
    static ImageFloat groundTruth(SAMPLING_IMAGE_SIZE(), SAMPLING_IMAGE_SIZE());
    static bool firstCall = true;
    bool thisIsTheFirstCall = firstCall;
    if (firstCall)
    {
        firstCall = false;
        static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float* pixel = groundTruth.m_pixels.data();
        for (size_t y = 0; y < SAMPLING_IMAGE_SIZE(); ++y)
        {
            for (size_t x = 0; x < SAMPLING_IMAGE_SIZE(); ++x)
            {
                for (size_t sampleIndex = 1; sampleIndex <= GROUND_TRUTH_SAMPLES(); ++sampleIndex)
                {
                    Vec2 jitteredUV;
                    jitteredUV[0] = (float(x) + dist(RNG()));
                    jitteredUV[1] = (float(y) + dist(RNG()));

                    float value = SampleZonePlate<false, false, SAMPLING_IMAGE_SIZE()>(jitteredUV[0], jitteredUV[1], 4.0f);
                    pixel[0] = Lerp(pixel[0], value, 1.0f / float(sampleIndex + 1));
                }

                pixel[1] = pixel[2] = pixel[0];
                pixel[3] = 1.0f;
                pixel += 4;
            }
        }

        Image result;
        char fileName[256];
        ImageFloatToImage(groundTruth, result);
        sprintf_s(fileName, "out/SamplingZP/__truth.png");
        SaveImage(fileName, result);
    }

    size_t sampleCounts[] =
    {
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024
    };

    if (thisIsTheFirstCall)
    {
        sampleCountArray.resize(sizeof(sampleCounts) / sizeof(sampleCounts[0]) - 1);
        for (size_t i = 0; i < sampleCountArray.size(); ++i)
            sampleCountArray[i] = sampleCounts[i + 1];
    }

    ImageFloat resultFloat(SAMPLING_IMAGE_SIZE(), SAMPLING_IMAGE_SIZE());
    for (size_t sampleCountIndex = 1; sampleCountIndex < sizeof(sampleCounts) / sizeof(sampleCounts[0]); ++sampleCountIndex)
    {
        float* pixel = resultFloat.m_pixels.data();
        for (size_t y = 0; y < SAMPLING_IMAGE_SIZE(); ++y)
        {
            for (size_t x = 0; x < SAMPLING_IMAGE_SIZE(); ++x)
            {
                for (size_t sampleIndex = sampleCounts[sampleCountIndex-1]; sampleIndex < sampleCounts[sampleCountIndex]; ++sampleIndex)
                {
                    Vec2 jitteredUV;
                    jitteredUV[0] = (float(x) + points[sampleIndex][0]);
                    jitteredUV[1] = (float(y) + points[sampleIndex][1]);

                    float value = SampleZonePlate<false, false, SAMPLING_IMAGE_SIZE()>(jitteredUV[0], jitteredUV[1], 4.0f);
                    pixel[0] = Lerp(pixel[0], value, 1.0f / float(sampleIndex + 1));
                }

                pixel[1] = pixel[2] = pixel[0];
                pixel[3] = 1.0f;
                pixel += 4;
            }
        }

        char fileName[1024];
        Image result;
        ImageFloatToImage(resultFloat, result);
        sprintf_s(fileName, "out/SamplingZP/%s_%zu.png", label, sampleCounts[sampleCountIndex]);
        SaveImage(fileName, result);

        float rootMeanSquaredError = sqrtf(MeanSquaredError(resultFloat, groundTruth));
        RMSE.push_back(rootMeanSquaredError);
    }
}

void DoTest2D (const GeneratePoints& generatePoints, Log& log, const char* label, int noiseType)
{
    // generate the sample points and save them as an image
    generatePoints(log.pointsSmall[noiseType], NUM_SAMPLES_SMALL());
    MakeSamplesImage(log.pointsSmall[noiseType], label, true);
    generatePoints(log.points[noiseType], NUM_SAMPLES());
    MakeSamplesImage(log.points[noiseType], label, false);

    // test the sample points for integration
    Integrate(SampleImage_Disk,     log.points[noiseType], c_referenceValue_Disk,     log.logs[0], log.errors[noiseType][0]);
    Integrate(SampleImage_Triangle, log.points[noiseType], c_referenceValue_Triangle, log.logs[1], log.errors[noiseType][1]);
    Integrate(SampleImage_Step,     log.points[noiseType], c_referenceValue_Step,     log.logs[2], log.errors[noiseType][2]);
    Integrate(SampleImage_Gaussian, log.points[noiseType], c_referenceValue_Gaussian, log.logs[3], log.errors[noiseType][3]);
    Integrate(SampleImage_Bilinear, log.points[noiseType], c_referenceValue_Bilinear, log.logs[4], log.errors[noiseType][4]);
}

// Note: could use generalized golden ratio to come up with parameters to S and/or V too by making 2d or 3d low discrepancy sequences.
// generalized golden ratio here: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
Vec3 HUEtoRGB(float H)
{
    Vec3 ret;
    ret[0] = Clamp(abs(H * 6.0f - 3.0f) - 1.0f, 0.0f, 1.0f);
    ret[1] = Clamp(2.0f - abs(H * 6.0f - 2.0f), 0.0f, 1.0f);
    ret[2] = Clamp(2.0f - abs(H * 6.0f - 4.0f), 0.0f, 1.0f);
    return ret;
}

Vec3 HSVtoRGB(const Vec3& HSV)
{
    Vec3 RGB = HUEtoRGB(HSV[0]);

    Vec3 ret;
    for (int i = 0; i < 3; ++i)
        ret[i] = ((RGB[i] - 1.0f) * HSV[1] + 1.0f) * HSV[2];
    return ret;
}

Vec3 IndexToColor(int index)
{
    // use the golden ratio to make N colors that are very different from each other.
    static const float c_goldenRatioConjugate = 0.618033988749895f;
    float h = std::fmodf(c_goldenRatioConjugate * float(index), 1.0f);
    return HSVtoRGB({ h, 0.75f, 0.95f });
}

void MakeErrorGraph(const Log& log, int test, const char* fileName)
{
    Image image(GRAPH_IMAGE_SIZE(), GRAPH_IMAGE_SIZE());
    ClearImage(image, 224, 224, 224);

    // get the x axis min and max
    float xAxisMin = log10f(1.0f);
    float xAxisMax = ceilf(log10f(float(NUM_SAMPLES())));

    // get the y axis min and max
    float yAxisMin = 0.0f;
    float yAxisMax = 0.0f;
    bool foundSample = false;
    for (auto& a : log.errors)
    {
        for (auto& b : a[test])
        {
            if (foundSample)
            {
                yAxisMin = std::min(yAxisMin, b);
                yAxisMax = std::max(yAxisMax, b);
            }
            else
            {
                foundSample = true;
                yAxisMin = b;
                yAxisMax = b;
            }
        }
    }
    yAxisMin = std::max(yAxisMin, 0.00001f);
    yAxisMin = log10f(yAxisMin);
    yAxisMax = log10f(yAxisMax);

    int colorIndex = 0;
    for (int sampleType = 0; sampleType < log.errors.size(); ++sampleType)
    {
        Vec3 colorFloat = IndexToColor(colorIndex);
        uint8 color[3];
        for (int i = 0; i < 3; ++i)
            color[i] = uint8(Clamp(colorFloat[i] * 255.0f + 0.5f, 0.0f, 255.0f));

        bool firstPoint = true;
        Vec2 lastUV;
        for (int sampleIndex = 0; sampleIndex < log.errors[sampleType][test].size(); ++sampleIndex)
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

                DrawLine(image, x1, y1, x2, y2, color[0], color[1], color[2]);
            }
            lastUV = uv;
        }
        ++colorIndex;
    }

    // TODO: make legend somehow. maybe a source image that gets loaded and slapped on/

    // TODO: draw axis loglines?

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
        LogLineAppend(l, 0, "\"Sample\",\"N^-0.5\",\"N^-0.75\",\"N^-1\"");

        for (int i = 0; i < c_numSamplingPatterns; ++i)
        {
            LogLineAppend(l, 0, ",\"");
            LogLineAppend(l, 0, g_samplingPatterns[i].nameHuman);
            LogLineAppend(l, 0, "\"");
        }

        for (int i = 1; i <= NUM_SAMPLES(); ++i)
            LogLineAppend(l, i, "\"%i\",\"%f\",\"%f\",\"%f\"", i, powf(float(i), -0.5f), powf(float(i), -0.75f), powf(float(i), -1.0f));
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
    GroundTruthRaytrace();
    for (size_t samplingPattern = 0; samplingPattern < c_numSamplingPatterns; ++samplingPattern)
    {
        const SamplingPattern& pattern = g_samplingPatterns[samplingPattern];
        if (!pattern.enable)
            continue;

        printf("%s...\n", pattern.nameHuman);
        DoTest2D(pattern.generatePoints, log, pattern.nameFile, (int)samplingPattern);

        DoTestRaytrace(log.points[samplingPattern], pattern.nameFile);
        DoTestAO(log.points[samplingPattern], pattern.nameFile);
        DoTestBlur(log.points[samplingPattern], pattern.nameFile);
        DoTestSampling(log.points[samplingPattern], log.samplingRMSE[samplingPattern], log.samplingRMSE_SampleCounts, pattern.nameFile);
        DoTestSamplingZonePlate(log.points[samplingPattern], log.samplingRMSEZP[samplingPattern], log.samplingRMSEZP_SampleCounts, pattern.nameFile);
    }

    // write out sampling RMSE
    if(DO_SAMPLING_TEST())
    {
        FILE* csvFile = nullptr;
        fopen_s(&csvFile, "out/Sampling/__RMSE.csv", "w+t");

        // labels
        fprintf(csvFile, "\"Sample Counts\"");
        for (size_t j = 0; j < c_numSamplingPatterns; ++j)
        {
            if (g_samplingPatterns[j].enable)
                fprintf(csvFile, ",\"%s\"", g_samplingPatterns[j].nameHuman);
        }
        fprintf(csvFile, "\n");

        // data
        for (size_t i = 0; i < log.samplingRMSE_SampleCounts.size(); ++i)
        {
            fprintf(csvFile, "\"%zu\"", log.samplingRMSE_SampleCounts[i]);
            for (size_t j = 0; j < c_numSamplingPatterns; ++j)
            {
                if(g_samplingPatterns[j].enable)
                    fprintf(csvFile, ",\"%f\"", log.samplingRMSE[j][i]);
            }
            fprintf(csvFile, "\n");
        }
        fprintf(csvFile, "\n");
        fclose(csvFile);
    }


    // write out zone plate sampling RMSE
    if(DO_SAMPLING_ZONEPLATE_TEST())
    {
        FILE* csvFile = nullptr;
        fopen_s(&csvFile, "out/SamplingZP/__RMSE.csv", "w+t");

        // labels
        fprintf(csvFile, "\"Sample Counts\"");
        for (size_t j = 0; j < c_numSamplingPatterns; ++j)
        {
            if (g_samplingPatterns[j].enable)
                fprintf(csvFile, ",\"%s\"", g_samplingPatterns[j].nameHuman);
        }
        fprintf(csvFile, "\n");

        // data
        for (size_t i = 0; i < log.samplingRMSEZP_SampleCounts.size(); ++i)
        {
            fprintf(csvFile, "\"%zu\"", log.samplingRMSEZP_SampleCounts[i]);
            for (size_t j = 0; j < c_numSamplingPatterns; ++j)
            {
                if (g_samplingPatterns[j].enable)
                    fprintf(csvFile, ",\"%f\"", log.samplingRMSEZP[j][i]);
            }
            fprintf(csvFile, "\n");
        }
        fprintf(csvFile, "\n");
        fclose(csvFile);
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

* screenspace AO is probably the sampling thing you want to try for R2

* may not be doing enough samples for ground truth in zone plate!

* make projected points be lines instead of dots.

* the thing about raytracing images being bad... it looks like it's just the first call? ground truth and golden ratio. check into that. uninitialized value? static variable doing something weird?

* idea for auto-comparing error graphs: integrate! ie get area under curve and use to score.
 * could also show winner at each sample count, up to the final amount.

* after getting info about your projective blue noise, and the R2 stuff, make MetaSampler - clean this up and make it a clean implementation / testing suite for sampling patterns. catch em all. make open source!

* calculate error of sampling tests vs the ground truth
 * make the ground truth

* maybe try the multijittered sequence too?

* sample nathan reed's function as a good 2d test

* double check that the integral result values you have hard coded are correct.
 * could just calculate them yourself via a whole bunch of white noise samples, then use that value for the rest of the tests.

* regenerate all the raytracing, etc images w/ new sample types

* the raytracing looks b0rked for some sample types, whats up with that?

* need to fetch before you can push, hrm.

* make dft of small sample counts!

* blur: Nah! decorrelation still needed (I think?)
* blur: hammersley doesn't look good because it isn't progressive, and the blur sample points isn't a power of 2!
* blur: make a ground truth
* blur: maybe 4/8/16 sample count would be more telling? showing the full range of all possible sample counts up to the non stochastic amount doesn't seem useful.

* another good usage case could be image resizing (and reconstruction filtering?)

* maybe should do 2d and 1d zone plate tests too.

* it seems like white noise (and GR2) aren't converging?! maybe try more samples to see if that's true or not.

* try without decorrelation to see what it looks like.

* if there are specific locations that show off quality better in your final renders, grab them from the render and blow them up (nearest neighbor) to view them zoomed in more easily.
 * yes, do this, and show an error metric for that tile, like projective blue noise does.
 * and make a ground truth with like 100,000 white noise samples?

* need owen scrambled hammersley

Note for how the raytracing works:
 * use Cranley-Patterson rotation to decorrelate samples between pixels.
 * aka pick a random number per pixel and add add that value to the samples, then use that value mod 1.
 * using blue noise for this instead of white would be good for quality of results, but that seems like it wouldn't be appropriate for the testing.

* make a header for vec2/vec3

* multithread the raytracing?

* we shouldn't generate new points for ray tracing, we should use existing
 * generate the max # of points between the # needed for the 2d thing, and needed for ray tracing.
 * make them into globals and pass the points to the ray tracer.

* add a histogram of some kind to the projected axes

* need to figure out how to calculate: spectrum radial average, spectrum anisotropy, 1d power specutrum (vs the 2d power spectrum)

* soft shadow of a sphere on a plane

* for integration of randomized techniques, average multiple runs

* use acceleration structure for blue noise too.

* how to compare projblue and projblue2?
 * integration
 * lower sample counts shows the projection stuff more easily
* need an anisotropy thingy from the DFT for one
* need a histogram of projected points i think.

* in good candidate algorithm, rename scores terminology to subspaces

* test projective blue noise with lower sample count but higher multiplier
 * find a good value for projective blue noise, even if it's restrictively high?
 * should be better after acceleration structures are added

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


----- Sample Fox or whatever -----

* multithreaded work!

* good logging setup

* autogen documentation?

* put stuff into caches when possible - like ground truth results
 * put that and output into a specific folder. can kill that folder to clear cache etc.

* read this and references and all other papers you can.
 * This specifically has info about calculating the power spectrum (not fourier magnitude!), normalizing it, radial averaging it, and has c++ source code to do so.
 * https://cs.dartmouth.edu/wjarosz/publications/subr16fourier.html

* also this paper "Quasi-Monte Carlo Sampling by Art B. Owen"
 * talks about lots and lots of stuff. digital nets, lattices, etc. also owen scrambling, for eg owen scrambled sobol.

* low discrepancy blue noise sampling implementation: https://twitter.com/dcoeurjo/status/1066029958676578304

* turning halton (?) into an ordered dithering pattern: https://twitter.com/pixelmager/status/1065717648074448896?s=03

* take the average of multiple sampling runs for the randomized ones to get a real reading

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


* maybe some useful test metrics from this paper:
"Discrepancy as a Quality Measure for Sample Distributions"
https://www.cs.utah.edu/~shirley/papers/euro91.pdf

*/