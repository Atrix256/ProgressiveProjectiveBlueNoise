#include "raytrace.h"
#include <array>

typedef std::array<float, 3> Vec3;

inline float LengthSQ(const Vec3& v)
{
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

inline Vec3 Normalize(const Vec3& v)
{
    float len = sqrtf(LengthSQ(v));
    Vec3 ret;
    ret[0] = v[0] / len;
    ret[1] = v[1] / len;
    ret[2] = v[2] / len;
    return ret;
}

inline Vec3 Cross(const Vec3& a, const Vec3& b)
{
    return
    {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

inline float Dot(const Vec3& a, const Vec3& b)
{
    return
        a[0] * b[0] +
        a[1] * b[1] +
        a[2] * b[2];
}

inline Vec3 operator * (const Vec3& v, float f)
{
    return
    {
        v[0] * f,
        v[1] * f,
        v[2] * f,
    };
}

inline Vec3 operator / (const Vec3& v, float f)
{
    return
    {
        v[0] / f,
        v[1] / f,
        v[2] / f,
    };
}

inline Vec3 operator *= (Vec3& v, float f)
{
    v[0] *= f;
    v[1] *= f;
    v[2] *= f;
    return v;
}

inline Vec3 operator += (Vec3& a, const Vec3& b)
{
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
    return a;
}

inline Vec3 operator * (const Vec3& a, const Vec3& b)
{
    return
    {
        a[0] * b[0],
        a[1] * b[1],
        a[2] * b[2],
    };
}

inline Vec3 operator + (const Vec3& a, const Vec3& b)
{
    return
    {
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    };
}

inline Vec3 operator - (const Vec3& a, const Vec3& b)
{
    return
    {
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    };
}

static const Vec3 c_ptCameraPos = { 0.0f, 0.0f, 0.0f };
static const Vec3 c_ptCameraFwd = { 0.0f, 0.0f, 1.0f };
static const Vec3 c_ptCameraUp  = { 0.0f, 1.0f, 0.0f };

static const float c_ptNearPlaneDistance = 0.1f;
static const float c_ptCameraVerticalFOV = 40.0f * c_pi / 180.0f;

struct Sphere
{
    int id;
    Vec3 position;
    float radius;
    Vec3 color;
};

struct Triangle
{
    int id;
    Vec3 A;
    Vec3 B;
    Vec3 C;
    Vec3 Normal;
    Vec3 color;
};

int g_nextId = 0;

static Sphere s_Spheres[] =
{
    {++g_nextId, {-1.0f, 0.0f, 5.0f}, 1.0f, {0.1f, 1.0f, 0.1f}}
};

static Triangle s_Triangles[] =
{
    {++g_nextId, {-10.0f, -2.0f, 0.0f}, {10.0f, -2.0f, 0.0f}, {10.0f, -2.0f, 20.0f}, {}, {0.5f, 0.5f, 0.5f}},
    {++g_nextId, {-10.0f, -2.0f, 0.0f}, {10.0f, -2.0f, 20.0f}, {-10.0f, -2.0f, 20.0f}, {}, {0.5f, 0.5f, 0.5f}},

    {++g_nextId, {0.0f, -2.0f, 5.5f}, {-0.75f, -1.0f, 5.5f}, {2.0f,-1.0f, 7.0f}, {}, {1.0f, 0.1f, 0.1f}},

    {++g_nextId, {1.0f, -0.5f, 4.0f}, {-1.25f, 1.5f, 4.0f}, {2.0f, 1.5f, 20.0f}, {}, {0.1f, 0.1f, 1.0f}},
};

static Sphere s_Lights[] =
{
    {++g_nextId, {-5.0f, 10.0f, -5.0f}, 1.0f, {1.0f, 0.5f, 0.5f}},
    {++g_nextId, {2.0f, 8.0f, -5.0f}, 1.0f, {0.5f, 1.0f, 0.5f}},
    {++g_nextId, {-1.0f, 6.0f, -2.0f}, 1.0f, {0.5f, 0.5f, 1.0f}}
};

struct RayHitInfo
{
    float time = FLT_MAX;
    Vec3 position;
    Vec3 normal;
    Vec3 color;
    int id;
};

static bool g_initialized = false;

void Initialize()
{
    g_initialized = true;
    for (size_t i = 0; i < sizeof(s_Triangles) / sizeof(s_Triangles[0]); ++i)
    {
        Vec3 AB = s_Triangles[i].B - s_Triangles[i].A;
        Vec3 AC = s_Triangles[i].C - s_Triangles[i].A;
        s_Triangles[i].Normal = Normalize(Cross(AB, AC));
    }
}

inline bool RayIntersects(const Vec3& rayPos, const Vec3& rayDir, const Triangle& triangle, RayHitInfo& info)
{
    // This function adapted from GraphicsCodex.com

    /* If ray P + tw hits triangle V[0], V[1], V[2], then the function returns true,
    stores the barycentric coordinates in b[], and stores the distance to the intersection
    in t. Otherwise returns false and the other output parameters are undefined.*/

    // Edge vectors
    Vec3 e_1 = triangle.B - triangle.A;
    Vec3 e_2 = triangle.C - triangle.A;

    const Vec3& q = Cross(rayDir, e_2);
    const float a = Dot(e_1, q);

    if (abs(a) == 0.0f)
        return false;

    const Vec3 s = (rayPos - triangle.A) / a;
    const Vec3 r = Cross(s, e_1);
    Vec3 b; // b is barycentric coordinates
    b[0] = Dot(s, q);
    b[1] = Dot(r, rayDir);
    b[2] = 1.0f - b[0] - b[1];
    // Intersected outside triangle?
    if ((b[0] < 0.0f) || (b[1] < 0.0f) || (b[2] < 0.0f)) return false;
    float t = Dot(e_2, r);
    if (t < 0.0f)
        return false;

    //enforce a max distance if we should
    if (t > info.time)
        return false;

    // make sure normal is facing opposite of ray direction.
    // this is for if we are hitting the object from the inside / back side.
    Vec3 normal = triangle.Normal;
    if (Dot(triangle.Normal, rayDir) > 0.0f)
        normal *= -1.0f;

    info.time = t;
    info.position = rayPos + rayDir * t;
    info.normal = normal;
    info.color = triangle.color;
    info.id = triangle.id;
    return true;
}

inline bool RayIntersects(const Vec3& rayPos, const Vec3& rayDir, const Sphere& sphere, RayHitInfo& info)
{
    //get the vector from the center of this circle to where the ray begins.
    Vec3 m = rayPos - sphere.position;

    //get the dot product of the above vector and the ray's vector
    float b = Dot(m, rayDir);

    float c = Dot(m, m) - sphere.radius * sphere.radius;

    //exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
    if (c > 0.0 && b > 0.0)
        return false;

    //calculate discriminant
    float discr = b * b - c;

    //a negative discriminant corresponds to ray missing sphere
    if (discr <= 0.0)
        return false;

    //ray now found to intersect sphere, compute smallest t value of intersection
    float collisionTime = -b - sqrt(discr);

    //if t is negative, ray started inside sphere so clamp t to zero and remember that we hit from the inside
    if (collisionTime < 0.0)
        collisionTime = -b + sqrt(discr);

    //enforce a max distance if we should
    if (collisionTime > info.time)
        return false;

    Vec3 normal = Normalize((rayPos + rayDir * collisionTime) - sphere.position);

    // make sure normal is facing opposite of ray direction.
    // this is for if we are hitting the object from the inside / back side.
    if (Dot(normal, rayDir) > 0.0f)
        normal *= -1.0f;

    info.time = collisionTime;
    info.position = rayPos + rayDir * collisionTime;
    info.normal = normal;
    info.color = sphere.color;
    info.id = sphere.id;
    return true;
}

template <bool FIRST_HIT_EXITS>
void RayIntersectScene(const Vec3& rayPos, const Vec3& rayDir, RayHitInfo& info, bool testLights)
{
    info.time = FLT_MAX;

    for (size_t i = 0; i < sizeof(s_Triangles) / sizeof(s_Triangles[0]); ++i)
    {
        RayIntersects(rayPos, rayDir, s_Triangles[i], info);
        if (FIRST_HIT_EXITS && info.time != FLT_MAX)
            return;
    }

    for (size_t i = 0; i < sizeof(s_Spheres) / sizeof(s_Spheres[0]); ++i)
    {
        RayIntersects(rayPos, rayDir, s_Spheres[i], info);
        if (FIRST_HIT_EXITS && info.time != FLT_MAX)
            return;
    }

    if (testLights)
    {
        for (size_t i = 0; i < sizeof(s_Lights) / sizeof(s_Lights[0]); ++i)
        {
            RayIntersects(rayPos, rayDir, s_Lights[i], info);
            if (FIRST_HIT_EXITS && info.time != FLT_MAX)
                return;
        }
    }
}

void SamplePixel(float* pixel, const Vec3& rayPos, const Vec3& rayDir, size_t startSampleCount, size_t endSampleCount, const std::vector<Vec2>& points, const Vec2& rnd)
{
    // TODO: set up a good scene
    // TODO: What is shadow casting geo? a couple spheres and a couple triangles?

    RayHitInfo initialHitInfo;
    RayIntersectScene<false>(rayPos, rayDir, initialHitInfo, false);
    if (initialHitInfo.time == FLT_MAX)
    {
        // TODO: formalize ambient lighting. maybe make it directional.
        pixel[0] = 0.125f;
        pixel[1] = 0.125f;
        pixel[2] = 0.125f;
        return;
    }

    Vec3 shadowRayStart = initialHitInfo.position + initialHitInfo.normal * 0.01f;

    struct PrecalculatedLightData
    {
        Vec3 sw;
        Vec3 su;
        Vec3 sv;
        float cosAMax;
    };
    std::vector<PrecalculatedLightData> precalculatedLightData;
    precalculatedLightData.resize(sizeof(s_Lights) / sizeof(s_Lights[0]));
    for (size_t lightIndex = 0; lightIndex < sizeof(s_Lights) / sizeof(s_Lights[0]); ++lightIndex)
    {
        // create a random direction towards sphere
        // coord system for sampling: sw, su, sv
        precalculatedLightData[lightIndex].sw = Normalize(s_Lights[lightIndex].position - initialHitInfo.position);
        precalculatedLightData[lightIndex].su = Normalize(Cross(abs(precalculatedLightData[lightIndex].sw[0]) > 0.01f ? Vec3{ 0, 1, 0 } : Vec3{ 1, 0, 0 }, precalculatedLightData[lightIndex].sw));
        precalculatedLightData[lightIndex].sv = Cross(precalculatedLightData[lightIndex].sw, precalculatedLightData[lightIndex].su);

        precalculatedLightData[lightIndex].cosAMax = sqrt(1.0f - s_Lights[lightIndex].radius*s_Lights[lightIndex].radius / LengthSQ(initialHitInfo.position - s_Lights[lightIndex].position));
    }

    for (size_t sampleIndex = startSampleCount; sampleIndex < endSampleCount; ++sampleIndex)
    {
        if (sampleIndex >= points.size())
            break;

        float lerpAmount = 1.0f / float(sampleIndex + 1);

        // use the samples passed to us
        // decorrelate with Cranley Patterson Rotation
        float rand1 = std::fmodf(points[sampleIndex][0] + rnd[0], 1.0f);
        float rand2 = std::fmodf(points[sampleIndex][1] + rnd[1], 1.0f);

        // sample each light
        Vec3 sampleResult = { 0.0f, 0.0f, 0.0f };
        for (size_t lightIndex = 0; lightIndex < sizeof(s_Lights) / sizeof(s_Lights[0]); ++lightIndex)
        {
            auto& lightData = precalculatedLightData[lightIndex];

            // sample sphere by solid angle
            float eps1 = rand1;
            float eps2 = rand2;
            float cosA = 1.0f - eps1 + eps1 * lightData.cosAMax;
            float sinA = sqrt(1.0f - cosA * cosA);
            float phi = 2 * c_pi* eps2;
            Vec3 l = lightData.su * cos(phi) * sinA + lightData.sv * sin(phi) * sinA + lightData.sw * cosA;
            l = Normalize(l);

            // raytrace against the scene
            RayHitInfo shadowHitInfo;
            RayIntersectScene<true>(shadowRayStart, l, shadowHitInfo, true);

            if (shadowHitInfo.time == FLT_MAX || shadowHitInfo.id == s_Lights[lightIndex].id)
            {
                float omega = c_pi;// 2 * c_pi * (1 - cosAMax);
                float nDotL = std::max(Dot(initialHitInfo.normal, l), 0.0f);
                sampleResult += initialHitInfo.color * s_Lights[lightIndex].color * nDotL * omega / c_pi;
            }
        }

        // TODO: formalize ambient lighting. maybe make it directional.
        sampleResult += initialHitInfo.color * 0.125f;

        pixel[0] = Lerp(pixel[0], sampleResult[0], lerpAmount);
        pixel[1] = Lerp(pixel[1], sampleResult[1], lerpAmount);
        pixel[2] = Lerp(pixel[2], sampleResult[2], lerpAmount);
    }
}

void RaytraceTest(ImageFloat& image, size_t startSampleCount, size_t endSampleCount, const std::vector<Vec2>& points, std::vector<Vec2>& whiteNoise, bool decorrelate)
{
    if (!g_initialized)
        Initialize();

    const float c_aspectRatio = float(image.m_width) / float(image.m_height);
    const float c_cameraHorizFOV = c_ptCameraVerticalFOV * c_aspectRatio;
    const float c_windowTop = tan(c_ptCameraVerticalFOV / 2.0f) * c_ptNearPlaneDistance;
    const float c_windowRight = tan(c_cameraHorizFOV / 2.0f) * c_ptNearPlaneDistance;
    const Vec3 c_cameraRight = Cross(c_ptCameraUp, c_ptCameraFwd);

    size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.resize(numThreads);

    char prefix[256];
    sprintf_s(prefix, "Raytracing %zu samples with %zu threads: ", (endSampleCount - startSampleCount), numThreads);

    std::atomic<size_t> nextRow(0);
    for (std::thread& t : threads)
    {
        t = std::thread(
            [&]()
            {
                size_t y = nextRow.fetch_add(1);
                bool reportProgress = y == 0;
                int lastPercent = -1;

                while (y < image.m_height)
                {
                    float* pixel = &image.m_pixels[y*image.m_width*4];
                    const Vec2* rnd = &whiteNoise[y*image.m_width];

                    // raytrace every pixel / frequency in this row
                    for (size_t x = 0; x < image.m_width; ++x)
                    {
                        float u = float(x) / float(image.m_width - 1);
                        float v = float(y) / float(image.m_height - 1);

                        // make (u,v) go from [-1,1] instead of [0,1]
                        u = u * 2.0f - 1.0f;
                        v = v * 2.0f - 1.0f;
                        v *= -1.0f;

                        // find where the ray hits the near plane, and normalize that vector to get the ray direction.
                        Vec3 rayPos = c_ptCameraPos + c_ptCameraFwd * c_ptNearPlaneDistance;
                        rayPos += c_cameraRight * c_windowRight * u;
                        rayPos += c_ptCameraUp * c_windowTop * v;
                        Vec3 rayDir = Normalize(rayPos - c_ptCameraPos);

                        if (decorrelate)
                            SamplePixel(pixel, rayPos, rayDir, startSampleCount, endSampleCount, points, *rnd);
                        else
                            SamplePixel(pixel, rayPos, rayDir, startSampleCount, endSampleCount, points, Vec2{ 0.0f, 0.0f });
                        pixel += 4;
                        ++rnd;
                    }

                    // report progress if we should
                    if (reportProgress)
                    {
                        int percent = int(100.0f * float(y) / float(image.m_height));
                        if (lastPercent != percent)
                        {
                            lastPercent = percent;
                            printf("\r%s %i%%", prefix, lastPercent);
                        }
                    }

                    // go to the next row
                    y = nextRow.fetch_add(1);
                }
            }
        );
    }

    for (std::thread& t : threads)
        t.join();
    printf("\r%s 100%%\n", prefix);
}