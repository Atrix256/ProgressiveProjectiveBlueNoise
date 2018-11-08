#include "pathtrace.h"
#include <array>

typedef std::array<float, 3> Vec3;

static const Vec3 c_ptCameraPos = { 0.0f, 0.0f, 0.0f };
static const Vec3 c_ptCameraFwd = { 0.0f, 0.0f, 1.0f };
static const Vec3 c_ptCameraUp  = { 0.0f, 1.0f, 0.0f };

static const float c_ptNearPlaneDistance = 0.1f;
static const float c_ptCameraVerticalFOV = 40.0f * c_pi / 180.0f;

struct Sphere
{
    Vec3 position;
    float radius;
};

struct Triangle
{
    Vec3 A;
    Vec3 B;
    Vec3 C;
    Vec3 Normal;
};

static const Sphere c_Spheres[] =
{
    {{0.0f, 0.0f, 10.0f}, 1.0f}
};

static const Triangle c_Triangles[] =
{
    {{0.0f, -2.0f, 0.0f}, {2.0f, -2.0f, 0.0f}, {2.0f, -2.0f, 2.0f}, {}},
};

static const Sphere c_Lights[] =
{
    {{0.0f, 1.0f, 0.0f}, 0.1f}
};

inline Vec3 Normalize(const Vec3& v)
{
    float len = sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
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

inline Vec3 operator * (const Vec3& v, float f)
{
    return
    {
        v[0] * f,
        v[1] * f,
        v[2] * f,
    };
}

inline Vec3 operator += (Vec3& a, const Vec3& b)
{
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
    return a;
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

void SamplePixel(float* pixel, const Vec3& rayPos, const Vec3& rayDir, size_t startSampleCount, size_t endSampleCount)
{
    // TODO: find the initial hit out of the loop since it will be the same each time. Note that we are not doing subpixel jitter, so there will be geo aliasing.
    // TODO: Shade with N dot L directional shadowed lighting 
    // TODO: Use a "point can see point?" query function for each sample to integrate result.
    // TODO: use sample arrays passed in as source for finding location to shoot ray towards
    // TODO: calculate triangle normals on the first call to PathtraceTest(). have a global bool that remembers if the init has happened or not.
    // TODO: multiple lights? or no? If so, shoot a ray to each light each frame to get result to integrate.
    // TODO: What is shadow casting geo? a couple spheres and a couple triangles?
    // TODO: should the light have a brightness?

    for (size_t sampleIndex = startSampleCount; sampleIndex < endSampleCount; ++sampleIndex)
    {
        float lerpAmount = 1.0f / float(startSampleCount + 1);

        float result = 0.0f;

        pixel[0] = Lerp(pixel[0], result, lerpAmount);
        pixel[1] = Lerp(pixel[1], result, lerpAmount);
        pixel[2] = Lerp(pixel[2], result, lerpAmount);
    }
}

void PathtraceTest(ImageFloat& image, size_t startSampleCount, size_t endSampleCount)
{
    const float c_aspectRatio = float(image.m_width) / float(image.m_height);
    const float c_cameraHorizFOV = c_ptCameraVerticalFOV * c_aspectRatio;
    const float c_windowTop = tan(c_ptCameraVerticalFOV / 2.0f) * c_ptNearPlaneDistance;
    const float c_windowRight = tan(c_cameraHorizFOV / 2.0f) * c_ptNearPlaneDistance;
    const Vec3 c_cameraRight = Cross({ 0.0f, 1.0f, 0.0f }, c_ptCameraFwd);
    const Vec3 c_cameraUp = Cross(c_ptCameraFwd, c_cameraRight);

    float* pixel = image.m_pixels.data();

    for (size_t y = 0; y < image.m_height; ++y)
    {
        for (size_t x = 0; x < image.m_width; ++x)
        {
            float u = float(x) / float(image.m_width - 1);
            float v = float(y) / float(image.m_height - 1);

            // make (u,v) go from [-1,1] instead of [0,1]
            u = u * 2.0f - 1.0f;
            v = v * 2.0f - 1.0f;

            // find where the ray hits the near plane, and normalize that vector to get the ray direction.
            Vec3 rayPos = c_ptCameraPos + c_ptCameraFwd * c_ptNearPlaneDistance;
            rayPos += c_cameraRight * c_windowRight * u;
            rayPos += c_cameraUp * c_windowTop * v;
            Vec3 rayDir = Normalize(rayPos - c_ptCameraPos);

            SamplePixel(pixel, rayPos, rayDir, startSampleCount, endSampleCount);
            pixel += 4;
        }
    }
}