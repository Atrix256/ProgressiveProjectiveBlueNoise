#pragma once

#include "image.h"
#include <vector>

typedef std::array<float, 2> Vec2;
typedef std::array<float, 3> Vec3;
typedef std::array<float, 4> Vec4;
typedef std::array<Vec4, 4> Mtx44;

struct SSAOGBufferPixel
{
    float normal[3];
    float tangent[4];
    float depth;
};

typedef std::vector<SSAOGBufferPixel> SSAOGBuffer;

void SSAOTestGetGBuffer(SSAOGBuffer& gbuffer, Mtx44& viewProjMtx, int width, int height);

void SSAOTest(ImageFloat& image, size_t startSampleCount, size_t endSampleCount, const std::vector<Vec2>& points);