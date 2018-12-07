#pragma once

#include "image.h"
#include <vector>

typedef std::array<float, 2> Vec2;

void SSAOTestGetGBuffer(ImageFloat& gbuffer);

void SSAOTest(ImageFloat& image, size_t startSampleCount, size_t endSampleCount, const std::vector<Vec2>& points, std::vector<Vec2>& whiteNoise, bool decorrelate);