#pragma once

#include <vector>
#include <complex>
#include <stdint.h>
#include <thread>
#include <atomic>

typedef uint8_t uint8;

static const float c_pi = 3.14159265359f;

// -------------------------------------------------------------------------------

struct Image
{
    Image(int width, int height)
    {
        m_width = width;
        m_height = height;
        m_pixels.resize(m_width*m_height * 4); // 4 channels per pixel
        std::fill(m_pixels.begin(), m_pixels.end(), 255);
    }

    int m_width;
    int m_height;
    std::vector<uint8> m_pixels;
};

// -------------------------------------------------------------------------------

struct ImageComplex
{
    ImageComplex(int width, int height)
    {
        m_width = width;
        m_height = height;
        m_pixels.resize(m_width*m_height);
        std::fill(m_pixels.begin(), m_pixels.end(), std::complex<float>(0.0f, 0.0f));
    }

    int m_width;
    int m_height;
    std::vector<std::complex<float>> m_pixels;
};

// -------------------------------------------------------------------------------

std::complex<float> DFTPixel (const Image &image, int K, int L)
{
    std::complex<float> ret(0.0f, 0.0f);

    const uint8* pixel = image.m_pixels.data();
    for (int y = 0; y < image.m_height; ++y)
    {
        for (int x = 0; x < image.m_width; ++x)
        {
            float grey = float(pixel[0]) / 255.0f;
            float v = float(K * x) / float(image.m_width);
            v += float(L * y) / float(image.m_height);
            ret += std::complex<float>(grey, 0.0f) * std::polar<float>(1.0f, -2.0f * c_pi * v);
            pixel += 4;
        }
    }

    return ret;
}

void DFTImage (const Image &srcImage, ImageComplex &destImage, bool printProgress)
{
    // calculate 2d dft (brute force, not using fast fourier transform)
    destImage = ImageComplex(srcImage.m_width, srcImage.m_height);
    std::complex<float>* pixel = destImage.m_pixels.data();

    size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.resize(numThreads);
    printf("Doing DFT with %zu threads...\n", numThreads);

    std::atomic<size_t> nextRow(0);
    for (std::thread& t : threads)
    {
        t = std::thread(
            [&]()
            {
                size_t row = nextRow.fetch_add(1);
                bool reportProgress = printProgress && (row == 0);
                int lastPercent = -1;

                while (row < srcImage.m_height)
                {
                    // calculate the DFT for every pixel / frequency in this row
                    for (size_t x = 0; x < srcImage.m_width; ++x)
                    {
                        destImage.m_pixels[row * destImage.m_width + x] = DFTPixel(srcImage, (int)x, (int)row);
                    }

                    // report progress if we should
                    if (reportProgress)
                    {
                        int percent = int(100.0f * float(row) / float(srcImage.m_height));
                        if (lastPercent != percent)
                        {
                            lastPercent = percent;
                            printf("            \rDFT: %i%%", lastPercent);
                        }
                    }

                    // go to the next row
                    row = nextRow.fetch_add(1);
                }
            }
        );
    }

    for (std::thread& t : threads)
        t.join();

    if(printProgress)
        printf("            \rDFT: 100%%\n");
}

void GetMagnitudeData (const ImageComplex& srcImage, Image& destImage)
{
    // size the output image
    destImage = Image(srcImage.m_width, srcImage.m_height);

    // get floating point magnitude data
    std::vector<float> magArray;
    magArray.resize(srcImage.m_width*srcImage.m_height);
    float maxmag = 0.0f;
    for (int x = 0; x < srcImage.m_width; ++x)
    {
        for (int y = 0; y < srcImage.m_height; ++y)
        {
            // Offset the information by half width & height in the positive direction.
            // This makes frequency 0 (DC) be at the image origin, like most diagrams show it.
            int k = (x + srcImage.m_width / 2) % srcImage.m_width;
            int l = (y + srcImage.m_height / 2) % srcImage.m_height;
            const std::complex<float> &src = srcImage.m_pixels[l*srcImage.m_width + k];

            float mag = std::abs(src);
            if (mag > maxmag)
                maxmag = mag;

            magArray[y*srcImage.m_width + x] = mag;
        }
    }
    if (maxmag == 0.0f)
        maxmag = 1.0f;

    const float c = 255.0f / log(1.0f+maxmag);

    // normalize the magnitude data and send it back in [0, 255]
    for (int x = 0; x < srcImage.m_width; ++x)
    {
        for (int y = 0; y < srcImage.m_height; ++y)
        {
            float src = c * log(1.0f + magArray[y*srcImage.m_width + x]);

            uint8 magu8 = uint8(src);

            uint8* dest = &destImage.m_pixels[(y*destImage.m_width + x) * 4];
            dest[0] = magu8;
            dest[1] = magu8;
            dest[2] = magu8;
            dest[3] = 255;
        }
    }
}

// -------------------------------------------------------------------------------
void SaveImage(const char* fileName, Image& image)
{
    image.m_pixels[((image.m_width*image.m_height - 1) * 4) + 3] = 0; // make the last pixel be transparent so eg twitter doesn't use jpg compression.
    stbi_write_png(fileName, image.m_width, image.m_height, 4, image.m_pixels.data(), 0);
}

// -------------------------------------------------------------------------------

float SmoothStep(float value, float min, float max)
{
    float x = (value - min) / (max - min);
    x = std::min(x, 1.0f);
    x = std::max(x, 0.0f);

    return 3.0f * x * x - 2.0f * x * x * x;
}

// -------------------------------------------------------------------------------

template <typename T>
T Lerp(T A, T B, float t)
{
    return T(float(A) * (1.0f - t) + float(B) * t);
}

// -------------------------------------------------------------------------------

void DrawLine(Image& image, int x1, int y1, int x2, int y2, uint8 R, uint8 G, uint8 B)
{
    // pad the AABB of pixels we scan, to account for anti aliasing
    int startX = std::max(std::min(x1, x2) - 4, 0);
    int startY = std::max(std::min(y1, y2) - 4, 0);
    int endX = std::min(std::max(x1, x2) + 4, image.m_width - 1);
    int endY = std::min(std::max(y1, y2) + 4, image.m_height - 1);

    // if (x1,y1) is A and (x2,y2) is B, get a normalized vector from A to B called AB
    float ABX = float(x2 - x1);
    float ABY = float(y2 - y1);
    float ABLen = std::sqrtf(ABX*ABX + ABY * ABY);
    ABX /= ABLen;
    ABY /= ABLen;

    // scan the AABB of our line segment, drawing pixels for the line, as is appropriate
    for (int iy = startY; iy <= endY; ++iy)
    {
        uint8* pixel = &image.m_pixels[(iy * image.m_width + startX) * 4];
        for (int ix = startX; ix <= endX; ++ix)
        {
            // project this current pixel onto the line segment to get the closest point on the line segment to the point
            float ACX = float(ix - x1);
            float ACY = float(iy - y1);
            float lineSegmentT = ACX * ABX + ACY * ABY;
            lineSegmentT = std::min(lineSegmentT, ABLen);
            lineSegmentT = std::max(lineSegmentT, 0.0f);
            float closestX = float(x1) + lineSegmentT * ABX;
            float closestY = float(y1) + lineSegmentT * ABY;

            // calculate the distance from this pixel to the closest point on the line segment
            float distanceX = float(ix) - closestX;
            float distanceY = float(iy) - closestY;
            float distance = std::sqrtf(distanceX*distanceX + distanceY * distanceY);

            // use the distance to figure out how transparent the pixel should be, and apply the color to the pixel
            float alpha = SmoothStep(distance, 2.0f, 0.0f);

            if (alpha > 0.0f)
            {
                pixel[0] = Lerp(pixel[0], R, alpha);
                pixel[1] = Lerp(pixel[1], G, alpha);
                pixel[2] = Lerp(pixel[2], B, alpha);
            }

            pixel += 4;
        }
    }
}

// -------------------------------------------------------------------------------

void ClearImage(Image& image, uint8 R, uint8 G, uint8 B)
{
    uint8* pixel = image.m_pixels.data();
    for (int i = 0, c = image.m_width * image.m_height; i < c; ++i)
    {
        pixel[0] = R;
        pixel[1] = G;
        pixel[2] = B;
        pixel[3] = 255;
        pixel += 4;
    }
}

// -------------------------------------------------------------------------------

void DrawCircle(Image& image, int cx, int cy, int radius, uint8 R, uint8 G, uint8 B)
{
    int startX = std::max(cx - radius - 4, 0);
    int startY = std::max(cy - radius - 4, 0);
    int endX = std::min(cx + radius + 4, image.m_width - 1);
    int endY = std::min(cy + radius + 4, image.m_height - 1);

    for (int iy = startY; iy <= endY; ++iy)
    {
        float dy = float(cy - iy);
        uint8* pixel = &image.m_pixels[(iy * image.m_width + startX) * 4];
        for (int ix = startX; ix <= endX; ++ix)
        {
            float dx = float(cx - ix);

            float distance = std::max(std::sqrtf(dx * dx + dy * dy) - float(radius), 0.0f);

            float alpha = SmoothStep(distance, 2.0f, 0.0f);

            if (alpha > 0.0f)
            {
                pixel[0] = Lerp(pixel[0], R, alpha);
                pixel[1] = Lerp(pixel[1], G, alpha);
                pixel[2] = Lerp(pixel[2], B, alpha);
            }

            pixel += 4;
        }
    }
}