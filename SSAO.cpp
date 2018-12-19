#include "SSAO.h"
#include <array>
#include "tiny_obj_loader.h"
#include "cache.h"

static const char* objFileName = "assets/teapot.obj";
//static const char* objFileName = "assets/bunny.obj";
//static const char* objFileName = "assets/dragon.obj";
//static const char* objFileName = "assets/erato.obj";

static SSAOGBuffer g_gbuffer;

// Define a vector as an array of floats
template<size_t N>
using TVector = std::array<float, N>;

// Define a matrix as an array of vectors
template<size_t M, size_t N>
using TMatrix = std::array<TVector<N>, M>;


static float DegreesToRadians(float degrees)
{
    return degrees * c_pi / 180.0f;
}

static float cotangent(float x)
{
    return cosf(x) / sinf(x);
}

float Dot(const Vec4& A, const Vec4& B)
{
    float ret = 0;
    for (int i = 0; i < 4; ++i)
        ret += A[i] * B[i];
    return ret;
}

// Make a specific row have a 1 in the colIndex, and make all other rows have 0 there
template <size_t M, size_t N>
bool MakeRowClaimVariable(TMatrix<M, N>& matrix, size_t rowIndex, size_t colIndex)
{
    // Find a row that has a non zero value in this column and swap it with this row
    {
        // Find a row that has a non zero value
        size_t nonZeroRowIndex = rowIndex;
        while (nonZeroRowIndex < M && matrix[nonZeroRowIndex][colIndex] == 0.0f)
            ++nonZeroRowIndex;

        // If there isn't one, nothing to do
        if (nonZeroRowIndex == M)
            return false;

        // Otherwise, swap the row
        if (rowIndex != nonZeroRowIndex)
            std::swap(matrix[rowIndex], matrix[nonZeroRowIndex]);
    }

    // Scale this row so that it has a leading one
    float scale = 1.0f / matrix[rowIndex][colIndex];
    for (size_t normalizeColIndex = colIndex; normalizeColIndex < N; ++normalizeColIndex)
        matrix[rowIndex][normalizeColIndex] *= scale;

    // Make sure all rows except this one have a zero in this column.
    // Do this by subtracting this row from other rows, multiplied by a multiple that makes the column disappear.
    for (size_t eliminateRowIndex = 0; eliminateRowIndex < M; ++eliminateRowIndex)
    {
        if (eliminateRowIndex == rowIndex)
            continue;

        float scale = matrix[eliminateRowIndex][colIndex];
        for (size_t eliminateColIndex = 0; eliminateColIndex < N; ++eliminateColIndex)
            matrix[eliminateRowIndex][eliminateColIndex] -= matrix[rowIndex][eliminateColIndex] * scale;
    }

    return true;
}

// make matrix into reduced row echelon form
template <size_t M, size_t N>
void GaussJordanElimination(TMatrix<M, N>& matrix)
{
    size_t rowIndex = 0;
    for (size_t colIndex = 0; colIndex < N; ++colIndex)
    {
        if (MakeRowClaimVariable(matrix, rowIndex, colIndex))
        {
            ++rowIndex;
            if (rowIndex == M)
                return;
        }
    }
}

Mtx44 InvertMatrix(const Mtx44& matrix)
{
    // build an augmented matrix
    TMatrix<4, 8> augmentedMatrix;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            // matrix on the left
            augmentedMatrix[i][j] = matrix[i][j];

            // identity on the right
            augmentedMatrix[i][j + 4] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // invert
    GaussJordanElimination(augmentedMatrix);

    // extract the inverted matrix from the right side
    Mtx44 ret;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            ret[i][j] = augmentedMatrix[i][j + 4];
    return ret;
}

static Mtx44 IdentityMatrix()
{
    Mtx44 ret;
    ret[0] = { 1.0f, 0.0f, 0.0f, 0.0f };
    ret[1] = { 0.0f, 1.0f, 0.0f, 0.0f };
    ret[2] = { 0.0f, 0.0f, 1.0f, 0.0f };
    ret[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
    return ret;
}

static Mtx44 Transpose(const Mtx44& mtx)
{
    Mtx44 ret;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            ret[i][j] = mtx[j][i];
    return ret;
}

static Mtx44 ProjectionMatrix(float fovy, float aspectRatio, float znear, float zfar)
{
    float yscale = cotangent(fovy / 2.0f);
    float xscale = yscale / aspectRatio;
    float A = zfar / (zfar - znear);
    float B = -znear * A;

    Mtx44 ret;
    ret[0] = { xscale,   0.0f, 0.0f, 0.0f };
    ret[1] = {   0.0f, yscale, 0.0f, 0.0f };
    ret[2] = {   0.0f,   0.0f,    A,    B };
    ret[3] = {   0.0f,   0.0f, 1.0f, 0.0f };
    return ret;
}

static Mtx44 ScaleMatrix(const Vec3& scale)
{
    Mtx44 ret;
    ret[0] = { scale[0], 0.0f, 0.0f, 0.0f };
    ret[1] = { 0.0f, scale[1], 0.0f, 0.0f };
    ret[2] = { 0.0f, 0.0f, scale[2], 0.0f };
    ret[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
    return ret;
}

static Mtx44 TranslationMatrix(const Vec3& translation)
{
    Mtx44 ret;
    ret[0] = { 1.0f, 0.0f, 0.0f, translation[0] };
    ret[1] = { 0.0f, 1.0f, 0.0f, translation[1] };
    ret[2] = { 0.0f, 0.0f, 1.0f, translation[2] };
    ret[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
    return ret;
}

Vec4 Multiply(const Mtx44& mtx, const Vec4& vec)
{
    Vec4 ret;
    for (int i = 0; i < 4; ++i)
        ret[i] = Dot(mtx[i], vec);
    return ret;
}

Mtx44 Multiply(const Mtx44& A, const Mtx44& B)
{
    Mtx44 BTransposed = Transpose(B);

    Mtx44 ret;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            ret[i][j] = Dot(A[i], BTransposed[j]);
        }
    }
    return ret;
}

Vec3 ProjectPoint(const Mtx44& mtx, const Vec3& p)
{
    Vec4 point = { p[0], p[1], p[2], 1.0f };
    point = Multiply(mtx, point);

    Vec3 ret;
    ret[0] = point[0] / point[3];
    ret[1] = point[1] / point[3];
    ret[2] = point[2] / point[3];
    return ret;
}

static Vec3 RandomUnitVector(float r1, float r2)
{
    float z = r1 * 2.0f - 1.0f;
    float a = r2 * 2.0f * 3.1415926f;
    float r = sqrt(1.0f - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return Vec3{ x, y, z };
}

static inline float LengthSQ(const Vec3& v)
{
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

static inline Vec3 Normalize(const Vec3& v)
{
    float len = sqrtf(LengthSQ(v));
    Vec3 ret;
    ret[0] = v[0] / len;
    ret[1] = v[1] / len;
    ret[2] = v[2] / len;
    return ret;
}

static inline Vec3 Cross(const Vec3& a, const Vec3& b)
{
    return
    {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

static inline float Dot(const Vec3& a, const Vec3& b)
{
    return
        a[0] * b[0] +
        a[1] * b[1] +
        a[2] * b[2];
}

static inline Vec3 operator * (const Vec3& v, float f)
{
    return
    {
        v[0] * f,
        v[1] * f,
        v[2] * f,
    };
}

static inline Vec2 operator * (const Vec2& v, float f)
{
    return
    {
        v[0] * f,
        v[1] * f,
    };
}

static inline Vec3 operator / (const Vec3& v, float f)
{
    return
    {
        v[0] / f,
        v[1] / f,
        v[2] / f,
    };
}

static inline Vec3 operator *= (Vec3& v, float f)
{
    v[0] *= f;
    v[1] *= f;
    v[2] *= f;
    return v;
}

static inline Vec2 operator *= (Vec2& a, const Vec2& b)
{
    a[0] *= b[0];
    a[1] *= b[1];
    return a;
}

static inline Vec3 operator += (Vec3& a, const Vec3& b)
{
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
    return a;
}

static inline Vec3 operator * (const Vec3& a, const Vec3& b)
{
    return
    {
        a[0] * b[0],
        a[1] * b[1],
        a[2] * b[2],
    };
}

static inline Vec3 operator + (const Vec3& a, const Vec3& b)
{
    return
    {
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    };
}

static inline Vec2 operator + (const Vec2& a, float f)
{
    return
    {
        a[0] + f,
        a[1] + f,
    };
}

static inline Vec3 operator - (const Vec3& a, const Vec3& b)
{
    return
    {
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    };
}

static inline Vec2 operator - (const Vec2& a, const Vec2& b)
{
    return
    {
        a[0] - b[0],
        a[1] - b[1],
    };
}

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
    Vec3 Tangent;
};

static int g_nextId = 0;

static std::vector<Triangle> s_Triangles;
static Vec3 s_sceneMin;
static Vec3 s_sceneMax;

struct RayHitInfo
{
    float time = FLT_MAX;
    Vec3 position = { 0.0f, 0.0f, 0.0f };
    Vec3 normal = { 0.0f, 0.0f, 0.0f };
    Vec3 tangent = { 0.0f, 0.0f, 0.0f };
    int id = -1;
};

static bool g_initialized = false;

static Vec3 CalculateTangent(const Vec3& APos, const Vec3& BPos, const Vec3& CPos, const Vec2& AUV, const Vec2& BUV, const Vec2& CUV)
{
    // TODO: c and b were flipped in this code regionally for both position and uv. keep it in mind if tangent is wrong! :P

    Vec3 acPos = CPos - APos;
    Vec3 abPos = BPos - APos;

    Vec2 acUV = CUV - AUV;
    Vec2 abUV = BUV - AUV;

    float f = 1.0f / (abUV[0] * acUV[1] - acUV[0] * abUV[1]);

    Vec3 tangent;
    tangent[0] = f * (acUV[1] * abPos[0] - abUV[1] * acPos[0]);
    tangent[1] = f * (acUV[1] * abPos[1] - abUV[1] * acPos[1]);
    tangent[2] = f * (acUV[1] * abPos[2] - abUV[1] * acPos[2]);
    tangent = Normalize(tangent);

    return tangent;
}

static void Initialize()
{
    g_initialized = true;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;
    bool ret = true;// tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFileName, nullptr, true);

    bool firstVert = true;
    for (const auto& shape : shapes)
    {
        size_t count = shape.mesh.indices.size();

        for (size_t index = 0; index < count; index += 3)
        {
            const auto& indexA = shape.mesh.indices[index + 0];
            const auto& indexB = shape.mesh.indices[index + 1];
            const auto& indexC = shape.mesh.indices[index + 2];

            Vec3& a = *(Vec3*)&attrib.vertices[indexA.vertex_index * 3];
            Vec3& b = *(Vec3*)&attrib.vertices[indexB.vertex_index * 3];
            Vec3& c = *(Vec3*)&attrib.vertices[indexC.vertex_index * 3];

            Vec2& aUV = *(Vec2*)&attrib.texcoords[indexA.texcoord_index * 2];
            Vec2& bUV = *(Vec2*)&attrib.texcoords[indexB.texcoord_index * 2];
            Vec2& cUV = *(Vec2*)&attrib.texcoords[indexC.texcoord_index * 2];

            if (firstVert)
            {
                firstVert = false;
                s_sceneMin = s_sceneMax = a;
            }

            for (int i = 0; i < 3; ++i)
            {
                if (a[i] < s_sceneMin[i])
                    s_sceneMin[i] = a[i];
                if (a[i] > s_sceneMax[i])
                    s_sceneMax[i] = a[i];

                if (b[i] < s_sceneMin[i])
                    s_sceneMin[i] = b[i];
                if (b[i] > s_sceneMax[i])
                    s_sceneMax[i] = b[i];

                if (c[i] < s_sceneMin[i])
                    s_sceneMin[i] = c[i];
                if (c[i] > s_sceneMax[i])
                    s_sceneMax[i] = c[i];
            }

            Triangle triangle;
            triangle.A = a;
            triangle.B = b;
            triangle.C = c;

            Vec3 AB = triangle.B - triangle.A;
            Vec3 AC = triangle.C - triangle.A;

            triangle.Normal = Normalize(Cross(AB, AC));
            triangle.Tangent = CalculateTangent(a, b, c, aUV, bUV, cUV);
            triangle.id = ++g_nextId;

            s_Triangles.push_back(triangle);

            // Do a sort of back face culling for the raytracing, based on camera look direction
            // at < 0.0 it was unreliable. at 0.2 it seems ok but i'm not 100% sure.  With a cached gbuffer we shouldn't screw with the possibility of getting bad data i think.
            //if (Dot(c_ptCameraFwd, triangle.Normal) < 0.2f)
                //s_Triangles.push_back(triangle);
        }
    }

    // center and normalize this so the longest axis is 1.0, and apply an offset for the camera
    Vec3 center = (s_sceneMin + s_sceneMax) / 2.0f;
    float longestRadius = 0.5f * std::max(s_sceneMax[0] - s_sceneMin[0], std::max(s_sceneMax[1] - s_sceneMin[1], s_sceneMax[2] - s_sceneMin[2]));
    longestRadius *= 1.1f; // give some extra padding so things aren't right against the wall
    for (auto& triangle : s_Triangles)
    {
        triangle.A = (triangle.A - center) / longestRadius;
        triangle.B = (triangle.B - center) / longestRadius;
        triangle.C = (triangle.C - center) / longestRadius;
    }

    // add a box that is -1 to 1 on each axis, and the offset is added
    Triangle triangle;

    // left wall
    {
        triangle.A = Vec3{ -1.0f, -1.0f, -1.0f };
        triangle.B = Vec3{ -1.0f, -1.0f,  1.0f };
        triangle.C = Vec3{ -1.0f,  1.0f,  1.0f };
        triangle.Normal = Vec3{ 1.0f, 0.0f, 0.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);

        triangle.A = Vec3{ -1.0f, -1.0f, -1.0f };
        triangle.B = Vec3{ -1.0f,  1.0f,  1.0f };
        triangle.C = Vec3{ -1.0f,  1.0f, -1.0f };
        triangle.Normal = Vec3{ 1.0f, 0.0f, 0.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);
    }

    // right wall
    {
        triangle.A = Vec3{ 1.0f, -1.0f, -1.0f };
        triangle.B = Vec3{ 1.0f, -1.0f,  1.0f };
        triangle.C = Vec3{  1.0f,  1.0f,  1.0f };
        triangle.Normal = Vec3{ -1.0f, 0.0f, 0.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);

        triangle.A = Vec3{ 1.0f, -1.0f, -1.0f };
        triangle.B = Vec3{ 1.0f,  1.0f,  1.0f };
        triangle.C = Vec3{ 1.0f,  1.0f, -1.0f };
        triangle.Normal = Vec3{ -1.0f, 0.0f, 0.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);
    }

    // top wall
    {
        triangle.A = Vec3{-1.0f,  1.0f, -1.0f };
        triangle.B = Vec3{-1.0f,  1.0f,  1.0f };
        triangle.C = Vec3{ 1.0f,  1.0f,  1.0f };
        triangle.Normal = Vec3{ 0.0f, -1.0f, 0.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);

        triangle.A = Vec3{-1.0f,  1.0f, -1.0f };
        triangle.B = Vec3{ 1.0f,  1.0f,  1.0f };
        triangle.C = Vec3{ 1.0f,  1.0f, -1.0f };
        triangle.Normal = Vec3{ 0.0f, -1.0f, 0.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);
    }

    // bottom wall
    {
        triangle.A = Vec3{ -1.0f,  -1.0f, -1.0f };
        triangle.B = Vec3{ -1.0f,  -1.0f,  1.0f };
        triangle.C = Vec3{ 1.0f,  -1.0f,  1.0f };
        triangle.Normal = Vec3{ 0.0f, 1.0f, 0.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);

        triangle.A = Vec3{ -1.0f,  -1.0f, -1.0f };
        triangle.B = Vec3{ 1.0f,  -1.0f,  1.0f };
        triangle.C = Vec3{ 1.0f,  -1.0f, -1.0f };
        triangle.Normal = Vec3{ 0.0f, 1.0f, 0.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);
    }

    // back wall
    {
        triangle.A = Vec3{ -1.0f,  -1.0f, 1.0f };
        triangle.B = Vec3{ -1.0f,  1.0f,  1.0f };
        triangle.C = Vec3{ 1.0f,  1.0f,  1.0f };
        triangle.Normal = Vec3{ 0.0f, 0.0f, -1.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);

        triangle.A = Vec3{ -1.0f, -1.0f, 1.0f };
        triangle.B = Vec3{ 1.0f,  1.0f,  1.0f };
        triangle.C = Vec3{ 1.0f, -1.0f, 1.0f };
        triangle.Normal = Vec3{ 0.0f, 0.0f, -1.0f };
        triangle.id = ++g_nextId;
        s_Triangles.push_back(triangle);
    }

    // set the scene min and max
    s_sceneMin = Vec3{ -1.0f, -1.0f, -1.0f };
    s_sceneMax = Vec3{  1.0f,  1.0f,  1.0f };
}

static inline bool RayIntersectsBox(const Vec3& rayPos, const Vec3& rayDir, const Vec3& min, const Vec3& max)
{
    float tmin = (min[0] - rayPos[0]) / rayDir[0];
    float tmax = (max[0] - rayPos[0]) / rayDir[0];

    if (tmin > tmax) std::swap(tmin, tmax);

    float tymin = (min[1] - rayPos[1]) / rayDir[1];
    float tymax = (max[1] - rayPos[1]) / rayDir[1];

    if (tymin > tymax) std::swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (min[1] - rayPos[2]) / rayDir[2];
    float tzmax = (max[1] - rayPos[2]) / rayDir[2];

    if (tzmin > tzmax) std::swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    return true;
}

static inline bool RayIntersects(const Vec3& rayPos, const Vec3& rayDir, const Triangle& triangle, RayHitInfo& info)
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
    info.tangent = triangle.Tangent;
    info.id = triangle.id;
    return true;
}

static inline bool RayIntersects(const Vec3& rayPos, const Vec3& rayDir, const Sphere& sphere, RayHitInfo& info)
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
    info.id = sphere.id;
    return true;
}

static void RayIntersectScene(const Vec3& rayPos, const Vec3& rayDir, RayHitInfo& info)
{
    if (!RayIntersectsBox(rayPos, rayDir, s_sceneMin, s_sceneMax))
        return;

    info.time = FLT_MAX;
    for (size_t i = 0; i < s_Triangles.size(); ++i)
        RayIntersects(rayPos, rayDir, s_Triangles[i], info);
}

static void SamplePixelGBuffer(SSAOGBufferPixel* pixel, const Vec3& rayPos, const Vec3& rayDir)
{
    RayHitInfo hitInfo;
    hitInfo.normal[0] = 0.0f;
    hitInfo.normal[1] = 0.0f;
    hitInfo.normal[2] = 0.0f;
    hitInfo.tangent[0] = 0.0f;
    hitInfo.tangent[1] = 0.0f;
    hitInfo.tangent[2] = 0.0f;
    RayIntersectScene(rayPos, rayDir, hitInfo);

    for (int i = 0; i < 3; ++i)
    {
        pixel->normal[i] = hitInfo.normal[i];
        pixel->tangent[i] = hitInfo.tangent[i];
    }
    pixel->depth = hitInfo.time;
}

struct MakeGBufferParams
{
    MakeGBufferParams()
    {
        memset(this, 0, sizeof(MakeGBufferParams));
    }

    char objFileName[256];
    int width;
    int height;
    Mtx44 viewProjMtx;
};

inline void GetRayForPixel(const Mtx44& viewProjMtxInv, int x, int y, int width, int height, Vec3& rayPos, Vec3& rayDir)
{
    float u = float(x) / float(width - 1);
    float v = float(y) / float(height - 1);

    // make (u,v) go from [-1,1] instead of [0,1]
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    v *= -1.0f;

    rayPos = ProjectPoint(viewProjMtxInv, { u, v, 0.0f });
    Vec3 rayTarget = ProjectPoint(viewProjMtxInv, { u, v, 1.0f });
    rayDir = Normalize(rayTarget - rayPos);
}

void MakeGBuffer(const MakeGBufferParams& params, std::vector<unsigned char>& buffer)
{
    buffer.resize(sizeof(SSAOGBufferPixel)*params.width*params.width);
    SSAOGBufferPixel* pixels = (SSAOGBufferPixel*)buffer.data();

    const Mtx44& viewProjMtx = params.viewProjMtx;
    const Mtx44 viewProjMtxInv = InvertMatrix(viewProjMtx);

    size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.resize(numThreads);

    std::atomic<int> nextRow(0);
    for (std::thread& t : threads)
    {
        t = std::thread(
            [&]()
        {
            int y = nextRow.fetch_add(1);
            int lastPercent = -1;

            bool reportProgress = (y == 0);

            while (y < params.height)
            {
                if (reportProgress)
                    printf("\rSSAO gbuffer: %i%%", int(100.0f * float(y) / float(params.height)));

                SSAOGBufferPixel* pixel = &pixels[y*params.width];

                // raytrace every pixel / frequency in this row
                for (int x = 0; x < params.width; ++x)
                {
                    Vec3 rayPos;
                    Vec3 rayDir;
                    GetRayForPixel(viewProjMtxInv, x, y, params.width, params.height, rayPos, rayDir);

                    SamplePixelGBuffer(pixel, rayPos, rayDir);
                    pixel++;
                }

                // go to the next row
                y = nextRow.fetch_add(1);
            }

            if (reportProgress)
                printf("\rSSAO gbuffer: 100%%\n");
        }
        );
    }

    for (std::thread& t : threads)
        t.join();
}

void SSAOTestGetGBuffer(SSAOGBuffer& gbuffer, Mtx44& viewProjMtx, int width, int height)
{
    if (!g_initialized)
        Initialize();

    // make the view and projection matrices
    float aspectRatio = float(width) / float(height);
    Mtx44 projMtx = ProjectionMatrix(DegreesToRadians(40.0f), aspectRatio, 0.1f, 100.0f);
    Mtx44 viewMtx = TranslationMatrix({ 0.0f, 0.0f, 5.0f });
    viewProjMtx = Multiply(projMtx, viewMtx);

    // if the gbuffer already exists, copy it
    if (g_gbuffer.size() > 0)
    {
        gbuffer = g_gbuffer;
        return;
    }

    // get the data from the cache, or make it
    MakeGBufferParams params;
    params.width = width;
    params.height = height;
    params.viewProjMtx = viewProjMtx;
    // strcpy_s puts "debug" values in the buffer past the null terminator in debug so makes the hash different. WTF. So, using memcpy instead
    memcpy(params.objFileName, objFileName, strlen(objFileName) + 1); 
    std::vector<unsigned char> buffer;
    MakeDataCached(MakeGBuffer, params, buffer);

    // use the data to make the gbuffer
    gbuffer.resize(width*height);
    memcpy(gbuffer.data(), buffer.data(), buffer.size());

    // store a copy of this off to use again
    g_gbuffer = gbuffer;
}

Vec3 Sample2DToSphereSample(Vec2 sample2D)
{
    // calculate unfiromly distributed polar coordinates 
    float theta = sample2D[0] * c_pi * 2.0f;
    float phi = acos(2.0f * sample2D[1] - 1.0f);

    // TODO: need to do decorrelation in here. There is a thing involving random vectors (white noise). maybe that? sucks adding white noise in though.

    // TODO: get a 3rd random number in [0,1], and cube root it to get the distance
    float radius = 1.0f;

    Vec3 ret;
    ret[0] = radius * cos(theta) * sin(phi);
    ret[1] = radius * sin(theta) * sin(phi);
    ret[2] = radius * cos(phi);
    return ret;
}

void SSAOTest(ImageFloat& image, size_t startSampleCount, size_t endSampleCount, const std::vector<Vec2>& points)
{
    if (!g_initialized)
        Initialize();

    // get the gbuffer
    Mtx44 viewProjMtx, viewProjMtxInv;
    SSAOGBuffer gbuffer;
    SSAOTestGetGBuffer(gbuffer, viewProjMtx, image.m_width, image.m_height);
    viewProjMtxInv = InvertMatrix(viewProjMtx);

    // TODO: pass from caller?
    const float c_AORadius = 0.1f;

    // Do the SSAO
    const SSAOGBufferPixel* gbufferPixel = gbuffer.data();
    float* pixel = image.m_pixels.data();
    for (int y = 0; y < image.m_height; ++y)
    {
        for (int x = 0; x < image.m_width; ++x)
        {
            // skip pixels that were missed by rays
            if (gbufferPixel->depth == FLT_MAX)
            {
                pixel[0] = 0.0f;
                pixel[1] = 0.0f;
                pixel[2] = 0.0f;
                pixel[3] = 1.0f;

                gbufferPixel++;
                pixel += 4;

                continue;
            }

            Vec3 rayPos;
            Vec3 rayDir;
            GetRayForPixel(viewProjMtxInv, x, y, image.m_width, image.m_height, rayPos, rayDir);

            Vec3& gbufferNormal = *(Vec3*)&gbufferPixel->normal;

            Vec3 worldSpacePixelPos = rayPos + rayDir * gbufferPixel->depth;

            for (size_t sampleIndex = startSampleCount; sampleIndex < endSampleCount; ++sampleIndex)
            {
                Vec3 sphereSample = Sample2DToSphereSample(points[sampleIndex]);
                Vec3 hemisphereSample = sphereSample * (Dot(gbufferNormal, sphereSample) >= 0.0f ? 1.0f : -1.0f); // TODO: i don't think this is correct. eg reflecting golden ratio on a circle didn't preserve the proerties!!

                Vec3 worldSpaceSampleOffset = hemisphereSample * c_AORadius;

                Vec3 worldSpaceSamplePosition = worldSpacePixelPos + worldSpaceSampleOffset;

                Vec3 screenSpaceSamplePosition = ProjectPoint(viewProjMtx, worldSpaceSamplePosition);

                // convert xy from [-1,1] to [0,1] uv then to pixel location
                Vec2 pixelLocation = { screenSpaceSamplePosition[0], -screenSpaceSamplePosition[1] };
                pixelLocation = pixelLocation * 0.5f + 0.5f;
                pixelLocation *= Vec2{ float(image.m_width), float(image.m_height) };

                // TODO: should we bilinear interpolate? probably not but ....
                int sampleX = int(pixelLocation[0] + 0.5f);
                int sampleY = int(pixelLocation[1] + 0.5f);

                if (sampleX < 0)
                    sampleX = 0;
                else if (sampleX > image.m_width - 1)
                    sampleX = image.m_width - 1;

                if (sampleY < 0)
                    sampleY = 0;
                else if (sampleY > image.m_height - 1)
                    sampleY = image.m_height - 1;

                float sampleDepth = gbuffer[sampleY * image.m_width + sampleX].depth;

                bool occluded = sampleDepth < gbufferPixel->depth;
                float AOValue = occluded ? 0.0f : 1.0f;

                // TODO: consider occluded if it's occluded and the depth isn't too large. The rest of the stuff from the tutorial too

                float lerpAmount = 1.0f / float(sampleIndex + 1);

                pixel[0] = Lerp(pixel[0], AOValue, lerpAmount);
                pixel[1] = pixel[0];
                pixel[2] = pixel[0];
                pixel[3] = 1.0f;
            }

            gbufferPixel++;
            pixel += 4;
        }
    }

    // TODO: i think we need a full tangent basis to be able to do the normal oriented sampling. need a second gbuffer with tangents i guess. calculate them from uv's?
}