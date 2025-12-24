#include "engine/engine.h"
#include "renderer/sdl_renderer.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>

// ============================================================================
// Constructor / Destructor
// ============================================================================

Engine::Engine(int width, int height, float scale, float deltaTime, int substeps, bool headless)
    : m_pRenderer(nullptr), m_DeltaTime(deltaTime), m_Substeps(substeps),
      m_GravityX(0.0f), m_GravityY(-9.81f), m_bHeadless(headless)
{
    // Only create renderer if not in headless mode
    if (!m_bHeadless) {
        m_pRenderer = new SDLRenderer(width, height, scale);
    }
}

Engine::~Engine() {
    if (m_pRenderer) {
        delete m_pRenderer;
    }
    // Clean up colliders (engine owns them)
    for (Body* pCollider : m_Colliders) {
        delete pCollider;
    }
}

// ============================================================================
// Body and Collider Management
// ============================================================================

void Engine::AddBody(Body* pBody) {
    m_Bodies.push_back(pBody);
}

Body* Engine::AddCollider(float x, float y, float width, float height, float rotation, float friction) {
    Body* pCollider = Body::CreateStatic(x, y, width, height, rotation);
    pCollider->friction = friction;
    m_Colliders.push_back(pCollider);
    return pCollider;
}

void Engine::ClearColliders() {
    for (Body* pCollider : m_Colliders) {
        delete pCollider;
    }
    m_Colliders.clear();
}

void Engine::ClearBodies() {
    // Clear contact manager first
    m_ContactManager.Clear();
    
    // Just clear the vector - don't delete bodies
    // Python owns the Body objects and will garbage collect them
    m_Bodies.clear();
}

void Engine::SetGravity(float x, float y) {
    m_GravityX = x;
    m_GravityY = y;
}

// ============================================================================
// Physics Helpers
// ============================================================================

void Engine::ApplyGravity(Body* pBody, float subDt) {
    if (pBody->is_static) return;
    float mass = pBody->mass.Get(0, 0);
    std::vector<float> gravForce = {m_GravityX * mass, m_GravityY * mass};
    Tensor forceGravity(gravForce, false);
    pBody->ApplyForce(forceGravity);
}

void Engine::Integrate(Body* pBody, float subDt) {
    if (pBody->is_static) return;
    
    // No angular damping - angular momentum conserved in free fall
    // Friction during collisions provides natural damping
    
    pBody->Step(subDt);
}

// ============================================================================
// Collision Detection: Box vs Box (Multi-point SAT-based)
// ============================================================================

// Contact point structure for internal use
struct ContactPointLocal {
    float x, y;      // World position
    float pen;       // Penetration depth
};

// Detect collision between two boxes and return up to 4 contact points
// Returns number of contact points (0 = no collision)
int Engine::DetectBoxBoxMulti(Body* pBodyA, const Shape& shapeA, Body* pBodyB, const Shape& shapeB,
                               float& penDepth, float& nx, float& ny,
                               float* pContactsX, float* pContactsY, float* pContactsPen) {
    // Get transforms
    float ax = pBodyA->pos.Get(0, 0), ay = pBodyA->pos.Get(1, 0);
    float bx = pBodyB->pos.Get(0, 0), by = pBodyB->pos.Get(1, 0);
    float rotA = pBodyA->rotation.Get(0, 0);
    float rotB = pBodyB->rotation.Get(0, 0);
    
    float cosA = std::cos(rotA), sinA = std::sin(rotA);
    float cosB = std::cos(rotB), sinB = std::sin(rotB);
    
    float hwA = shapeA.width / 2.0f, hhA = shapeA.height / 2.0f;
    float hwB = shapeB.width / 2.0f, hhB = shapeB.height / 2.0f;
    
    // Get corners of A in world space
    float cornersA[4][2];
    float localA[4][2] = {{-hwA, -hhA}, {hwA, -hhA}, {hwA, hhA}, {-hwA, hhA}};
    for (int i = 0; i < 4; i++) {
        cornersA[i][0] = ax + cosA * localA[i][0] - sinA * localA[i][1];
        cornersA[i][1] = ay + sinA * localA[i][0] + cosA * localA[i][1];
    }
    
    // Get corners of B in world space
    float cornersB[4][2];
    float localB[4][2] = {{-hwB, -hhB}, {hwB, -hhB}, {hwB, hhB}, {-hwB, hhB}};
    for (int i = 0; i < 4; i++) {
        cornersB[i][0] = bx + cosB * localB[i][0] - sinB * localB[i][1];
        cornersB[i][1] = by + sinB * localB[i][0] + cosB * localB[i][1];
    }
    
    // SAT: check 4 axes (2 per box)
    float axes[4][2] = {
        {cosA, sinA}, {-sinA, cosA},  // A's axes
        {cosB, sinB}, {-sinB, cosB}   // B's axes
    };
    
    penDepth = std::numeric_limits<float>::infinity();
    
    for (int i = 0; i < 4; i++) {
        float axisX = axes[i][0], axisY = axes[i][1];
        
        float minA = std::numeric_limits<float>::infinity();
        float maxA = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; j++) {
            float proj = cornersA[j][0] * axisX + cornersA[j][1] * axisY;
            minA = std::min(minA, proj);
            maxA = std::max(maxA, proj);
        }
        
        float minB = std::numeric_limits<float>::infinity();
        float maxB = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; j++) {
            float proj = cornersB[j][0] * axisX + cornersB[j][1] * axisY;
            minB = std::min(minB, proj);
            maxB = std::max(maxB, proj);
        }
        
        // Check for separation
        if (maxA < minB || maxB < minA) {
            return 0;  // No collision
        }
        
        // Calculate overlap
        float overlap = std::min(maxA, maxB) - std::max(minA, minB);
        if (overlap < penDepth) {
            penDepth = overlap;
            nx = axisX;
            ny = axisY;
            
            // Ensure normal points from B to A
            float dx = ax - bx;
            float dy = ay - by;
            if (dx * nx + dy * ny < 0) {
                nx = -nx;
                ny = -ny;
            }
        }
    }
    
    // Find ALL penetrating corners (up to 4)
    int numContacts = 0;
    
    // Check corners of A inside B
    for (int i = 0; i < 4 && numContacts < 4; i++) {
        float px = cornersA[i][0];
        float py = cornersA[i][1];
        float dx = px - bx;
        float dy = py - by;
        float lx = cosB * dx + sinB * dy;
        float ly = -sinB * dx + cosB * dy;
        
        float penX = hwB - std::abs(lx);
        float penY = hhB - std::abs(ly);
        
        if (penX > 0 && penY > 0) {
            pContactsX[numContacts] = px;
            pContactsY[numContacts] = py;
            pContactsPen[numContacts] = std::min(penX, penY);
            numContacts++;
        }
    }
    
    // Check corners of B inside A
    for (int i = 0; i < 4 && numContacts < 4; i++) {
        float px = cornersB[i][0];
        float py = cornersB[i][1];
        float dx = px - ax;
        float dy = py - ay;
        float lx = cosA * dx + sinA * dy;
        float ly = -sinA * dx + cosA * dy;
        
        float penX = hwA - std::abs(lx);
        float penY = hhA - std::abs(ly);
        
        if (penX > 0 && penY > 0) {
            pContactsX[numContacts] = px;
            pContactsY[numContacts] = py;
            pContactsPen[numContacts] = std::min(penX, penY);
            numContacts++;
        }
    }
    
    // If no corners found, use midpoint as fallback
    if (numContacts == 0) {
        pContactsX[0] = (ax + bx) / 2.0f;
        pContactsY[0] = (ay + by) / 2.0f;
        pContactsPen[0] = penDepth;
        numContacts = 1;
    }
    
    return numContacts;
}

// ============================================================================
// Collision Detection: Box vs Box (SAT-based) - Original single-point version
// ============================================================================

bool Engine::DetectBoxBox(Body* pBodyA, const Shape& shapeA, Body* pBodyB, const Shape& shapeB,
                          float& penDepth, float& nx, float& ny, float& cx, float& cy) {
    // Get transforms
    float ax = pBodyA->pos.Get(0, 0), ay = pBodyA->pos.Get(1, 0);
    float bx = pBodyB->pos.Get(0, 0), by = pBodyB->pos.Get(1, 0);
    float rotA = pBodyA->rotation.Get(0, 0);
    float rotB = pBodyB->rotation.Get(0, 0);
    
    float cosA = std::cos(rotA), sinA = std::sin(rotA);
    float cosB = std::cos(rotB), sinB = std::sin(rotB);
    
    float hwA = shapeA.width / 2.0f, hhA = shapeA.height / 2.0f;
    float hwB = shapeB.width / 2.0f, hhB = shapeB.height / 2.0f;
    
    // Get corners of A in world space
    float cornersA[4][2];
    float localA[4][2] = {{-hwA, -hhA}, {hwA, -hhA}, {hwA, hhA}, {-hwA, hhA}};
    for (int i = 0; i < 4; i++) {
        cornersA[i][0] = ax + cosA * localA[i][0] - sinA * localA[i][1];
        cornersA[i][1] = ay + sinA * localA[i][0] + cosA * localA[i][1];
    }
    
    // Get corners of B in world space
    float cornersB[4][2];
    float localB[4][2] = {{-hwB, -hhB}, {hwB, -hhB}, {hwB, hhB}, {-hwB, hhB}};
    for (int i = 0; i < 4; i++) {
        cornersB[i][0] = bx + cosB * localB[i][0] - sinB * localB[i][1];
        cornersB[i][1] = by + sinB * localB[i][0] + cosB * localB[i][1];
    }
    
    // SAT: check 4 axes (2 per box)
    float axes[4][2] = {
        {cosA, sinA}, {-sinA, cosA},  // A's axes
        {cosB, sinB}, {-sinB, cosB}   // B's axes
    };
    
    penDepth = std::numeric_limits<float>::infinity();
    
    for (int i = 0; i < 4; i++) {
        float axisX = axes[i][0], axisY = axes[i][1];
        
        // Project all corners onto axis
        float minA = std::numeric_limits<float>::infinity();
        float maxA = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; j++) {
            float proj = cornersA[j][0] * axisX + cornersA[j][1] * axisY;
            minA = std::min(minA, proj);
            maxA = std::max(maxA, proj);
        }
        
        float minB = std::numeric_limits<float>::infinity();
        float maxB = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; j++) {
            float proj = cornersB[j][0] * axisX + cornersB[j][1] * axisY;
            minB = std::min(minB, proj);
            maxB = std::max(maxB, proj);
        }
        
        // Check for separation
        if (maxA < minB || maxB < minA) {
            return false;  // Separating axis found, no collision
        }
        
        // Calculate overlap
        float overlap = std::min(maxA, maxB) - std::max(minA, minB);
        
        if (overlap < penDepth) {
            penDepth = overlap;
            nx = axisX;
            ny = axisY;
            
            // Ensure normal points from B to A (standard impulse convention)
            float dx = ax - bx;
            float dy = ay - by;
            if (dx * nx + dy * ny < 0) {
                nx = -nx;
                ny = -ny;
            }
        }
    }
    
    // Find contact point: deepest penetrating corner
    float bestDepth = -std::numeric_limits<float>::infinity();
    cx = (ax + bx) / 2.0f;  // Default fallback
    cy = (ay + by) / 2.0f;
    
    for (int i = 0; i < 4; i++) {
        // Transform A's corner to B's local space
        float px = cornersA[i][0];
        float py = cornersA[i][1];
        float dx = px - bx;
        float dy = py - by;
        float lx = cosB * dx + sinB * dy;
        float ly = -sinB * dx + cosB * dy;
        
        // Check if inside B
        float penX = hwB - std::abs(lx);
        float penY = hhB - std::abs(ly);
        
        if (penX > 0 && penY > 0) {
            float depth = std::min(penX, penY);
            if (depth > bestDepth) {
                bestDepth = depth;
                cx = px;
                cy = py;
            }
        }
    }
    
    // Also check corners of B against A
    for (int i = 0; i < 4; i++) {
        float px = cornersB[i][0];
        float py = cornersB[i][1];
        float dx = px - ax;
        float dy = py - ay;
        float lx = cosA * dx + sinA * dy;
        float ly = -sinA * dx + cosA * dy;
        
        float penX = hwA - std::abs(lx);
        float penY = hhA - std::abs(ly);
        
        if (penX > 0 && penY > 0) {
            float depth = std::min(penX, penY);
            if (depth > bestDepth) {
                bestDepth = depth;
                cx = px;
                cy = py;
            }
        }
    }
    
    return true;
}

// ============================================================================
// Circle Collision Detection
// ============================================================================

bool Engine::DetectCircleCircle(Body* pBodyA, const Shape& shapeA, Body* pBodyB, const Shape& shapeB,
                                float& penDepth, float& nx, float& ny, float& cx, float& cy) {
    // Get circle centers in world space
    float ax = pBodyA->pos.Get(0, 0) + shapeA.offsetX;
    float ay = pBodyA->pos.Get(1, 0) + shapeA.offsetY;
    float bx = pBodyB->pos.Get(0, 0) + shapeB.offsetX;
    float by = pBodyB->pos.Get(1, 0) + shapeB.offsetY;
    
    float radiusA = shapeA.width;  // For circles, width stores radius
    float radiusB = shapeB.width;
    
    // Vector from B to A (so normal points in correct direction for separation)
    float dx = ax - bx;
    float dy = ay - by;
    float distSq = dx * dx + dy * dy;
    float radiusSum = radiusA + radiusB;
    
    // Check if circles overlap
    if (distSq >= radiusSum * radiusSum) {
        return false;  // No collision
    }
    
    float dist = std::sqrt(distSq);
    
    if (dist > 0.0001f) {
        // Normal from B to A (pushes A away from B)
        nx = dx / dist;
        ny = dy / dist;
    } else {
        // Circles at same position - push A upward
        nx = 0.0f;
        ny = 1.0f;
        dist = 0.0f;
    }
    
    penDepth = radiusSum - dist;
    
    // Contact point: midpoint between surfaces
    cx = (ax - nx * radiusA + bx + nx * radiusB) / 2.0f;
    cy = (ay - ny * radiusA + by + ny * radiusB) / 2.0f;
    
    return true;
}

bool Engine::DetectCircleBox(Body* pCircleBody, const Shape& circleShape, Body* pBoxBody, const Shape& boxShape,
                             float& penDepth, float& nx, float& ny, float& cx, float& cy) {
    // Get circle center in world space
    float circleX = pCircleBody->pos.Get(0, 0) + circleShape.offsetX;
    float circleY = pCircleBody->pos.Get(1, 0) + circleShape.offsetY;
    float radius = circleShape.width;
    
    // Get box transform
    float boxX = pBoxBody->pos.Get(0, 0) + boxShape.offsetX;
    float boxY = pBoxBody->pos.Get(1, 0) + boxShape.offsetY;
    float rotation = pBoxBody->rotation.Get(0, 0);
    float hw = boxShape.width / 2.0f;
    float hh = boxShape.height / 2.0f;
    
    float cosR = std::cos(rotation);
    float sinR = std::sin(rotation);
    
    // Transform circle center to box's local space
    float dx = circleX - boxX;
    float dy = circleY - boxY;
    float localX = cosR * dx + sinR * dy;
    float localY = -sinR * dx + cosR * dy;
    
    // Find closest point on box to circle center (in local space)
    float closestX = std::max(-hw, std::min(hw, localX));
    float closestY = std::max(-hh, std::min(hh, localY));
    
    // Vector from closest point to circle center
    float diffX = localX - closestX;
    float diffY = localY - closestY;
    float distSq = diffX * diffX + diffY * diffY;
    
    // Check if circle overlaps
    if (distSq >= radius * radius) {
        return false;  // No collision
    }
    
    float dist = std::sqrt(distSq);
    
    // Calculate collision normal (in local space, then transform to world)
    float localNx, localNy;
    if (dist > 0.0001f) {
        localNx = diffX / dist;
        localNy = diffY / dist;
    } else {
        // Circle center inside box - find nearest edge
        float penLeft = localX + hw;
        float penRight = hw - localX;
        float penBottom = localY + hh;
        float penTop = hh - localY;
        
        float minPen = std::min({penLeft, penRight, penBottom, penTop});
        if (minPen == penLeft) { localNx = -1; localNy = 0; }
        else if (minPen == penRight) { localNx = 1; localNy = 0; }
        else if (minPen == penBottom) { localNx = 0; localNy = -1; }
        else { localNx = 0; localNy = 1; }
        
        dist = 0.0f;
    }
    
    // Transform normal back to world space
    nx = cosR * localNx - sinR * localNy;
    ny = sinR * localNx + cosR * localNy;
    
    penDepth = radius - dist;
    
    // Contact point: closest point on box surface (in world space)
    cx = boxX + cosR * closestX - sinR * closestY;
    cy = boxY + sinR * closestX + cosR * closestY;
    
    return true;
}

// ============================================================================
// Triangle Collision Detection
// ============================================================================

// Helper: Transform triangle vertices to world space
void Engine::GetTriangleWorldVertices(Body* pBody, const Shape& shape, float* outVerts) {
    float rot = pBody->rotation.Get(0, 0);
    float cosR = std::cos(rot);
    float sinR = std::sin(rot);
    float bx = pBody->pos.Get(0, 0);
    float by = pBody->pos.Get(1, 0);
    
    for (int i = 0; i < 3; i++) {
        float lx = shape.vertices[i * 2];
        float ly = shape.vertices[i * 2 + 1];
        outVerts[i * 2]     = bx + cosR * lx - sinR * ly;
        outVerts[i * 2 + 1] = by + sinR * lx + cosR * ly;
    }
}

// Helper: Point to line segment distance (returns closest point)
static float PointToSegmentDist(float px, float py, float ax, float ay, float bx, float by,
                                 float& closestX, float& closestY) {
    float dx = bx - ax;
    float dy = by - ay;
    float lenSq = dx * dx + dy * dy;
    
    if (lenSq < 1e-8f) {
        closestX = ax;
        closestY = ay;
        float diffX = px - ax;
        float diffY = py - ay;
        return std::sqrt(diffX * diffX + diffY * diffY);
    }
    
    float t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
    t = std::max(0.0f, std::min(1.0f, t));
    
    closestX = ax + t * dx;
    closestY = ay + t * dy;
    
    float diffX = px - closestX;
    float diffY = py - closestY;
    return std::sqrt(diffX * diffX + diffY * diffY);
}

// Helper: Check if point is inside triangle using barycentric coords
static bool PointInTriangle(float px, float py, float* verts) {
    float x1 = verts[0], y1 = verts[1];
    float x2 = verts[2], y2 = verts[3];
    float x3 = verts[4], y3 = verts[5];
    
    float d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2);
    float d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3);
    float d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1);
    
    bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    
    return !(hasNeg && hasPos);
}

bool Engine::DetectTriangleCircle(Body* pTriBody, const Shape& triShape, Body* pCircleBody, const Shape& circleShape,
                                   float& penDepth, float& nx, float& ny, float& cx, float& cy) {
    // Get triangle world vertices
    float triVerts[6];
    GetTriangleWorldVertices(pTriBody, triShape, triVerts);
    
    // Get circle center
    float circleX = pCircleBody->pos.Get(0, 0) + circleShape.offsetX;
    float circleY = pCircleBody->pos.Get(1, 0) + circleShape.offsetY;
    float radius = circleShape.width;
    
    // Check if circle center is inside triangle
    if (PointInTriangle(circleX, circleY, triVerts)) {
        // Find closest edge
        float minDist = std::numeric_limits<float>::max();
        int closestEdge = 0;
        float closestPx = 0, closestPy = 0;
        
        for (int i = 0; i < 3; i++) {
            int j = (i + 1) % 3;
            float ax = triVerts[i * 2], ay = triVerts[i * 2 + 1];
            float bx = triVerts[j * 2], by = triVerts[j * 2 + 1];
            
            float cpx, cpy;
            float dist = PointToSegmentDist(circleX, circleY, ax, ay, bx, by, cpx, cpy);
            if (dist < minDist) {
                minDist = dist;
                closestEdge = i;
                closestPx = cpx;
                closestPy = cpy;
            }
        }
        
        // Normal points from triangle to circle (outward)
        float edgeNx = circleX - closestPx;
        float edgeNy = circleY - closestPy;
        float len = std::sqrt(edgeNx * edgeNx + edgeNy * edgeNy);
        if (len > 1e-6f) {
            nx = edgeNx / len;
            ny = edgeNy / len;
        } else {
            // Default normal (up)
            nx = 0; ny = 1;
        }
        
        penDepth = radius + minDist;
        cx = closestPx;
        cy = closestPy;
        return true;
    }
    
    // Circle center outside triangle - check edges
    float minDist = std::numeric_limits<float>::max();
    float closestPx = 0, closestPy = 0;
    
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        float ax = triVerts[i * 2], ay = triVerts[i * 2 + 1];
        float bx = triVerts[j * 2], by = triVerts[j * 2 + 1];
        
        float cpx, cpy;
        float dist = PointToSegmentDist(circleX, circleY, ax, ay, bx, by, cpx, cpy);
        if (dist < minDist) {
            minDist = dist;
            closestPx = cpx;
            closestPy = cpy;
        }
    }
    
    if (minDist > radius) return false;  // No collision
    
    // Normal from closest point to circle center
    float diffX = circleX - closestPx;
    float diffY = circleY - closestPy;
    float len = std::sqrt(diffX * diffX + diffY * diffY);
    
    if (len > 1e-6f) {
        nx = diffX / len;
        ny = diffY / len;
    } else {
        nx = 0; ny = 1;
    }
    
    penDepth = radius - minDist;
    cx = closestPx;
    cy = closestPy;
    return true;
}

bool Engine::DetectTriangleBox(Body* pTriBody, const Shape& triShape, Body* pBoxBody, const Shape& boxShape,
                                float& penDepth, float& nx, float& ny, float& cx, float& cy) {
    // Get triangle world vertices
    float triVerts[6];
    GetTriangleWorldVertices(pTriBody, triShape, triVerts);
    
    // Get box world vertices
    float boxRot = pBoxBody->rotation.Get(0, 0);
    float cosR = std::cos(boxRot), sinR = std::sin(boxRot);
    float bx = pBoxBody->pos.Get(0, 0) + boxShape.offsetX;
    float by = pBoxBody->pos.Get(1, 0) + boxShape.offsetY;
    float hw = boxShape.width / 2.0f, hh = boxShape.height / 2.0f;
    
    float boxVerts[8];
    float localCorners[4][2] = {{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}};
    for (int i = 0; i < 4; i++) {
        boxVerts[i * 2]     = bx + cosR * localCorners[i][0] - sinR * localCorners[i][1];
        boxVerts[i * 2 + 1] = by + sinR * localCorners[i][0] + cosR * localCorners[i][1];
    }
    
    // SAT: Test all potential separating axes
    // Triangle normals (3 edges)
    // Box normals (2 unique due to symmetry, but we test all 4 edges for robustness)
    
    float axes[7][2];
    int numAxes = 0;
    
    // Triangle edge normals
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        float ex = triVerts[j * 2] - triVerts[i * 2];
        float ey = triVerts[j * 2 + 1] - triVerts[i * 2 + 1];
        float len = std::sqrt(ex * ex + ey * ey);
        if (len > 1e-6f) {
            axes[numAxes][0] = -ey / len;  // Perpendicular
            axes[numAxes][1] = ex / len;
            numAxes++;
        }
    }
    
    // Box edge normals (transformed)
    axes[numAxes][0] = cosR;  axes[numAxes][1] = sinR;   numAxes++;
    axes[numAxes][0] = -sinR; axes[numAxes][1] = cosR;   numAxes++;
    
    float minPen = std::numeric_limits<float>::max();
    float bestNx = 0, bestNy = 0;
    
    for (int a = 0; a < numAxes; a++) {
        float ax = axes[a][0], ay = axes[a][1];
        
        // Project triangle
        float triMin = std::numeric_limits<float>::max();
        float triMax = std::numeric_limits<float>::lowest();
        for (int i = 0; i < 3; i++) {
            float proj = triVerts[i * 2] * ax + triVerts[i * 2 + 1] * ay;
            triMin = std::min(triMin, proj);
            triMax = std::max(triMax, proj);
        }
        
        // Project box
        float boxMin = std::numeric_limits<float>::max();
        float boxMax = std::numeric_limits<float>::lowest();
        for (int i = 0; i < 4; i++) {
            float proj = boxVerts[i * 2] * ax + boxVerts[i * 2 + 1] * ay;
            boxMin = std::min(boxMin, proj);
            boxMax = std::max(boxMax, proj);
        }
        
        // Check overlap
        float overlap1 = triMax - boxMin;
        float overlap2 = boxMax - triMin;
        
        if (overlap1 < 0 || overlap2 < 0) return false;  // Separating axis found
        
        float pen = std::min(overlap1, overlap2);
        if (pen < minPen) {
            minPen = pen;
            // Determine normal direction (from BOX to TRIANGLE for proper separation)
            float triCenter = (triMin + triMax) / 2.0f;
            float boxCenter = (boxMin + boxMax) / 2.0f;
            if (triCenter > boxCenter) {
                // Triangle is "ahead" on this axis, normal points towards triangle
                bestNx = ax;
                bestNy = ay;
            } else {
                // Triangle is "behind" on this axis
                bestNx = -ax;
                bestNy = -ay;
            }
        }
    }
    
    // Collision detected
    penDepth = minPen;
    nx = bestNx;
    ny = bestNy;
    
    // Contact point: find triangle vertex with deepest penetration into box
    // This is the vertex that's furthest in the -normal direction relative to box
    float boxCx = bx;
    float boxCy = by;
    
    float bestDot = std::numeric_limits<float>::max();
    cx = triVerts[0];
    cy = triVerts[1];
    
    for (int i = 0; i < 3; i++) {
        // Project vertex onto normal axis relative to box center
        float vx = triVerts[i * 2];
        float vy = triVerts[i * 2 + 1];
        float dot = (vx - boxCx) * nx + (vy - boxCy) * ny;
        if (dot < bestDot) {
            bestDot = dot;
            cx = vx;
            cy = vy;
        }
    }
    
    return true;
}

bool Engine::DetectTriangleTriangle(Body* pBodyA, const Shape& shapeA, Body* pBodyB, const Shape& shapeB,
                                     float& penDepth, float& nx, float& ny, float& cx, float& cy) {
    // Get world vertices for both triangles
    float vertsA[6], vertsB[6];
    GetTriangleWorldVertices(pBodyA, shapeA, vertsA);
    GetTriangleWorldVertices(pBodyB, shapeB, vertsB);
    
    // SAT: Test 6 edge normals (3 from each triangle)
    float axes[6][2];
    int numAxes = 0;
    
    // Triangle A edge normals
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        float ex = vertsA[j * 2] - vertsA[i * 2];
        float ey = vertsA[j * 2 + 1] - vertsA[i * 2 + 1];
        float len = std::sqrt(ex * ex + ey * ey);
        if (len > 1e-6f) {
            axes[numAxes][0] = -ey / len;
            axes[numAxes][1] = ex / len;
            numAxes++;
        }
    }
    
    // Triangle B edge normals
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        float ex = vertsB[j * 2] - vertsB[i * 2];
        float ey = vertsB[j * 2 + 1] - vertsB[i * 2 + 1];
        float len = std::sqrt(ex * ex + ey * ey);
        if (len > 1e-6f) {
            axes[numAxes][0] = -ey / len;
            axes[numAxes][1] = ex / len;
            numAxes++;
        }
    }
    
    float minPen = std::numeric_limits<float>::max();
    float bestNx = 0, bestNy = 0;
    
    for (int a = 0; a < numAxes; a++) {
        float ax = axes[a][0], ay = axes[a][1];
        
        // Project triangle A
        float minA = std::numeric_limits<float>::max();
        float maxA = std::numeric_limits<float>::lowest();
        for (int i = 0; i < 3; i++) {
            float proj = vertsA[i * 2] * ax + vertsA[i * 2 + 1] * ay;
            minA = std::min(minA, proj);
            maxA = std::max(maxA, proj);
        }
        
        // Project triangle B
        float minB = std::numeric_limits<float>::max();
        float maxB = std::numeric_limits<float>::lowest();
        for (int i = 0; i < 3; i++) {
            float proj = vertsB[i * 2] * ax + vertsB[i * 2 + 1] * ay;
            minB = std::min(minB, proj);
            maxB = std::max(maxB, proj);
        }
        
        // Check overlap
        float overlap1 = maxA - minB;
        float overlap2 = maxB - minA;
        
        if (overlap1 < 0 || overlap2 < 0) return false;  // Separating axis
        
        float pen = std::min(overlap1, overlap2);
        if (pen < minPen) {
            minPen = pen;
            // Normal should point from B to A for ApplyImpulse convention
            float centerA = (minA + maxA) / 2.0f;
            float centerB = (minB + maxB) / 2.0f;
            if (centerA > centerB) {
                // A is ahead of B, normal points toward A (positive direction)
                bestNx = ax;
                bestNy = ay;
            } else {
                // B is ahead of A, normal points toward A (negative direction)
                bestNx = -ax;
                bestNy = -ay;
            }
        }
    }
    
    penDepth = minPen;
    nx = bestNx;
    ny = bestNy;
    
    // Contact point: deepest vertex of A into B
    float centBx = (vertsB[0] + vertsB[2] + vertsB[4]) / 3.0f;
    float centBy = (vertsB[1] + vertsB[3] + vertsB[5]) / 3.0f;
    
    float bestDot = std::numeric_limits<float>::max();
    cx = vertsA[0];
    cy = vertsA[1];
    
    for (int i = 0; i < 3; i++) {
        float vx = vertsA[i * 2];
        float vy = vertsA[i * 2 + 1];
        float dot = (vx - centBx) * nx + (vy - centBy) * ny;
        if (dot < bestDot) {
            bestDot = dot;
            cx = vx;
            cy = vy;
        }
    }
    
    return true;
}

// ============================================================================
// Sequential Impulse Solver: Collision Detection with Manifolds
// ============================================================================

// Clip line segment to a line
// Returns number of output points (0, 1, or 2)
int Engine::ClipSegmentToLine(float* pOut, float* pIn, float nx, float ny, float offset) {
    int numOut = 0;
    
    // Distances of end points to the line
    float dist0 = nx * pIn[0] + ny * pIn[1] - offset;
    float dist1 = nx * pIn[2] + ny * pIn[3] - offset;
    
    // If points are behind line, keep them
    if (dist0 <= 0) {
        pOut[numOut * 2] = pIn[0];
        pOut[numOut * 2 + 1] = pIn[1];
        numOut++;
    }
    if (dist1 <= 0) {
        pOut[numOut * 2] = pIn[2];
        pOut[numOut * 2 + 1] = pIn[3];
        numOut++;
    }
    
    // If points are on opposite sides, compute intersection
    if (dist0 * dist1 < 0) {
        float t = dist0 / (dist0 - dist1);
        pOut[numOut * 2] = pIn[0] + t * (pIn[2] - pIn[0]);
        pOut[numOut * 2 + 1] = pIn[1] + t * (pIn[3] - pIn[1]);
        numOut++;
    }
    
    return numOut;
}

// Find the incident face on body B given the reference face normal
void Engine::FindIncidentFace(float* pVertices, Body* pBody, float refNx, float refNy) {
    float bodyRot = pBody->rotation.Get(0, 0);
    float cosB = std::cos(bodyRot), sinB = std::sin(bodyRot);
    float bx = pBody->pos.Get(0, 0), by = pBody->pos.Get(1, 0);
    float hw = pBody->shapes[0].width / 2.0f, hh = pBody->shapes[0].height / 2.0f;
    
    // B's face normals in world space
    float normals[4][2] = {
        {cosB, sinB},       // Right face
        {-sinB, cosB},      // Top face
        {-cosB, -sinB},     // Left face
        {sinB, -cosB}       // Bottom face
    };
    
    // Find face most anti-parallel to reference normal
    int incidentFace = 0;
    float minDot = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 4; ++i) {
        float dot = refNx * normals[i][0] + refNy * normals[i][1];
        if (dot < minDot) {
            minDot = dot;
            incidentFace = i;
        }
    }
    
    // Get the two vertices of the incident face
    float local[4][2] = {{hw, -hh}, {hw, hh}, {-hw, hh}, {-hw, -hh}};
    int v0Idx = incidentFace;
    int v1Idx = (incidentFace + 1) % 4;
    
    pVertices[0] = bx + cosB * local[v0Idx][0] - sinB * local[v0Idx][1];
    pVertices[1] = by + sinB * local[v0Idx][0] + cosB * local[v0Idx][1];
    pVertices[2] = bx + cosB * local[v1Idx][0] - sinB * local[v1Idx][1];
    pVertices[3] = by + sinB * local[v1Idx][0] + cosB * local[v1Idx][1];
}

// Detect collision and populate manifold with contact points
bool Engine::DetectCollision(Body* pBodyA, Body* pBodyB, ContactManifold& manifold) {
    // Get transforms
    float ax = pBodyA->pos.Get(0, 0), ay = pBodyA->pos.Get(1, 0);
    float bx = pBodyB->pos.Get(0, 0), by = pBodyB->pos.Get(1, 0);
    float rotA = pBodyA->rotation.Get(0, 0);
    float rotB = pBodyB->rotation.Get(0, 0);
    
    float cosA = std::cos(rotA), sinA = std::sin(rotA);
    float cosB = std::cos(rotB), sinB = std::sin(rotB);
    
    float hwA = pBodyA->shapes[0].width / 2.0f, hhA = pBodyA->shapes[0].height / 2.0f;
    float hwB = pBodyB->shapes[0].width / 2.0f, hhB = pBodyB->shapes[0].height / 2.0f;
    
    // DEBUG - only when box A is very low (near or below floor)
    static int detectDebug = 0;
    bool bDebugThis = (detectDebug < 20 && ay < 0.8f);
    if (bDebugThis) {
        std::cout << "DetectCollision: A at (" << ax << "," << ay << ") hw=" << hwA << " hh=" << hhA << std::endl;
    }
    
    // Get corners of both boxes in world space
    float localA[4][2] = {{-hwA, -hhA}, {hwA, -hhA}, {hwA, hhA}, {-hwA, hhA}};
    float localB[4][2] = {{-hwB, -hhB}, {hwB, -hhB}, {hwB, hhB}, {-hwB, hhB}};
    
    float cornersA[4][2], cornersB[4][2];
    for (int i = 0; i < 4; ++i) {
        cornersA[i][0] = ax + cosA * localA[i][0] - sinA * localA[i][1];
        cornersA[i][1] = ay + sinA * localA[i][0] + cosA * localA[i][1];
        cornersB[i][0] = bx + cosB * localB[i][0] - sinB * localB[i][1];
        cornersB[i][1] = by + sinB * localB[i][0] + cosB * localB[i][1];
    }
    
    // SAT: check 4 axes (2 per box)
    float axes[4][2] = {
        {cosA, sinA}, {-sinA, cosA},
        {cosB, sinB}, {-sinB, cosB}
    };
    
    float penDepth = std::numeric_limits<float>::infinity();
    float nx = 0, ny = 1;
    int refAxis = 0;
    
    for (int i = 0; i < 4; ++i) {
        float axisX = axes[i][0], axisY = axes[i][1];
        
        float minA = std::numeric_limits<float>::infinity();
        float maxA = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; ++j) {
            float proj = cornersA[j][0] * axisX + cornersA[j][1] * axisY;
            minA = std::min(minA, proj);
            maxA = std::max(maxA, proj);
        }
        
        float minB = std::numeric_limits<float>::infinity();
        float maxB = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; ++j) {
            float proj = cornersB[j][0] * axisX + cornersB[j][1] * axisY;
            minB = std::min(minB, proj);
            maxB = std::max(maxB, proj);
        }
        
        if (maxA < minB || maxB < minA) {
            if (bDebugThis) {
                std::cout << "SAT SEPARATED: axis " << i << " maxA=" << maxA << " minB=" << minB << " maxB=" << maxB << " minA=" << minA << std::endl;
                detectDebug++;
            }
            return false;  // Separating axis found
        }
        
        float overlap = std::min(maxA, maxB) - std::max(minA, minB);
        if (overlap < penDepth) {
            penDepth = overlap;
            nx = axisX;
            ny = axisY;
            refAxis = i;
        }
    }
    
    // Ensure normal points from B to A
    float dx = ax - bx, dy = ay - by;
    if (dx * nx + dy * ny < 0) {
        nx = -nx;
        ny = -ny;
    }
    
    manifold.normal[0] = nx;
    manifold.normal[1] = ny;
    manifold.compute_tangent();
    
    // Determine reference and incident body
    Body* pRefBody = (refAxis < 2) ? pBodyA : pBodyB;
    Body* pIncBody = (refAxis < 2) ? pBodyB : pBodyA;
    
    // Get incident face vertices
    float incidentFace[4];
    FindIncidentFace(incidentFace, pIncBody, nx, ny);
    
    // Reference face: compute side planes
    float refRot = pRefBody->rotation.Get(0, 0);
    float refCos = std::cos(refRot), refSin = std::sin(refRot);
    float refX = pRefBody->pos.Get(0, 0), refY = pRefBody->pos.Get(1, 0);
    float refHw = pRefBody->shapes[0].width / 2.0f;
    float refHh = pRefBody->shapes[0].height / 2.0f;
    
    // Side plane normals (perpendicular to reference normal)
    float sideNx = -ny, sideNy = nx;
    
    // Compute clipping planes
    float refC = nx * refX + ny * refY;  // Reference face offset
    float sideOffset = refHw;  // Side plane offset (approximate)
    
    // Clip incident face against side planes
    float clip1[4], clip2[4];
    int num = ClipSegmentToLine(clip1, incidentFace, sideNx, sideNy, -sideOffset + (sideNx * refX + sideNy * refY));
    if (num < 2) return false;
    
    num = ClipSegmentToLine(clip2, clip1, -sideNx, -sideNy, -sideOffset - (sideNx * refX + sideNy * refY));
    if (num < 2) return false;
    
    // Calculate reference face offset
    float front = nx * refX + ny * refY + ((refAxis % 2 == 0) ? refHw : refHh);
    
    // Keep points below reference face
    manifold.point_count = 0;
    for (int i = 0; i < num && manifold.point_count < MAX_CONTACT_POINTS; ++i) {
        float px = clip2[i * 2];
        float py = clip2[i * 2 + 1];
        float sep = nx * px + ny * py - front;
        
        if (sep <= 0) {
            ContactPoint& cp = manifold.points[manifold.point_count];
            cp.position[0] = px;
            cp.position[1] = py;
            cp.penetration = -sep;
            cp.normal_impulse = 0;
            cp.tangent_impulse = 0;
            cp.feature_id = (refAxis << 8) | manifold.point_count;
            manifold.point_count++;
        }
    }
    
    return manifold.point_count > 0;
}

// ============================================================================
// Collision Response: Impulse-based (Legacy)
// ============================================================================

void Engine::ApplyImpulse(Body* pBodyA, Body* pBodyB, float nx, float ny, float px, float py) {
    // Mass and inertia
    float massA = pBodyA->is_static ? 1e10f : pBodyA->mass.Get(0, 0);
    float massB = pBodyB->is_static ? 1e10f : pBodyB->mass.Get(0, 0);
    float inertiaA = pBodyA->is_static ? 1e10f : pBodyA->inertia.Get(0, 0);
    float inertiaB = pBodyB->is_static ? 1e10f : pBodyB->inertia.Get(0, 0);
    
    float invMassA = pBodyA->is_static ? 0.0f : 1.0f / massA;
    float invMassB = pBodyB->is_static ? 0.0f : 1.0f / massB;
    float invInertiaA = pBodyA->is_static ? 0.0f : 1.0f / inertiaA;
    float invInertiaB = pBodyB->is_static ? 0.0f : 1.0f / inertiaB;
    
    // Positions
    float ax = pBodyA->pos.Get(0, 0), ay = pBodyA->pos.Get(1, 0);
    float bx = pBodyB->pos.Get(0, 0), by = pBodyB->pos.Get(1, 0);
    
    // Vectors from centers to contact point
    float raX = px - ax, raY = py - ay;
    float rbX = px - bx, rbY = py - by;
    
    // Velocities at contact point
    float vaX = pBodyA->vel.Get(0, 0), vaY = pBodyA->vel.Get(1, 0);
    float vbX = pBodyB->vel.Get(0, 0), vbY = pBodyB->vel.Get(1, 0);
    float omegaA = pBodyA->ang_vel.Get(0, 0);
    float omegaB = pBodyB->ang_vel.Get(0, 0);
    
    // Add rotational contribution
    vaX += -omegaA * raY;
    vaY +=  omegaA * raX;
    vbX += -omegaB * rbY;
    vbY +=  omegaB * rbX;
    
    // Relative velocity
    float vRelX = vaX - vbX;
    float vRelY = vaY - vbY;
    float vRelN = vRelX * nx + vRelY * ny;
    
    // Don't resolve if separating
    if (vRelN > 0) return;
    
    // Coefficient of restitution (average)
    float e = (pBodyA->restitution + pBodyB->restitution) / 2.0f;
    
    // Cross products for rotational contribution
    float raCrossN = raX * ny - raY * nx;
    float rbCrossN = rbX * ny - rbY * nx;
    
    // Impulse magnitude
    float denom = invMassA + invMassB + 
                  raCrossN * raCrossN * invInertiaA +
                  rbCrossN * rbCrossN * invInertiaB;
    
    // Safety check to avoid division by zero
    if (denom < 0.0001f) return;
    
    float j = -(1.0f + e) * vRelN / denom;
    
    // Safety check for NaN/Inf
    if (std::isnan(j) || std::isinf(j)) return;
    
    // Apply impulse
    if (!pBodyA->is_static) {
        float newVaX = pBodyA->vel.Get(0, 0) + j * nx * invMassA;
        float newVaY = pBodyA->vel.Get(1, 0) + j * ny * invMassA;
        float newOmegaA = pBodyA->ang_vel.Get(0, 0) + raCrossN * j * invInertiaA;
        
        // Clamp angular velocity to prevent instability
        const float MAX_OMEGA = 3.0f;  // ~170 degrees/sec
        if (newOmegaA > MAX_OMEGA) newOmegaA = MAX_OMEGA;
        if (newOmegaA < -MAX_OMEGA) newOmegaA = -MAX_OMEGA;
        
        std::vector<float> newVelA = {newVaX, newVaY};
        pBodyA->vel = Tensor(newVelA, true);
        std::vector<float> newOmegaAVec = {newOmegaA};
        pBodyA->ang_vel = Tensor(newOmegaAVec, true);
    }
    
    if (!pBodyB->is_static) {
        float newVbX = pBodyB->vel.Get(0, 0) - j * nx * invMassB;
        float newVbY = pBodyB->vel.Get(1, 0) - j * ny * invMassB;
        float newOmegaB = pBodyB->ang_vel.Get(0, 0) - rbCrossN * j * invInertiaB;
        
        std::vector<float> newVelB = {newVbX, newVbY};
        pBodyB->vel = Tensor(newVelB, true);
        std::vector<float> newOmegaBVec = {newOmegaB};
        pBodyB->ang_vel = Tensor(newOmegaBVec, true);
    }
    
    // --- FRICTION ---
    // Recalculate relative velocity after normal impulse
    float vaX2 = pBodyA->vel.Get(0, 0);
    float vaY2 = pBodyA->vel.Get(1, 0);
    float vbX2 = pBodyB->vel.Get(0, 0);
    float vbY2 = pBodyB->vel.Get(1, 0);
    float omegaA2 = pBodyA->ang_vel.Get(0, 0);
    float omegaB2 = pBodyB->ang_vel.Get(0, 0);
    
    vaX2 += -omegaA2 * raY;
    vaY2 +=  omegaA2 * raX;
    vbX2 += -omegaB2 * rbY;
    vbY2 +=  omegaB2 * rbX;
    
    float tx = -ny, ty = nx;  // Tangent
    float vRelT = (vaX2 - vbX2) * tx + (vaY2 - vbY2) * ty;
    
    // Cross products for tangent
    float raCrossT = raX * ty - raY * tx;
    float rbCrossT = rbX * ty - rbY * tx;
    
    float denomT = invMassA + invMassB + 
                   raCrossT * raCrossT * invInertiaA +
                   rbCrossT * rbCrossT * invInertiaB;
    
    float frictionCoef = (pBodyA->friction + pBodyB->friction) / 2.0f;
    float jt = -vRelT / denomT;
    jt = std::max(-frictionCoef * std::abs(j), std::min(frictionCoef * std::abs(j), jt));  // Clamp to Coulomb cone
    
    if (!pBodyA->is_static) {
        float newVaX = pBodyA->vel.Get(0, 0) + jt * tx * invMassA;
        float newVaY = pBodyA->vel.Get(1, 0) + jt * ty * invMassA;
        float newOmegaA = pBodyA->ang_vel.Get(0, 0) + raCrossT * jt * invInertiaA;
        std::vector<float> newVelA = {newVaX, newVaY};
        pBodyA->vel = Tensor(newVelA, true);
        std::vector<float> newOmegaAVec = {newOmegaA};
        pBodyA->ang_vel = Tensor(newOmegaAVec, true);
    }
    
    if (!pBodyB->is_static) {
        float newVbX = pBodyB->vel.Get(0, 0) - jt * tx * invMassB;
        float newVbY = pBodyB->vel.Get(1, 0) - jt * ty * invMassB;
        float newOmegaB = pBodyB->ang_vel.Get(0, 0) - rbCrossT * jt * invInertiaB;
        std::vector<float> newVelB = {newVbX, newVbY};
        pBodyB->vel = Tensor(newVelB, true);
        std::vector<float> newOmegaBVec = {newOmegaB};
        pBodyB->ang_vel = Tensor(newOmegaBVec, true);
    }
}

void Engine::ResolveCollision(Body* pBodyA, Body* pBodyB) {
    for (const Shape& shapeA : pBodyA->shapes) {
        for (const Shape& shapeB : pBodyB->shapes) {
            float pen = 0, nx = 0, ny = 0, cx = 0, cy = 0;
            bool collision = false;
            
            // Dispatch based on shape types
            if (shapeA.type == Shape::BOX && shapeB.type == Shape::BOX) {
                float contactsX[4], contactsY[4], contactsPen[4];
                int numContacts = DetectBoxBoxMulti(pBodyA, shapeA, pBodyB, shapeB, pen, nx, ny,
                                                    contactsX, contactsY, contactsPen);
                if (numContacts > 0) {
                    collision = true;
                    cx = contactsX[0];
                    cy = contactsY[0];
                    
                    // Apply impulse at each contact point
                    for (int i = 0; i < numContacts; ++i) {
                        ApplyImpulse(pBodyA, pBodyB, nx, ny, contactsX[i], contactsY[i]);
                    }
                }
            }
            else if (shapeA.type == Shape::CIRCLE && shapeB.type == Shape::CIRCLE) {
                collision = DetectCircleCircle(pBodyA, shapeA, pBodyB, shapeB, pen, nx, ny, cx, cy);
                if (collision) {
                    ApplyImpulse(pBodyA, pBodyB, nx, ny, cx, cy);
                }
            }
            else if (shapeA.type == Shape::CIRCLE && shapeB.type == Shape::BOX) {
                collision = DetectCircleBox(pBodyA, shapeA, pBodyB, shapeB, pen, nx, ny, cx, cy);
                if (collision) {
                    ApplyImpulse(pBodyA, pBodyB, nx, ny, cx, cy);
                }
            }
            else if (shapeA.type == Shape::BOX && shapeB.type == Shape::CIRCLE) {
                // Swap order: DetectCircleBox expects circle first
                collision = DetectCircleBox(pBodyB, shapeB, pBodyA, shapeA, pen, nx, ny, cx, cy);
                if (collision) {
                    // Normal points from circle to box, flip for consistent impulse
                    nx = -nx;
                    ny = -ny;
                    ApplyImpulse(pBodyA, pBodyB, nx, ny, cx, cy);
                }
            }
            // Triangle vs Circle
            else if (shapeA.type == Shape::TRIANGLE && shapeB.type == Shape::CIRCLE) {
                collision = DetectTriangleCircle(pBodyA, shapeA, pBodyB, shapeB, pen, nx, ny, cx, cy);
                if (collision) {
                    // DetectTriangleCircle returns normal from triangle TO circle
                    // For ApplyImpulse, normal should point from B to A (circle to triangle)
                    // So flip it
                    nx = -nx;
                    ny = -ny;
                    ApplyImpulse(pBodyA, pBodyB, nx, ny, cx, cy);
                }
            }
            else if (shapeA.type == Shape::CIRCLE && shapeB.type == Shape::TRIANGLE) {
                collision = DetectTriangleCircle(pBodyB, shapeB, pBodyA, shapeA, pen, nx, ny, cx, cy);
                if (collision) {
                    // Normal is from triangle (B) to circle (A), which is what we need
                    ApplyImpulse(pBodyA, pBodyB, nx, ny, cx, cy);
                }
            }
            // Triangle vs Box
            else if (shapeA.type == Shape::TRIANGLE && shapeB.type == Shape::BOX) {
                collision = DetectTriangleBox(pBodyA, shapeA, pBodyB, shapeB, pen, nx, ny, cx, cy);
                if (collision) {
                    ApplyImpulse(pBodyA, pBodyB, nx, ny, cx, cy);
                }
            }
            else if (shapeA.type == Shape::BOX && shapeB.type == Shape::TRIANGLE) {
                collision = DetectTriangleBox(pBodyB, shapeB, pBodyA, shapeA, pen, nx, ny, cx, cy);
                if (collision) {
                    nx = -nx;
                    ny = -ny;
                    ApplyImpulse(pBodyA, pBodyB, nx, ny, cx, cy);
                }
            }
            // Triangle vs Triangle
            else if (shapeA.type == Shape::TRIANGLE && shapeB.type == Shape::TRIANGLE) {
                collision = DetectTriangleTriangle(pBodyA, shapeA, pBodyB, shapeB, pen, nx, ny, cx, cy);
                if (collision) {
                    ApplyImpulse(pBodyA, pBodyB, nx, ny, cx, cy);
                }
            }
            
            // Position correction (Baumgarte stabilization)
            if (collision && pen > 0.001f) {
                float slop = 0.01f;
                float baumgarte = 0.4f;
                float correction = std::max(pen - slop, 0.0f) * baumgarte;
                
                if (!pBodyA->is_static && !pBodyB->is_static) {
                    float totalMass = pBodyA->mass.Get(0, 0) + pBodyB->mass.Get(0, 0);
                    float ratioA = pBodyB->mass.Get(0, 0) / totalMass;
                    float ratioB = pBodyA->mass.Get(0, 0) / totalMass;
                    
                    float newAx = pBodyA->pos.Get(0, 0) + nx * correction * ratioA;
                    float newAy = pBodyA->pos.Get(1, 0) + ny * correction * ratioA;
                    pBodyA->pos = Tensor(std::vector<float>{newAx, newAy}, true);
                    
                    float newBx = pBodyB->pos.Get(0, 0) - nx * correction * ratioB;
                    float newBy = pBodyB->pos.Get(1, 0) - ny * correction * ratioB;
                    pBodyB->pos = Tensor(std::vector<float>{newBx, newBy}, true);
                } else if (!pBodyA->is_static) {
                    float newAx = pBodyA->pos.Get(0, 0) + nx * correction;
                    float newAy = pBodyA->pos.Get(1, 0) + ny * correction;
                    pBodyA->pos = Tensor(std::vector<float>{newAx, newAy}, true);
                } else if (!pBodyB->is_static) {
                    float newBx = pBodyB->pos.Get(0, 0) - nx * correction;
                    float newBy = pBodyB->pos.Get(1, 0) - ny * correction;
                    pBodyB->pos = Tensor(std::vector<float>{newBx, newBy}, true);
                }
            }
        }
    }
}

// ============================================================================
// Sequential Impulse Solver: Core Functions
// ============================================================================

void Engine::DetectAllCollisions() {
    m_ContactManager.BeginFrame();
    
    // Dynamic vs Dynamic
    for (size_t i = 0; i < m_Bodies.size(); ++i) {
        for (size_t j = i + 1; j < m_Bodies.size(); ++j) {
            ContactManifold* pManifold = m_ContactManager.GetOrCreate(m_Bodies[i], m_Bodies[j]);
            if (DetectCollision(m_Bodies[i], m_Bodies[j], *pManifold)) {
                pManifold->touching = true;
                pManifold->compute_mass();
            }
        }
    }
    
    // Dynamic vs Static
    static int debugCount = 0;
    for (Body* pBody : m_Bodies) {
        for (Body* pCollider : m_Colliders) {
            ContactManifold* pManifold = m_ContactManager.GetOrCreate(pBody, pCollider);
            if (DetectCollision(pBody, pCollider, *pManifold)) {
                pManifold->touching = true;
                pManifold->compute_mass();
                if (debugCount < 3) {
                    std::cout << "COLLISION DETECTED: " << pManifold->point_count << " points, normal=(" 
                              << pManifold->normal[0] << "," << pManifold->normal[1] << ")" << std::endl;
                    debugCount++;
                }
            }
        }
    }
    
    m_ContactManager.EndFrame();
    
    // Debug: how many manifolds?
    static int frameCount = 0;
    if (frameCount < 5) {
        std::cout << "Active manifolds: " << m_ContactManager.GetManifolds().size() << std::endl;
        frameCount++;
    }
}

void Engine::WarmStart() {
    for (ContactManifold* pManifold : m_ContactManager.GetManifolds()) {
        Body* pBodyA = pManifold->body_a;
        Body* pBodyB = pManifold->body_b;
        
        float invMassA = pBodyA->is_static ? 0.0f : 1.0f / pBodyA->mass.Get(0, 0);
        float invMassB = pBodyB->is_static ? 0.0f : 1.0f / pBodyB->mass.Get(0, 0);
        float invInertiaA = pBodyA->is_static ? 0.0f : 1.0f / pBodyA->inertia.Get(0, 0);
        float invInertiaB = pBodyB->is_static ? 0.0f : 1.0f / pBodyB->inertia.Get(0, 0);
        
        for (int i = 0; i < pManifold->point_count; ++i) {
            ContactPoint& cp = pManifold->points[i];
            
            // Apply cached impulses
            float px = cp.normal_impulse * pManifold->normal[0] + cp.tangent_impulse * pManifold->tangent[0];
            float py = cp.normal_impulse * pManifold->normal[1] + cp.tangent_impulse * pManifold->tangent[1];
            
            float ax = pBodyA->pos.Get(0, 0), ay = pBodyA->pos.Get(1, 0);
            float bx = pBodyB->pos.Get(0, 0), by = pBodyB->pos.Get(1, 0);
            float raX = cp.position[0] - ax, raY = cp.position[1] - ay;
            float rbX = cp.position[0] - bx, rbY = cp.position[1] - by;
            
            if (!pBodyA->is_static) {
                float vaX = pBodyA->vel.Get(0, 0) - px * invMassA;
                float vaY = pBodyA->vel.Get(1, 0) - py * invMassA;
                float omegaA = pBodyA->ang_vel.Get(0, 0) - (raX * py - raY * px) * invInertiaA;
                pBodyA->vel = Tensor(std::vector<float>{vaX, vaY}, true);
                pBodyA->ang_vel = Tensor(std::vector<float>{omegaA}, true);
            }
            if (!pBodyB->is_static) {
                float vbX = pBodyB->vel.Get(0, 0) + px * invMassB;
                float vbY = pBodyB->vel.Get(1, 0) + py * invMassB;
                float omegaB = pBodyB->ang_vel.Get(0, 0) + (rbX * py - rbY * px) * invInertiaB;
                pBodyB->vel = Tensor(std::vector<float>{vbX, vbY}, true);
                pBodyB->ang_vel = Tensor(std::vector<float>{omegaB}, true);
            }
        }
    }
}

void Engine::ApplyContactImpulse(ContactManifold& manifold, int idx) {
    Body* pBodyA = manifold.body_a;
    Body* pBodyB = manifold.body_b;
    ContactPoint& cp = manifold.points[idx];
    
    float invMassA = pBodyA->is_static ? 0.0f : 1.0f / pBodyA->mass.Get(0, 0);
    float invMassB = pBodyB->is_static ? 0.0f : 1.0f / pBodyB->mass.Get(0, 0);
    float invInertiaA = pBodyA->is_static ? 0.0f : 1.0f / pBodyA->inertia.Get(0, 0);
    float invInertiaB = pBodyB->is_static ? 0.0f : 1.0f / pBodyB->inertia.Get(0, 0);
    
    float ax = pBodyA->pos.Get(0, 0), ay = pBodyA->pos.Get(1, 0);
    float bx = pBodyB->pos.Get(0, 0), by = pBodyB->pos.Get(1, 0);
    float raX = cp.position[0] - ax, raY = cp.position[1] - ay;
    float rbX = cp.position[0] - bx, rbY = cp.position[1] - by;
    
    // Compute velocity at contact
    float vaX = pBodyA->vel.Get(0, 0) + (-pBodyA->ang_vel.Get(0, 0) * raY);
    float vaY = pBodyA->vel.Get(1, 0) + ( pBodyA->ang_vel.Get(0, 0) * raX);
    float vbX = pBodyB->vel.Get(0, 0) + (-pBodyB->ang_vel.Get(0, 0) * rbY);
    float vbY = pBodyB->vel.Get(1, 0) + ( pBodyB->ang_vel.Get(0, 0) * rbX);
    
    float vRelX = vaX - vbX;
    float vRelY = vaY - vbY;
    float vRelN = vRelX * manifold.normal[0] + vRelY * manifold.normal[1];
    
    // Normal impulse
    float deltaJ = -vRelN * manifold.normal_mass[idx];
    
    // Clamp accumulated impulse
    float oldImpulse = cp.normal_impulse;
    cp.normal_impulse = std::max(0.0f, oldImpulse + deltaJ);
    deltaJ = cp.normal_impulse - oldImpulse;
    
    float px = deltaJ * manifold.normal[0];
    float py = deltaJ * manifold.normal[1];
    
    if (!pBodyA->is_static) {
        float newVaX = pBodyA->vel.Get(0, 0) + px * invMassA;
        float newVaY = pBodyA->vel.Get(1, 0) + py * invMassA;
        float newOmegaA = pBodyA->ang_vel.Get(0, 0) + (raX * py - raY * px) * invInertiaA;
        pBodyA->vel = Tensor(std::vector<float>{newVaX, newVaY}, true);
        pBodyA->ang_vel = Tensor(std::vector<float>{newOmegaA}, true);
    }
    if (!pBodyB->is_static) {
        float newVbX = pBodyB->vel.Get(0, 0) - px * invMassB;
        float newVbY = pBodyB->vel.Get(1, 0) - py * invMassB;
        float newOmegaB = pBodyB->ang_vel.Get(0, 0) - (rbX * py - rbY * px) * invInertiaB;
        pBodyB->vel = Tensor(std::vector<float>{newVbX, newVbY}, true);
        pBodyB->ang_vel = Tensor(std::vector<float>{newOmegaB}, true);
    }
    
    // Friction impulse
    float vRelT = vRelX * manifold.tangent[0] + vRelY * manifold.tangent[1];
    float deltaJt = -vRelT * manifold.tangent_mass[idx];
    
    // Clamp friction by Coulomb's law
    float maxFriction = manifold.friction * cp.normal_impulse;
    float oldTangent = cp.tangent_impulse;
    cp.tangent_impulse = std::max(-maxFriction, std::min(oldTangent + deltaJt, maxFriction));
    deltaJt = cp.tangent_impulse - oldTangent;
    
    float tx = deltaJt * manifold.tangent[0];
    float ty = deltaJt * manifold.tangent[1];
    
    if (!pBodyA->is_static) {
        float newVaX = pBodyA->vel.Get(0, 0) + tx * invMassA;
        float newVaY = pBodyA->vel.Get(1, 0) + ty * invMassA;
        float newOmegaA = pBodyA->ang_vel.Get(0, 0) + (raX * ty - raY * tx) * invInertiaA;
        pBodyA->vel = Tensor(std::vector<float>{newVaX, newVaY}, true);
        pBodyA->ang_vel = Tensor(std::vector<float>{newOmegaA}, true);
    }
    if (!pBodyB->is_static) {
        float newVbX = pBodyB->vel.Get(0, 0) - tx * invMassB;
        float newVbY = pBodyB->vel.Get(1, 0) - ty * invMassB;
        float newOmegaB = pBodyB->ang_vel.Get(0, 0) - (rbX * ty - rbY * tx) * invInertiaB;
        pBodyB->vel = Tensor(std::vector<float>{newVbX, newVbY}, true);
        pBodyB->ang_vel = Tensor(std::vector<float>{newOmegaB}, true);
    }
}

void Engine::SolveVelocityConstraints() {
    for (ContactManifold* pManifold : m_ContactManager.GetManifolds()) {
        for (int i = 0; i < pManifold->point_count; ++i) {
            ApplyContactImpulse(*pManifold, i);
        }
    }
}

void Engine::SolvePositionConstraints() {
    for (ContactManifold* pManifold : m_ContactManager.GetManifolds()) {
        Body* pBodyA = pManifold->body_a;
        Body* pBodyB = pManifold->body_b;
        
        for (int i = 0; i < pManifold->point_count; ++i) {
            ContactPoint& cp = pManifold->points[i];
            
            if (cp.penetration <= 0.001f) continue;  // Slop
            
            float correction = (cp.penetration - 0.001f) * 0.2f;  // Baumgarte
            
            if (!pBodyA->is_static && !pBodyB->is_static) {
                float ax = pBodyA->pos.Get(0, 0) + pManifold->normal[0] * correction * 0.5f;
                float ay = pBodyA->pos.Get(1, 0) + pManifold->normal[1] * correction * 0.5f;
                pBodyA->pos = Tensor(std::vector<float>{ax, ay}, true);
                float bx = pBodyB->pos.Get(0, 0) - pManifold->normal[0] * correction * 0.5f;
                float by = pBodyB->pos.Get(1, 0) - pManifold->normal[1] * correction * 0.5f;
                pBodyB->pos = Tensor(std::vector<float>{bx, by}, true);
            } else if (!pBodyA->is_static) {
                float ax = pBodyA->pos.Get(0, 0) + pManifold->normal[0] * correction;
                float ay = pBodyA->pos.Get(1, 0) + pManifold->normal[1] * correction;
                pBodyA->pos = Tensor(std::vector<float>{ax, ay}, true);
            } else if (!pBodyB->is_static) {
                float bx = pBodyB->pos.Get(0, 0) - pManifold->normal[0] * correction;
                float by = pBodyB->pos.Get(1, 0) - pManifold->normal[1] * correction;
                pBodyB->pos = Tensor(std::vector<float>{bx, by}, true);
            }
        }
    }
}

// ============================================================================
// Main Update Loop
// ============================================================================

void Engine::Update() {
    float subDt = m_DeltaTime / static_cast<float>(m_Substeps);
    
    for (int step = 0; step < m_Substeps; ++step) {
        // 0. Apply motor forces
        for (Body* pBody : m_Bodies) {
            pBody->ApplyMotorForces();
        }
        
        // 1. Apply gravity
        for (Body* pBody : m_Bodies) {
            ApplyGravity(pBody, subDt);
        }
        
        // 2. Integrate positions and velocities
        for (Body* pBody : m_Bodies) {
            Integrate(pBody, subDt);
        }
        
        // 3. Collision detection and response using OLD working system
        // Dynamic vs Dynamic
        for (size_t i = 0; i < m_Bodies.size(); ++i) {
            for (size_t j = i + 1; j < m_Bodies.size(); ++j) {
                ResolveCollision(m_Bodies[i], m_Bodies[j]);
            }
        }
        
        // Dynamic vs Static (colliders)
        for (Body* pBody : m_Bodies) {
            for (Body* pCollider : m_Colliders) {
                ResolveCollision(pBody, pCollider);
            }
        }
    }
    
    // Clear garbage collectors
    for (Body* pBody : m_Bodies) {
        pBody->garbage_collector.clear();
    }
}

// ============================================================================
// Rendering
// ============================================================================

void Engine::RenderBodies() {
    // Skip rendering in headless mode
    if (m_bHeadless || !m_pRenderer) return;
    
    // Helper lambda to transform local point to world coordinates
    auto transformPoint = [](float localX, float localY, float bodyX, float bodyY, float rotation, float& worldX, float& worldY) {
        float cosR = std::cos(rotation);
        float sinR = std::sin(rotation);
        worldX = bodyX + localX * cosR - localY * sinR;
        worldY = bodyY + localX * sinR + localY * cosR;
    };
    
    // Render colliders (static geometry) in gray - filled
    for (Body* pCollider : m_Colliders) {
        float x = pCollider->GetX();
        float y = pCollider->GetY();
        float rot = pCollider->GetRotation();
        
        for (const auto& shape : pCollider->shapes) {
            if (shape.type == Shape::BOX) {
                m_pRenderer->DrawBoxFilled(x + shape.offsetX, y + shape.offsetY, 
                                   shape.width, shape.height, rot,
                                   0.4f, 0.4f, 0.4f);
            } else if (shape.type == Shape::CIRCLE) {
                float wx, wy;
                transformPoint(shape.offsetX, shape.offsetY, x, y, rot, wx, wy);
                m_pRenderer->DrawCircleFilled(wx, wy, shape.width, 0.4f, 0.4f, 0.4f);
            } else if (shape.type == Shape::TRIANGLE) {
                float wx1, wy1, wx2, wy2, wx3, wy3;
                transformPoint(shape.vertices[0], shape.vertices[1], x, y, rot, wx1, wy1);
                transformPoint(shape.vertices[2], shape.vertices[3], x, y, rot, wx2, wy2);
                transformPoint(shape.vertices[4], shape.vertices[5], x, y, rot, wx3, wy3);
                m_pRenderer->DrawTriangleFilled(wx1, wy1, wx2, wy2, wx3, wy3, 0.4f, 0.4f, 0.4f);
            }
        }
    }
    
    // Render dynamic bodies - filled with color
    for (Body* pBody : m_Bodies) {
        float x = pBody->GetX();
        float y = pBody->GetY();
        float rot = pBody->GetRotation();
        
        for (const auto& shape : pBody->shapes) {
            if (shape.type == Shape::BOX) {
                m_pRenderer->DrawBoxFilled(x + shape.offsetX, y + shape.offsetY, 
                                   shape.width, shape.height, rot,
                                   0.3f, 0.5f, 0.8f);  // Blue fill
                m_pRenderer->DrawBox(x + shape.offsetX, y + shape.offsetY, 
                                   shape.width, shape.height, rot,
                                   1.0f, 1.0f, 1.0f);  // White outline
            } else if (shape.type == Shape::CIRCLE) {
                float wx, wy;
                transformPoint(shape.offsetX, shape.offsetY, x, y, rot, wx, wy);
                float radius = shape.width;
                m_pRenderer->DrawCircleFilled(wx, wy, radius, 0.3f, 0.5f, 0.8f);  // Blue fill
                m_pRenderer->DrawCircle(wx, wy, radius, 1.0f, 1.0f, 1.0f);  // White outline
                // Rotation indicator line (white on top)
                float lineStartX = wx - radius * std::cos(rot);
                float lineStartY = wy - radius * std::sin(rot);
                float lineEndX = wx + radius * std::cos(rot);
                float lineEndY = wy + radius * std::sin(rot);
                m_pRenderer->DrawLine(lineStartX, lineStartY, lineEndX, lineEndY, 1.0f, 1.0f, 1.0f);
            } else if (shape.type == Shape::TRIANGLE) {
                float wx1, wy1, wx2, wy2, wx3, wy3;
                transformPoint(shape.vertices[0], shape.vertices[1], x, y, rot, wx1, wy1);
                transformPoint(shape.vertices[2], shape.vertices[3], x, y, rot, wx2, wy2);
                transformPoint(shape.vertices[4], shape.vertices[5], x, y, rot, wx3, wy3);
                m_pRenderer->DrawTriangleFilled(wx1, wy1, wx2, wy2, wx3, wy3, 0.3f, 0.5f, 0.8f);  // Blue
                m_pRenderer->DrawTriangle(wx1, wy1, wx2, wy2, wx3, wy3, 1.0f, 1.0f, 1.0f);  // White outline
            }
        }
    }
}

bool Engine::Step() {
    // In headless mode, just run physics (no events, no render, no frame limiting)
    if (m_bHeadless) {
        Update();
        return true;
    }
    
    // Normal mode: full frame with rendering
    if (!m_pRenderer->ProcessEvents()) {
        return false;
    }
    
    // Fixed timestep accumulator for consistent physics
    // Physics runs at 60Hz (m_DeltaTime), display at 144fps
    static auto lastTime = std::chrono::high_resolution_clock::now();
    static float accumulator = 0.0f;
    
    auto currentTime = std::chrono::high_resolution_clock::now();
    float frameTime = std::chrono::duration<float>(currentTime - lastTime).count();
    lastTime = currentTime;
    
    // Clamp frame time to prevent spiral of death
    if (frameTime > 0.25f) frameTime = 0.25f;
    
    accumulator += frameTime;
    
    // Run physics at fixed timestep
    while (accumulator >= m_DeltaTime) {
        Update();
        accumulator -= m_DeltaTime;
    }
    
    // Render
    m_pRenderer->Clear();
    RenderBodies();
    m_pRenderer->Present();
    
    // Frame rate limiting: 144fps = 6944 microseconds per frame
    constexpr int TARGET_FRAME_TIME_US = 6944;  // 1000000 / 144
    std::this_thread::sleep_for(std::chrono::microseconds(TARGET_FRAME_TIME_US));
    return true;
}
