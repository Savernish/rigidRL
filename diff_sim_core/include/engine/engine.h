#ifndef ENGINE_H
#define ENGINE_H

#include <vector>
#include "renderer/renderer.h"
#include "engine/body.h"
#include "engine/tensor.h"
#include "engine/contact.h"

class Engine {
private:
    // Core components
    Renderer* m_pRenderer;
    std::vector<Body*> m_Bodies;          // Dynamic bodies
    std::vector<Body*> m_Colliders;       // Static colliders (ground, walls, etc.)
    ContactManager m_ContactManager;       // Sequential impulse solver
    
    // Simulation parameters
    float m_DeltaTime;
    int m_Substeps;
    float m_GravityX;
    float m_GravityY;
    
    // Solver settings
    int m_VelocityIterations = 8;
    int m_PositionIterations = 3;
    
    // Rendering mode
    bool m_bHeadless;

public:
    // Constructor / Destructor
    Engine(int width = 800, int height = 600, float scale = 50.0f, 
           float deltaTime = 0.016f, int substeps = 10, bool headless = false);
    ~Engine();
    
    // Body management
    void AddBody(Body* pBody);
    void ClearBodies();
    
    // Static geometry (ground, walls, platforms)
    Body* AddCollider(float x, float y, float width, float height, 
                      float rotation = 0.0f, float friction = 0.5f);
    void ClearColliders();
    
    // Environment
    void SetGravity(float x, float y);
    
    // Simulation
    void Update();          // Physics step only
    void RenderBodies();    // Render all bodies + colliders
    bool Step();            // Full frame: events + physics + render
    
    // Accessors
    Renderer* GetRenderer() { return m_pRenderer; }
    bool IsHeadless() const { return m_bHeadless; }

private:
    // Physics helpers
    void ApplyGravity(Body* pBody, float subDt);
    void Integrate(Body* pBody, float subDt);
    
    // Collision detection
    bool DetectCollision(Body* pBodyA, Body* pBodyB, ContactManifold& manifold);
    void FindIncidentFace(float* pVertices, Body* pBody, float nx, float ny);
    int ClipSegmentToLine(float* pOut, float* pIn, float nx, float ny, float offset);
    
    // Sequential impulse solver
    void DetectAllCollisions();
    void WarmStart();
    void SolveVelocityConstraints();
    void SolvePositionConstraints();
    void ApplyContactImpulse(ContactManifold& manifold, int pointIndex);
    
    // Legacy collision (kept for compatibility)
    void ResolveCollision(Body* pBodyA, Body* pBodyB);
    bool DetectBoxBox(Body* pBodyA, const Shape& shapeA, Body* pBodyB, const Shape& shapeB,
                      float& penDepth, float& nx, float& ny, float& cx, float& cy);
    int DetectBoxBoxMulti(Body* pBodyA, const Shape& shapeA, Body* pBodyB, const Shape& shapeB,
                          float& penDepth, float& nx, float& ny,
                          float* pContactsX, float* pContactsY, float* pContactsPen);
    void ApplyImpulse(Body* pBodyA, Body* pBodyB, float nx, float ny, float cx, float cy);
    
    // Shape-specific collision detection
    bool DetectCircleCircle(Body* pBodyA, const Shape& shapeA, Body* pBodyB, const Shape& shapeB,
                            float& penDepth, float& nx, float& ny, float& cx, float& cy);
    bool DetectCircleBox(Body* pCircleBody, const Shape& circleShape, Body* pBoxBody, const Shape& boxShape,
                         float& penDepth, float& nx, float& ny, float& cx, float& cy);
};

#endif // ENGINE_H
