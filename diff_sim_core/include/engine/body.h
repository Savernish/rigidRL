#ifndef BODY_H
#define BODY_H

#include "engine/tensor.h"
#include "engine/motor.h"
#include <vector>
#include <list>
#include <string>
#include <stdexcept>

// Shape definition supporting multiple primitive types
struct Shape {
    enum Type { BOX, CIRCLE, TRIANGLE };
    Type type;
    
    // BOX: width, height
    // CIRCLE: width = radius (height unused)
    // TRIANGLE: uses vertices array
    float width;
    float height;
    
    // Relative offset from body center (for BOX and CIRCLE)
    float offsetX; 
    float offsetY;
    
    // Triangle vertices (local coordinates, relative to body center)
    // Only used when type == TRIANGLE
    float vertices[6];  // [x1, y1, x2, y2, x3, y3]
    
    // Factory methods for clarity
    static Shape CreateBox(float w, float h, float offX = 0, float offY = 0) {
        Shape s;
        s.type = BOX;
        s.width = w;
        s.height = h;
        s.offsetX = offX;
        s.offsetY = offY;
        return s;
    }
    
    static Shape CreateCircle(float radius, float offX = 0, float offY = 0) {
        Shape s;
        s.type = CIRCLE;
        s.width = radius;
        s.height = 0;
        s.offsetX = offX;
        s.offsetY = offY;
        return s;
    }
    
    static Shape CreateTriangle(float x1, float y1, float x2, float y2, float x3, float y3) {
        Shape s;
        s.type = TRIANGLE;
        s.width = 0;
        s.height = 0;
        s.offsetX = 0;
        s.offsetY = 0;
        s.vertices[0] = x1; s.vertices[1] = y1;
        s.vertices[2] = x2; s.vertices[3] = y2;
        s.vertices[4] = x3; s.vertices[5] = y3;
        return s;
    }
};

struct AABB {
    float minX, minY;
    float maxX, maxY;
};

class Body {
public:
    // State Tensors (Differentiable)
    Tensor pos;      // (2, 1) [x, y]
    Tensor vel;      // (2, 1) [vx, vy]
    Tensor rotation; // (1, 1) [theta]
    Tensor ang_vel;  // (1, 1) [omega] - kept as ang_vel for Python API compatibility

    // Properties (Potentially differentiable!)
    Tensor mass;     // (1, 1)
    Tensor inertia;  // (1, 1)
    
    // Force Accumulators
    Tensor m_ForceAccumulator;  // (2, 1)
    Tensor m_TorqueAccumulator; // (1, 1)
    
    // Geometry (Static for now)
    std::vector<Shape> shapes;
    std::string m_Name;
    
    // Motors attached to this body
    std::vector<Motor*> motors;
    
    // Physics properties
    bool is_static;     // Static bodies don't move (infinite mass for collision)
    float friction;     // Friction coefficient [0, 1]
    float restitution;  // Bounciness [0 = no bounce, 1 = full bounce]

    // Constructor (creates a box shape by default)
    Body(float x, float y, float massVal, float width, float height);
    
    // Static body factory (for ground/walls)
    static Body* CreateStatic(float x, float y, float width, float height, float rotation = 0.0f);
    
    // Shape body factories - cleaner API: Body.Circle, Body.Rect, Body.Triangle
    static Body* Circle(float x, float y, float mass, float radius, 
                        float friction = 0.3f, float restitution = 0.2f) {
        Body* pBody = new Body(x, y, mass, radius * 2, radius * 2);
        pBody->shapes.clear();
        pBody->shapes.push_back(Shape::CreateCircle(radius));
        pBody->friction = friction;
        pBody->restitution = restitution;
        return pBody;
    }
    
    static Body* Rect(float x, float y, float mass, float width, float height,
                      float friction = 0.3f, float restitution = 0.2f) {
        Body* pBody = new Body(x, y, mass, width, height);
        pBody->friction = friction;
        pBody->restitution = restitution;
        return pBody;
    }
    
    static Body* Triangle(float x, float y, float mass, 
                          float x1, float y1, float x2, float y2, float x3, float y3,
                          float friction = 0.3f, float restitution = 0.2f) {
        Body* pBody = new Body(x, y, mass, 1.0f, 1.0f);
        pBody->shapes.clear();
        pBody->shapes.push_back(Shape::CreateTriangle(x1, y1, x2, y2, x3, y3));
        pBody->friction = friction;
        pBody->restitution = restitution;
        return pBody;
    }
    
    // Motor management
    void AddMotor(Motor* pMotor) {
        // Check for overlap with existing motors
        for (Motor* pExisting : motors) {
            if (pExisting->Overlaps(*pMotor)) {
                throw std::runtime_error("Motor overlap detected! Cannot attach motor - it collides with an existing motor.");
            }
        }
        pMotor->parent = this;
        motors.push_back(pMotor);
        
        // Update mass (motor mass adds to body mass)
        float newMass = mass.Get(0, 0) + pMotor->mass;
        mass = Tensor(std::vector<float>{newMass}, true);
        
        // Update inertia (I = I + m*r^2 for point mass at distance r)
        float rSq = pMotor->local_x * pMotor->local_x + pMotor->local_y * pMotor->local_y;
        float newInertia = inertia.Get(0, 0) + pMotor->mass * rSq;
        inertia = Tensor(std::vector<float>{newInertia}, true);
    }
    
    // Apply all motor forces
    void ApplyMotorForces();
    
    // Physics integration step
    // Old method (Manual):
    void Step(const Tensor& forces, const Tensor& torque, float dt);

    // New method (Automatic): Uses accumulators and clears them
    void Step(float dt);

    void ApplyForce(const Tensor& f);
    void ApplyForceAtPoint(const Tensor& force, const Tensor& point);
    void ApplyTorque(const Tensor& t);
    void ResetForces();
    
    // Getters for rendering
    float GetX() const;
    float GetY() const;
    float GetRotation() const;

    std::vector<Tensor> GetCorners();

    // Shape management
    void AddBoxShape(float w, float h, float offX = 0, float offY = 0) {
        shapes.push_back(Shape::CreateBox(w, h, offX, offY));
    }
    
    void AddCircleShape(float radius, float offX = 0, float offY = 0) {
        shapes.push_back(Shape::CreateCircle(radius, offX, offY));
    }
    
    void AddTriangleShape(float x1, float y1, float x2, float y2, float x3, float y3) {
        shapes.push_back(Shape::CreateTriangle(x1, y1, x2, y2, x3, y3));
    }
    
    void ClearShapes() { shapes.clear(); }

    AABB GetAABB() const;

    // Internal memory management for C++ variables to survive autograd
    // Must be std::list to prevent pointer invalidation on push_back!
    std::list<Tensor> garbage_collector; 
    
    // Helper to keep a tensor alive and return a stable reference
    Tensor& Keep(const Tensor& t);
};

#endif // BODY_H
