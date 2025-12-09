#ifndef BODY_H
#define BODY_H

#include "engine/tensor.h"
#include <vector>
#include <list>
#include <string>

// Simple shape definition for now
struct Shape {
    enum Type { BOX, CIRCLE };
    Type type;
    float width;  // or radius
    float height;
    // Relative offset from body center
    float offset_x; 
    float offset_y;
};

struct AABB {
    float min_x, min_y;
    float max_x, max_y;

};

class Body {
public:
    // State Tensors (Differentiable)
    Tensor pos;      // (2, 1) [x, y]
    Tensor vel;      // (2, 1) [vx, vy]
    Tensor rotation; // (1, 1) [theta]
    Tensor ang_vel;  // (1, 1) [omega]

    // Properties (Potentially differentiable!)
    Tensor mass;     // (1, 1)
    Tensor inertia;  // (1, 1)
    
    // Force Accumulators
    Tensor force_accumulator; // (2, 1)
    Tensor torque_accumulator; // (1, 1)
    
    // Geometry (Static for now)
    std::vector<Shape> shapes;
    std::string name;
    
    // Physics properties
    bool is_static;     // Static bodies don't move (infinite mass for collision)
    float friction;     // Friction coefficient [0, 1]
    float restitution;  // Bounciness [0 = no bounce, 1 = full bounce]

    Body(float x, float y, float mass_val, float width, float height);
    
    // Static body factory (for ground/walls)
    static Body* create_static(float x, float y, float width, float height, float rotation = 0.0f);
    
    // Physics integration step
    // Old method (Manual):
    void step(const Tensor& forces, const Tensor& torque, float dt);

    // New method (Automatic): Uses accumulators and clears them
    void step(float dt);

    void apply_force(const Tensor& f);
    void apply_force_at_point(const Tensor& force, const Tensor& point);
    void apply_torque(const Tensor& t);
    void reset_forces();
    
    // Getters for rendering
    float get_x() const;
    float get_y() const;
    float get_rotation() const;

    std::vector<Tensor> get_corners();

    AABB get_aabb() const;

    // Internal memory management for C++ variables to survive autograd
    // Must be std::list to prevent pointer invalidation on push_back!
    std::list<Tensor> garbage_collector; 
    
    // Helper to keep a tensor alive and return a stable reference
    Tensor& keep(const Tensor& t);
};

#endif // BODY_H
