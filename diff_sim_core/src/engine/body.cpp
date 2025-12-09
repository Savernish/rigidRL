#include "engine/body.h"
#include <iostream>
#include <cmath>

Body::Body(float x, float y, float mass_val, float width, float height) 
    : name("Body"), is_static(false), friction(0.5f), restitution(0.0f)  // No bounce by default
{
    // Initialize State (requires_grad=false by default for state, but can be turned on)
    std::vector<float> pos_vec = {x, y};
    pos = Tensor(pos_vec, true); // Allow gradients to flow through pos
    
    std::vector<float> vel_vec = {0.0f, 0.0f};
    vel = Tensor(vel_vec, true);
    
    std::vector<float> rot_vec = {0.0f};
    rotation = Tensor(rot_vec, true);
    
    std::vector<float> ang_vel_vec = {0.0f};
    ang_vel = Tensor(ang_vel_vec, true);

    // Properties
    std::vector<float> mass_vec = {mass_val};
    mass = Tensor(mass_vec, false); 
    
    // Box inertia: m * (w^2 + h^2) / 12
    float I = mass_val * (width*width + height*height) / 12.0f;
    std::vector<float> I_vec = {I};
    inertia = Tensor(I_vec, false);

    // Shape
    Shape s;
    s.type = Shape::BOX;
    s.width = width;
    s.height = height;
    s.offset_x = 0;
    s.offset_y = 0;
    shapes.push_back(s);

    // Initialize accumulators
    reset_forces();
}

// Factory for static colliders (ground, walls, platforms)
Body* Body::create_static(float x, float y, float width, float height, float rotation_val) {
    // Use mass=1 internally but mark as static  
    Body* b = new Body(x, y, 1.0f, width, height);
    b->is_static = true;
    b->friction = 0.8f;  // Static objects typically have higher friction
    b->restitution = 0.0f;  // No bounce for ground
    
    // Set rotation if specified
    if (rotation_val != 0.0f) {
        std::vector<float> rot_vec = {rotation_val};
        b->rotation = Tensor(rot_vec, false);
    }
    
    return b;
}

void Body::step(const Tensor& forces, const Tensor& torque, float dt) {
    // 1. Linear Acceleration: a = F / m
    std::vector<float> one = {1.0f};
    Tensor inv_mass = Tensor(one, false) / mass; 
    Tensor acc = forces * inv_mass;

    // 2. Angular Acceleration: alpha = tau / I
    Tensor inv_I = Tensor(one, false) / inertia;
    Tensor alpha = torque * inv_I;

    // 3. Integration (Semi-Implicit Euler)
    std::vector<float> dt_vec = {dt};
    Tensor dt_t(dt_vec, false);

    // v_new = v + a * dt
    vel = vel + acc * dt_t;
    
    // pos_new = pos + v_new * dt
    pos = pos + vel * dt_t;

    // omega_new = omega + alpha * dt
    ang_vel = ang_vel + alpha * dt_t;

    // theta_new = theta + omega_new * dt
    rotation = rotation + ang_vel * dt_t;
}

void Body::step(float dt) {
    step(force_accumulator, torque_accumulator, dt);
    reset_forces();
}

void Body::apply_force(const Tensor& f) {
    // If accumulator is zero (no grad), and f has grad, result has grad.
    force_accumulator = force_accumulator + f;
}

void Body::apply_force_at_point(const Tensor& force, const Tensor& point) {
    // 1. Apply Linear Force
    apply_force(force);

    // 2. Calculate Torque = (point - pos) x force
    // r = point - pos
    // Note: 'point' should be world coordinates.

    // Access components (select is now const-qualified)
    Tensor px = pos.select(0); 
    Tensor py = pos.select(1);
    garbage_collector.push_back(px);
    garbage_collector.push_back(py);

    Tensor p_x = point.select(0);
    Tensor p_y = point.select(1);
    garbage_collector.push_back(p_x);
    garbage_collector.push_back(p_y);

    Tensor dx = p_x - px;
    Tensor dy = p_y - py;
    garbage_collector.push_back(dx);
    garbage_collector.push_back(dy);

    Tensor fx = force.select(0);
    Tensor fy = force.select(1);
    garbage_collector.push_back(fx);
    garbage_collector.push_back(fy);

    // Cross product 2D: rx * fy - ry * fx
    Tensor t1 = dx * fy;
    Tensor t2 = dy * fx;
    garbage_collector.push_back(t1);
    garbage_collector.push_back(t2);

    Tensor torque = t1 - t2;
    garbage_collector.push_back(torque);

    apply_torque(torque);
}

void Body::apply_torque(const Tensor& t) {
    torque_accumulator = torque_accumulator + t;
}

void Body::reset_forces() {
    std::vector<float> zero_vec = {0.0f, 0.0f};
    force_accumulator = Tensor(zero_vec, false);
    
    std::vector<float> zero_rot = {0.0f};
    torque_accumulator = Tensor(zero_rot, false);
}

float Body::get_x() const {
    return const_cast<Tensor*>(&pos)->get(0,0);
}

float Body::get_y() const {
    return const_cast<Tensor*>(&pos)->get(1,0); 
    // Wait, pos is (2,1)? x=0, y=1?
    // Body constructor: pos = Tensor({x, y}, true) -> (2,1).
    // So yes.
}

float Body::get_rotation() const {
    return const_cast<Tensor*>(&rotation)->get(0,0);
}

std::vector<Tensor> Body::get_corners() {
    std::vector<Tensor> corners;
    // Clear old graph nodes (assume single step usage)
    garbage_collector.clear();

    float w = shapes[0].width;
    float h = shapes[0].height;
    float hw = w / 2.0f;
    float hh = h / 2.0f;

    struct Point { float x, y; };
    // Corners: TR, TL, BL, BR
    std::vector<Point> offsets = {
        {hw, hh}, {-hw, hh}, {-hw, -hh}, {hw, -hh}
    };

    Tensor cos_t = rotation.cos();
    Tensor sin_t = rotation.sin();
    
    // Save to GC to keep alive
    garbage_collector.push_back(cos_t);
    garbage_collector.push_back(sin_t);

    Tensor px = pos.select(0);
    Tensor py = pos.select(1);
    garbage_collector.push_back(px);
    garbage_collector.push_back(py);

    for (const auto& off : offsets) {
        // rot_x = off.x * cos - off.y * sin
        Tensor rot_x = cos_t * off.x - sin_t * off.y;
        garbage_collector.push_back(rot_x);

        // rot_y = off.x * sin + off.y * cos
        Tensor rot_y = sin_t * off.x + cos_t * off.y;
        garbage_collector.push_back(rot_y);

        Tensor final_x = px + rot_x;
        Tensor final_y = py + rot_y;
        
        // We push copies to GC, and also return copies.
        // The returned copies will have 'children' pointing to GC's Tensors.
        // This is SAFE.
        garbage_collector.push_back(final_x);
        garbage_collector.push_back(final_y);

        corners.push_back(final_x);
        corners.push_back(final_y);
    }
    return corners;
}

AABB Body::get_aabb() const {
    float w = shapes[0].width; // assuming shapes[0] exists
    float h = shapes[0].height;
    // Radius = distance from center to corner
    float radius = std::sqrt(w*w + h*h) / 2.0f;

    AABB aabb;
    float x = get_x();
    float y = get_y();
    
    aabb.min_x = x - radius;
    aabb.max_x = x + radius;
    aabb.min_y = y - radius;
    aabb.max_y = y + radius;
    return aabb;
}

Tensor& Body::keep(const Tensor& t) {
    garbage_collector.push_back(t);
    return garbage_collector.back();
}
