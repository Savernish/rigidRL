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

Engine::Engine(int width, int height, float scale, float dt_val, int substeps_val)
    : renderer(nullptr), dt(dt_val), substeps(substeps_val),
      gravity_x(0.0f), gravity_y(-9.81f)
{
    renderer = new SDLRenderer(width, height, scale);
}

Engine::~Engine() {
    delete renderer;
    // Clean up colliders (engine owns them)
    for (Body* c : colliders) {
        delete c;
    }
}

// ============================================================================
// Body and Collider Management
// ============================================================================

void Engine::add_body(Body* b) {
    bodies.push_back(b);
}

Body* Engine::add_collider(float x, float y, float width, float height, float rotation, float friction) {
    Body* c = Body::create_static(x, y, width, height, rotation);
    c->friction = friction;
    colliders.push_back(c);
    return c;
}

void Engine::clear_colliders() {
    for (Body* c : colliders) {
        delete c;
    }
    colliders.clear();
}

void Engine::clear_bodies() {
    // Clear contact manager first
    contact_manager.clear();
    
    // Just clear the vector - don't delete bodies
    // Python owns the Body objects and will garbage collect them
    bodies.clear();
}

void Engine::set_gravity(float x, float y) {
    gravity_x = x;
    gravity_y = y;
}

// ============================================================================
// Physics Helpers
// ============================================================================

void Engine::apply_gravity(Body* b, float sub_dt) {
    if (b->is_static) return;
    float m = b->mass.get(0, 0);
    std::vector<float> grav = {gravity_x * m, gravity_y * m};
    Tensor force_gravity(grav, false);
    b->apply_force(force_gravity);
}

void Engine::integrate(Body* b, float sub_dt) {
    if (b->is_static) return;
    
    // Angular velocity damping - only for bodies WITHOUT motors
    // Bodies with motors should naturally stabilize via thrust control
    if (b->motors.empty()) {
        float omega = b->ang_vel.get(0, 0);
        float damped_omega = omega * 0.99f;
        if (std::abs(damped_omega) < 0.01f) damped_omega = 0;
        b->ang_vel = Tensor(std::vector<float>{damped_omega}, true);
    }
    
    b->step(sub_dt);
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
int Engine::detect_box_box_multi(Body* a, const Shape& sa, Body* b, const Shape& sb,
                                  float& pen_depth, float& nx, float& ny,
                                  float* contacts_x, float* contacts_y, float* contacts_pen) {
    // Get transforms
    float ax = a->pos.get(0, 0), ay = a->pos.get(1, 0);
    float bx = b->pos.get(0, 0), by = b->pos.get(1, 0);
    float a_rot = a->rotation.get(0, 0);
    float b_rot = b->rotation.get(0, 0);
    
    float cos_a = std::cos(a_rot), sin_a = std::sin(a_rot);
    float cos_b = std::cos(b_rot), sin_b = std::sin(b_rot);
    
    float hw_a = sa.width / 2.0f, hh_a = sa.height / 2.0f;
    float hw_b = sb.width / 2.0f, hh_b = sb.height / 2.0f;
    
    // Get corners of A in world space
    float corners_a[4][2];
    float local_a[4][2] = {{-hw_a, -hh_a}, {hw_a, -hh_a}, {hw_a, hh_a}, {-hw_a, hh_a}};
    for (int i = 0; i < 4; i++) {
        corners_a[i][0] = ax + cos_a * local_a[i][0] - sin_a * local_a[i][1];
        corners_a[i][1] = ay + sin_a * local_a[i][0] + cos_a * local_a[i][1];
    }
    
    // Get corners of B in world space
    float corners_b[4][2];
    float local_b[4][2] = {{-hw_b, -hh_b}, {hw_b, -hh_b}, {hw_b, hh_b}, {-hw_b, hh_b}};
    for (int i = 0; i < 4; i++) {
        corners_b[i][0] = bx + cos_b * local_b[i][0] - sin_b * local_b[i][1];
        corners_b[i][1] = by + sin_b * local_b[i][0] + cos_b * local_b[i][1];
    }
    
    // SAT: check 4 axes (2 per box)
    float axes[4][2] = {
        {cos_a, sin_a}, {-sin_a, cos_a},  // A's axes
        {cos_b, sin_b}, {-sin_b, cos_b}   // B's axes
    };
    
    pen_depth = std::numeric_limits<float>::infinity();
    
    for (int i = 0; i < 4; i++) {
        float axis_x = axes[i][0], axis_y = axes[i][1];
        
        float min_a = std::numeric_limits<float>::infinity();
        float max_a = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; j++) {
            float proj = corners_a[j][0] * axis_x + corners_a[j][1] * axis_y;
            min_a = std::min(min_a, proj);
            max_a = std::max(max_a, proj);
        }
        
        float min_b = std::numeric_limits<float>::infinity();
        float max_b = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; j++) {
            float proj = corners_b[j][0] * axis_x + corners_b[j][1] * axis_y;
            min_b = std::min(min_b, proj);
            max_b = std::max(max_b, proj);
        }
        
        // Check for separation
        if (max_a < min_b || max_b < min_a) {
            return 0;  // No collision
        }
        
        // Calculate overlap
        float overlap = std::min(max_a, max_b) - std::max(min_a, min_b);
        if (overlap < pen_depth) {
            pen_depth = overlap;
            nx = axis_x;
            ny = axis_y;
            
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
    int num_contacts = 0;
    
    // Check corners of A inside B
    for (int i = 0; i < 4 && num_contacts < 4; i++) {
        float px = corners_a[i][0];
        float py = corners_a[i][1];
        float dx = px - bx;
        float dy = py - by;
        float lx = cos_b * dx + sin_b * dy;
        float ly = -sin_b * dx + cos_b * dy;
        
        float pen_x = hw_b - std::abs(lx);
        float pen_y = hh_b - std::abs(ly);
        
        if (pen_x > 0 && pen_y > 0) {
            contacts_x[num_contacts] = px;
            contacts_y[num_contacts] = py;
            contacts_pen[num_contacts] = std::min(pen_x, pen_y);
            num_contacts++;
        }
    }
    
    // Check corners of B inside A
    for (int i = 0; i < 4 && num_contacts < 4; i++) {
        float px = corners_b[i][0];
        float py = corners_b[i][1];
        float dx = px - ax;
        float dy = py - ay;
        float lx = cos_a * dx + sin_a * dy;
        float ly = -sin_a * dx + cos_a * dy;
        
        float pen_x = hw_a - std::abs(lx);
        float pen_y = hh_a - std::abs(ly);
        
        if (pen_x > 0 && pen_y > 0) {
            contacts_x[num_contacts] = px;
            contacts_y[num_contacts] = py;
            contacts_pen[num_contacts] = std::min(pen_x, pen_y);
            num_contacts++;
        }
    }
    
    // If no corners found, use midpoint as fallback
    if (num_contacts == 0) {
        contacts_x[0] = (ax + bx) / 2.0f;
        contacts_y[0] = (ay + by) / 2.0f;
        contacts_pen[0] = pen_depth;
        num_contacts = 1;
    }
    
    return num_contacts;
}

// ============================================================================
// Collision Detection: Box vs Box (SAT-based) - Original single-point version
// ============================================================================

bool Engine::detect_box_box(Body* a, const Shape& sa, Body* b, const Shape& sb,
                            float& pen_depth, float& nx, float& ny, float& cx, float& cy) {
    // Get transforms
    float ax = a->pos.get(0, 0), ay = a->pos.get(1, 0);
    float bx = b->pos.get(0, 0), by = b->pos.get(1, 0);
    float a_rot = a->rotation.get(0, 0);
    float b_rot = b->rotation.get(0, 0);
    
    float cos_a = std::cos(a_rot), sin_a = std::sin(a_rot);
    float cos_b = std::cos(b_rot), sin_b = std::sin(b_rot);
    
    float hw_a = sa.width / 2.0f, hh_a = sa.height / 2.0f;
    float hw_b = sb.width / 2.0f, hh_b = sb.height / 2.0f;
    
    // Get corners of A in world space
    float corners_a[4][2];
    float local_a[4][2] = {{-hw_a, -hh_a}, {hw_a, -hh_a}, {hw_a, hh_a}, {-hw_a, hh_a}};
    for (int i = 0; i < 4; i++) {
        corners_a[i][0] = ax + cos_a * local_a[i][0] - sin_a * local_a[i][1];
        corners_a[i][1] = ay + sin_a * local_a[i][0] + cos_a * local_a[i][1];
    }
    
    // Get corners of B in world space
    float corners_b[4][2];
    float local_b[4][2] = {{-hw_b, -hh_b}, {hw_b, -hh_b}, {hw_b, hh_b}, {-hw_b, hh_b}};
    for (int i = 0; i < 4; i++) {
        corners_b[i][0] = bx + cos_b * local_b[i][0] - sin_b * local_b[i][1];
        corners_b[i][1] = by + sin_b * local_b[i][0] + cos_b * local_b[i][1];
    }
    
    // SAT: check 4 axes (2 per box)
    float axes[4][2] = {
        {cos_a, sin_a}, {-sin_a, cos_a},  // A's axes
        {cos_b, sin_b}, {-sin_b, cos_b}   // B's axes
    };
    
    pen_depth = std::numeric_limits<float>::infinity();
    
    for (int i = 0; i < 4; i++) {
        float axis_x = axes[i][0], axis_y = axes[i][1];
        
        // Project all corners onto axis
        float min_a = std::numeric_limits<float>::infinity();
        float max_a = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; j++) {
            float proj = corners_a[j][0] * axis_x + corners_a[j][1] * axis_y;
            min_a = std::min(min_a, proj);
            max_a = std::max(max_a, proj);
        }
        
        float min_b = std::numeric_limits<float>::infinity();
        float max_b = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; j++) {
            float proj = corners_b[j][0] * axis_x + corners_b[j][1] * axis_y;
            min_b = std::min(min_b, proj);
            max_b = std::max(max_b, proj);
        }
        
        // Check for separation
        if (max_a < min_b || max_b < min_a) {
            return false;  // Separating axis found, no collision
        }
        
        // Calculate overlap
        float overlap = std::min(max_a, max_b) - std::max(min_a, min_b);
        
        // No axis debug
        
        if (overlap < pen_depth) {
            pen_depth = overlap;
            nx = axis_x;
            ny = axis_y;
            
            // Ensure normal points from B to A (standard impulse convention)
            // This way impulse pushes A away from B
            float dx = ax - bx;
            float dy = ay - by;
            if (dx * nx + dy * ny < 0) {
                nx = -nx;
                ny = -ny;
            }
        }
    }
    // Find contact point: deepest penetrating corner
    // Check corners of A against B
    float best_depth = -std::numeric_limits<float>::infinity();
    cx = (ax + bx) / 2.0f;  // Default fallback
    cy = (ay + by) / 2.0f;
    
    for (int i = 0; i < 4; i++) {
        // Transform A's corner to B's local space
        float px = corners_a[i][0];
        float py = corners_a[i][1];
        float dx = px - bx;
        float dy = py - by;
        float lx = cos_b * dx + sin_b * dy;
        float ly = -sin_b * dx + cos_b * dy;
        
        // Check if inside B
        float pen_x = hw_b - std::abs(lx);
        float pen_y = hh_b - std::abs(ly);
        
        if (pen_x > 0 && pen_y > 0) {
            float depth = std::min(pen_x, pen_y);
            if (depth > best_depth) {
                best_depth = depth;
                cx = px;
                cy = py;
            }
        }
    }
    
    // Also check corners of B against A
    for (int i = 0; i < 4; i++) {
        float px = corners_b[i][0];
        float py = corners_b[i][1];
        float dx = px - ax;
        float dy = py - ay;
        float lx = cos_a * dx + sin_a * dy;
        float ly = -sin_a * dx + cos_a * dy;
        
        float pen_x = hw_a - std::abs(lx);
        float pen_y = hh_a - std::abs(ly);
        
        if (pen_x > 0 && pen_y > 0) {
            float depth = std::min(pen_x, pen_y);
            if (depth > best_depth) {
                best_depth = depth;
                cx = px;
                cy = py;
            }
        }
    }
    
    return true;
}

// ============================================================================
// Sequential Impulse Solver: Collision Detection with Manifolds
// ============================================================================

// Clip line segment to a line
// Returns number of output points (0, 1, or 2)
int Engine::clip_segment_to_line(float* v_out, float* v_in, float nx, float ny, float offset) {
    int num_out = 0;
    
    // Distances of end points to the line
    float dist0 = nx * v_in[0] + ny * v_in[1] - offset;
    float dist1 = nx * v_in[2] + ny * v_in[3] - offset;
    
    // If points are behind line, keep them
    if (dist0 <= 0) {
        v_out[num_out * 2] = v_in[0];
        v_out[num_out * 2 + 1] = v_in[1];
        num_out++;
    }
    if (dist1 <= 0) {
        v_out[num_out * 2] = v_in[2];
        v_out[num_out * 2 + 1] = v_in[3];
        num_out++;
    }
    
    // If points are on opposite sides, compute intersection
    if (dist0 * dist1 < 0) {
        float t = dist0 / (dist0 - dist1);
        v_out[num_out * 2] = v_in[0] + t * (v_in[2] - v_in[0]);
        v_out[num_out * 2 + 1] = v_in[1] + t * (v_in[3] - v_in[1]);
        num_out++;
    }
    
    return num_out;
}

// Find the incident face on body B given the reference face normal
void Engine::find_incident_face(float* v, Body* b, float ref_nx, float ref_ny) {
    float b_rot = b->rotation.get(0, 0);
    float cos_b = std::cos(b_rot), sin_b = std::sin(b_rot);
    float bx = b->pos.get(0, 0), by = b->pos.get(1, 0);
    float hw = b->shapes[0].width / 2.0f, hh = b->shapes[0].height / 2.0f;
    
    // B's face normals in world space
    float normals[4][2] = {
        {cos_b, sin_b},       // Right face
        {-sin_b, cos_b},      // Top face
        {-cos_b, -sin_b},     // Left face
        {sin_b, -cos_b}       // Bottom face
    };
    
    // Find face most anti-parallel to reference normal
    int incident_face = 0;
    float min_dot = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 4; ++i) {
        float dot = ref_nx * normals[i][0] + ref_ny * normals[i][1];
        if (dot < min_dot) {
            min_dot = dot;
            incident_face = i;
        }
    }
    
    // Get the two vertices of the incident face
    float local[4][2] = {{hw, -hh}, {hw, hh}, {-hw, hh}, {-hw, -hh}};
    int v0_idx = incident_face;
    int v1_idx = (incident_face + 1) % 4;
    
    v[0] = bx + cos_b * local[v0_idx][0] - sin_b * local[v0_idx][1];
    v[1] = by + sin_b * local[v0_idx][0] + cos_b * local[v0_idx][1];
    v[2] = bx + cos_b * local[v1_idx][0] - sin_b * local[v1_idx][1];
    v[3] = by + sin_b * local[v1_idx][0] + cos_b * local[v1_idx][1];
}

// Detect collision and populate manifold with contact points
bool Engine::detect_collision(Body* a, Body* b, ContactManifold& manifold) {
    // Get transforms
    float ax = a->pos.get(0, 0), ay = a->pos.get(1, 0);
    float bx = b->pos.get(0, 0), by = b->pos.get(1, 0);
    float a_rot = a->rotation.get(0, 0);
    float b_rot = b->rotation.get(0, 0);
    
    float cos_a = std::cos(a_rot), sin_a = std::sin(a_rot);
    float cos_b = std::cos(b_rot), sin_b = std::sin(b_rot);
    
    float hw_a = a->shapes[0].width / 2.0f, hh_a = a->shapes[0].height / 2.0f;
    float hw_b = b->shapes[0].width / 2.0f, hh_b = b->shapes[0].height / 2.0f;
    
    // DEBUG - only when box A is very low (near or below floor)
    static int detect_debug = 0;
    bool debug_this = (detect_debug < 20 && ay < 0.8f);
    if (debug_this) {
        std::cout << "detect_collision: A at (" << ax << "," << ay << ") hw=" << hw_a << " hh=" << hh_a << std::endl;
    }
    
    // Get corners of both boxes in world space
    float local_a[4][2] = {{-hw_a, -hh_a}, {hw_a, -hh_a}, {hw_a, hh_a}, {-hw_a, hh_a}};
    float local_b[4][2] = {{-hw_b, -hh_b}, {hw_b, -hh_b}, {hw_b, hh_b}, {-hw_b, hh_b}};
    
    float corners_a[4][2], corners_b[4][2];
    for (int i = 0; i < 4; ++i) {
        corners_a[i][0] = ax + cos_a * local_a[i][0] - sin_a * local_a[i][1];
        corners_a[i][1] = ay + sin_a * local_a[i][0] + cos_a * local_a[i][1];
        corners_b[i][0] = bx + cos_b * local_b[i][0] - sin_b * local_b[i][1];
        corners_b[i][1] = by + sin_b * local_b[i][0] + cos_b * local_b[i][1];
    }
    
    // SAT: check 4 axes (2 per box)
    float axes[4][2] = {
        {cos_a, sin_a}, {-sin_a, cos_a},
        {cos_b, sin_b}, {-sin_b, cos_b}
    };
    
    float pen_depth = std::numeric_limits<float>::infinity();
    float nx = 0, ny = 1;
    int ref_axis = 0;
    
    for (int i = 0; i < 4; ++i) {
        float axis_x = axes[i][0], axis_y = axes[i][1];
        
        float min_a = std::numeric_limits<float>::infinity();
        float max_a = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; ++j) {
            float proj = corners_a[j][0] * axis_x + corners_a[j][1] * axis_y;
            min_a = std::min(min_a, proj);
            max_a = std::max(max_a, proj);
        }
        
        float min_b = std::numeric_limits<float>::infinity();
        float max_b = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; ++j) {
            float proj = corners_b[j][0] * axis_x + corners_b[j][1] * axis_y;
            min_b = std::min(min_b, proj);
            max_b = std::max(max_b, proj);
        }
        
        if (max_a < min_b || max_b < min_a) {
            if (debug_this) {
                std::cout << "SAT SEPARATED: axis " << i << " max_a=" << max_a << " min_b=" << min_b << " max_b=" << max_b << " min_a=" << min_a << std::endl;
                detect_debug++;
            }
            return false;  // Separating axis found
        }
        
        float overlap = std::min(max_a, max_b) - std::max(min_a, min_b);
        if (overlap < pen_depth) {
            pen_depth = overlap;
            nx = axis_x;
            ny = axis_y;
            ref_axis = i;
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
    Body* ref_body = (ref_axis < 2) ? a : b;
    Body* inc_body = (ref_axis < 2) ? b : a;
    
    // Get incident face vertices
    float incident_face[4];
    find_incident_face(incident_face, inc_body, nx, ny);
    
    // Reference face: compute side planes
    float ref_rot = ref_body->rotation.get(0, 0);
    float ref_cos = std::cos(ref_rot), ref_sin = std::sin(ref_rot);
    float ref_x = ref_body->pos.get(0, 0), ref_y = ref_body->pos.get(1, 0);
    float ref_hw = ref_body->shapes[0].width / 2.0f;
    float ref_hh = ref_body->shapes[0].height / 2.0f;
    
    // Side plane normals (perpendicular to reference normal)
    float side_nx = -ny, side_ny = nx;
    
    // Compute clipping planes
    float ref_c = nx * ref_x + ny * ref_y;  // Reference face offset
    float side_offset = ref_hw;  // Side plane offset (approximate)
    
    // Clip incident face against side planes
    float clip1[4], clip2[4];
    int num = clip_segment_to_line(clip1, incident_face, side_nx, side_ny, -side_offset + (side_nx * ref_x + side_ny * ref_y));
    if (num < 2) return false;
    
    num = clip_segment_to_line(clip2, clip1, -side_nx, -side_ny, -side_offset - (side_nx * ref_x + side_ny * ref_y));
    if (num < 2) return false;
    
    // Calculate reference face offset
    float front = nx * ref_x + ny * ref_y + ((ref_axis % 2 == 0) ? ref_hw : ref_hh);
    
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
            cp.feature_id = (ref_axis << 8) | manifold.point_count;
            manifold.point_count++;
        }
    }
    
    return manifold.point_count > 0;
}

// ============================================================================
// Collision Response: Impulse-based (Legacy)
// ============================================================================

void Engine::apply_impulse(Body* a, Body* b, float nx, float ny, float px, float py) {
    // Mass and inertia
    float m_a = a->is_static ? 1e10f : a->mass.get(0, 0);
    float m_b = b->is_static ? 1e10f : b->mass.get(0, 0);
    float I_a = a->is_static ? 1e10f : a->inertia.get(0, 0);
    float I_b = b->is_static ? 1e10f : b->inertia.get(0, 0);
    
    float inv_m_a = a->is_static ? 0.0f : 1.0f / m_a;
    float inv_m_b = b->is_static ? 0.0f : 1.0f / m_b;
    float inv_I_a = a->is_static ? 0.0f : 1.0f / I_a;
    float inv_I_b = b->is_static ? 0.0f : 1.0f / I_b;
    
    // Positions
    float ax = a->pos.get(0, 0), ay = a->pos.get(1, 0);
    float bx = b->pos.get(0, 0), by = b->pos.get(1, 0);
    
    // Vectors from centers to contact point
    float ra_x = px - ax, ra_y = py - ay;
    float rb_x = px - bx, rb_y = py - by;
    
    // Velocities at contact point
    float va_x = a->vel.get(0, 0), va_y = a->vel.get(1, 0);
    float vb_x = b->vel.get(0, 0), vb_y = b->vel.get(1, 0);
    float omega_a = a->ang_vel.get(0, 0);
    float omega_b = b->ang_vel.get(0, 0);
    
    // Add rotational contribution
    va_x += -omega_a * ra_y;
    va_y +=  omega_a * ra_x;
    vb_x += -omega_b * rb_y;
    vb_y +=  omega_b * rb_x;
    
    // Relative velocity
    float v_rel_x = va_x - vb_x;
    float v_rel_y = va_y - vb_y;
    float v_rel_n = v_rel_x * nx + v_rel_y * ny;
    
    // Don't resolve if separating
    if (v_rel_n > 0) return;
    
    // Coefficient of restitution (average)
    float e = (a->restitution + b->restitution) / 2.0f;
    
    // Cross products for rotational contribution
    float ra_cross_n = ra_x * ny - ra_y * nx;
    float rb_cross_n = rb_x * ny - rb_y * nx;
    
    // Impulse magnitude
    float denom = inv_m_a + inv_m_b + 
                  ra_cross_n * ra_cross_n * inv_I_a +
                  rb_cross_n * rb_cross_n * inv_I_b;
    
    float j = -(1.0f + e) * v_rel_n / denom;
    
    // Apply impulse
    if (!a->is_static) {
        float new_va_x = a->vel.get(0, 0) + j * nx * inv_m_a;
        float new_va_y = a->vel.get(1, 0) + j * ny * inv_m_a;
        float new_omega_a = a->ang_vel.get(0, 0) + ra_cross_n * j * inv_I_a;
        
        // Clamp angular velocity to prevent instability
        const float MAX_OMEGA = 3.0f;  // ~170 degrees/sec
        if (new_omega_a > MAX_OMEGA) new_omega_a = MAX_OMEGA;
        if (new_omega_a < -MAX_OMEGA) new_omega_a = -MAX_OMEGA;
        
        std::vector<float> new_vel_a = {new_va_x, new_va_y};
        a->vel = Tensor(new_vel_a, true);
        std::vector<float> new_omega_a_vec = {new_omega_a};
        a->ang_vel = Tensor(new_omega_a_vec, true);
    }
    
    if (!b->is_static) {
        float new_vb_x = b->vel.get(0, 0) - j * nx * inv_m_b;
        float new_vb_y = b->vel.get(1, 0) - j * ny * inv_m_b;
        float new_omega_b = b->ang_vel.get(0, 0) - rb_cross_n * j * inv_I_b;
        
        std::vector<float> new_vel_b = {new_vb_x, new_vb_y};
        b->vel = Tensor(new_vel_b, true);
        std::vector<float> new_omega_b_vec = {new_omega_b};
        b->ang_vel = Tensor(new_omega_b_vec, true);
    }
    
    // --- FRICTION ---
    // Recalculate relative velocity after normal impulse
    float va_x2 = a->vel.get(0, 0);
    float va_y2 = a->vel.get(1, 0);
    float vb_x2 = b->vel.get(0, 0);
    float vb_y2 = b->vel.get(1, 0);
    float omega_a2 = a->ang_vel.get(0, 0);
    float omega_b2 = b->ang_vel.get(0, 0);
    
    va_x2 += -omega_a2 * ra_y;
    va_y2 +=  omega_a2 * ra_x;
    vb_x2 += -omega_b2 * rb_y;
    vb_y2 +=  omega_b2 * rb_x;
    
    float tx = -ny, ty = nx;  // Tangent
    float v_rel_t = (va_x2 - vb_x2) * tx + (va_y2 - vb_y2) * ty;
    
    // Cross products for tangent
    float ra_cross_t = ra_x * ty - ra_y * tx;
    float rb_cross_t = rb_x * ty - rb_y * tx;
    
    float denom_t = inv_m_a + inv_m_b + 
                    ra_cross_t * ra_cross_t * inv_I_a +
                    rb_cross_t * rb_cross_t * inv_I_b;
    
    float friction_coef = (a->friction + b->friction) / 2.0f;
    float j_t = -v_rel_t / denom_t;
    j_t = std::max(-friction_coef * std::abs(j), std::min(friction_coef * std::abs(j), j_t));  // Clamp to Coulomb cone
    
    if (!a->is_static) {
        float new_va_x = a->vel.get(0, 0) + j_t * tx * inv_m_a;
        float new_va_y = a->vel.get(1, 0) + j_t * ty * inv_m_a;
        float new_omega_a = a->ang_vel.get(0, 0) + ra_cross_t * j_t * inv_I_a;
        std::vector<float> new_vel_a = {new_va_x, new_va_y};
        a->vel = Tensor(new_vel_a, true);
        std::vector<float> new_omega_a_vec = {new_omega_a};
        a->ang_vel = Tensor(new_omega_a_vec, true);
    }
    
    if (!b->is_static) {
        float new_vb_x = b->vel.get(0, 0) - j_t * tx * inv_m_b;
        float new_vb_y = b->vel.get(1, 0) - j_t * ty * inv_m_b;
        float new_omega_b = b->ang_vel.get(0, 0) - rb_cross_t * j_t * inv_I_b;
        std::vector<float> new_vel_b = {new_vb_x, new_vb_y};
        b->vel = Tensor(new_vel_b, true);
        std::vector<float> new_omega_b_vec = {new_omega_b};
        b->ang_vel = Tensor(new_omega_b_vec, true);
    }
}

void Engine::resolve_collision(Body* a, Body* b) {
    for (const Shape& sa : a->shapes) {
        for (const Shape& sb : b->shapes) {
            if (sa.type == Shape::BOX && sb.type == Shape::BOX) {
                float pen, nx, ny;
                float contacts_x[4], contacts_y[4], contacts_pen[4];
                
                int num_contacts = detect_box_box_multi(a, sa, b, sb, pen, nx, ny,
                                                        contacts_x, contacts_y, contacts_pen);
                
                if (num_contacts > 0) {
                    // Position correction (Baumgarte stabilization)
                    float slop = 0.01f;
                    float baumgarte = 0.4f;  // Stronger correction with multi-point
                    float correction = std::max(pen - slop, 0.0f) * baumgarte;
                    
                    if (!a->is_static && !b->is_static) {
                        float total_mass = a->mass.get(0, 0) + b->mass.get(0, 0);
                        float ratio_a = b->mass.get(0, 0) / total_mass;
                        float ratio_b = a->mass.get(0, 0) / total_mass;
                        
                        float new_ax = a->pos.get(0, 0) + nx * correction * ratio_a;
                        float new_ay = a->pos.get(1, 0) + ny * correction * ratio_a;
                        a->pos = Tensor(std::vector<float>{new_ax, new_ay}, true);
                        
                        float new_bx = b->pos.get(0, 0) - nx * correction * ratio_b;
                        float new_by = b->pos.get(1, 0) - ny * correction * ratio_b;
                        b->pos = Tensor(std::vector<float>{new_bx, new_by}, true);
                    } else if (!a->is_static) {
                        float new_ax = a->pos.get(0, 0) + nx * correction;
                        float new_ay = a->pos.get(1, 0) + ny * correction;
                        a->pos = Tensor(std::vector<float>{new_ax, new_ay}, true);
                    } else if (!b->is_static) {
                        float new_bx = b->pos.get(0, 0) - nx * correction;
                        float new_by = b->pos.get(1, 0) - ny * correction;
                        b->pos = Tensor(std::vector<float>{new_bx, new_by}, true);
                    }
                    
                    // Apply impulse at EACH contact point
                    for (int i = 0; i < num_contacts; ++i) {
                        apply_impulse(a, b, nx, ny, contacts_x[i], contacts_y[i]);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Sequential Impulse Solver: Core Functions
// ============================================================================

void Engine::detect_all_collisions() {
    contact_manager.begin_frame();
    
    // Dynamic vs Dynamic
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            ContactManifold* m = contact_manager.get_or_create(bodies[i], bodies[j]);
            if (detect_collision(bodies[i], bodies[j], *m)) {
                m->touching = true;
                m->compute_mass();
            }
        }
    }
    
    // Dynamic vs Static
    static int debug_count = 0;
    for (Body* b : bodies) {
        for (Body* c : colliders) {
            ContactManifold* m = contact_manager.get_or_create(b, c);
            if (detect_collision(b, c, *m)) {
                m->touching = true;
                m->compute_mass();
                if (debug_count < 3) {
                    std::cout << "COLLISION DETECTED: " << m->point_count << " points, normal=(" 
                              << m->normal[0] << "," << m->normal[1] << ")" << std::endl;
                    debug_count++;
                }
            }
        }
    }
    
    contact_manager.end_frame();
    
    // Debug: how many manifolds?
    static int frame_count = 0;
    if (frame_count < 5) {
        std::cout << "Active manifolds: " << contact_manager.get_manifolds().size() << std::endl;
        frame_count++;
    }
}

void Engine::warm_start() {
    for (ContactManifold* m : contact_manager.get_manifolds()) {
        Body* a = m->body_a;
        Body* b = m->body_b;
        
        float inv_m_a = a->is_static ? 0.0f : 1.0f / a->mass.get(0, 0);
        float inv_m_b = b->is_static ? 0.0f : 1.0f / b->mass.get(0, 0);
        float inv_I_a = a->is_static ? 0.0f : 1.0f / a->inertia.get(0, 0);
        float inv_I_b = b->is_static ? 0.0f : 1.0f / b->inertia.get(0, 0);
        
        for (int i = 0; i < m->point_count; ++i) {
            ContactPoint& cp = m->points[i];
            
            // Apply cached impulses
            float px = cp.normal_impulse * m->normal[0] + cp.tangent_impulse * m->tangent[0];
            float py = cp.normal_impulse * m->normal[1] + cp.tangent_impulse * m->tangent[1];
            
            float ax = a->pos.get(0, 0), ay = a->pos.get(1, 0);
            float bx = b->pos.get(0, 0), by = b->pos.get(1, 0);
            float ra_x = cp.position[0] - ax, ra_y = cp.position[1] - ay;
            float rb_x = cp.position[0] - bx, rb_y = cp.position[1] - by;
            
            if (!a->is_static) {
                float va_x = a->vel.get(0, 0) - px * inv_m_a;
                float va_y = a->vel.get(1, 0) - py * inv_m_a;
                float omega_a = a->ang_vel.get(0, 0) - (ra_x * py - ra_y * px) * inv_I_a;
                a->vel = Tensor(std::vector<float>{va_x, va_y}, true);
                a->ang_vel = Tensor(std::vector<float>{omega_a}, true);
            }
            if (!b->is_static) {
                float vb_x = b->vel.get(0, 0) + px * inv_m_b;
                float vb_y = b->vel.get(1, 0) + py * inv_m_b;
                float omega_b = b->ang_vel.get(0, 0) + (rb_x * py - rb_y * px) * inv_I_b;
                b->vel = Tensor(std::vector<float>{vb_x, vb_y}, true);
                b->ang_vel = Tensor(std::vector<float>{omega_b}, true);
            }
        }
    }
}

void Engine::apply_contact_impulse(ContactManifold& m, int idx) {
    Body* a = m.body_a;
    Body* b = m.body_b;
    ContactPoint& cp = m.points[idx];
    
    float inv_m_a = a->is_static ? 0.0f : 1.0f / a->mass.get(0, 0);
    float inv_m_b = b->is_static ? 0.0f : 1.0f / b->mass.get(0, 0);
    float inv_I_a = a->is_static ? 0.0f : 1.0f / a->inertia.get(0, 0);
    float inv_I_b = b->is_static ? 0.0f : 1.0f / b->inertia.get(0, 0);
    
    float ax = a->pos.get(0, 0), ay = a->pos.get(1, 0);
    float bx = b->pos.get(0, 0), by = b->pos.get(1, 0);
    float ra_x = cp.position[0] - ax, ra_y = cp.position[1] - ay;
    float rb_x = cp.position[0] - bx, rb_y = cp.position[1] - by;
    
    // Compute velocity at contact
    float va_x = a->vel.get(0, 0) + (-a->ang_vel.get(0, 0) * ra_y);
    float va_y = a->vel.get(1, 0) + ( a->ang_vel.get(0, 0) * ra_x);
    float vb_x = b->vel.get(0, 0) + (-b->ang_vel.get(0, 0) * rb_y);
    float vb_y = b->vel.get(1, 0) + ( b->ang_vel.get(0, 0) * rb_x);
    
    float v_rel_x = va_x - vb_x;
    float v_rel_y = va_y - vb_y;
    float v_rel_n = v_rel_x * m.normal[0] + v_rel_y * m.normal[1];
    
    // Normal impulse
    float delta_j = -v_rel_n * m.normal_mass[idx];
    
    // Clamp accumulated impulse
    float old_impulse = cp.normal_impulse;
    cp.normal_impulse = std::max(0.0f, old_impulse + delta_j);
    delta_j = cp.normal_impulse - old_impulse;
    
    float px = delta_j * m.normal[0];
    float py = delta_j * m.normal[1];
    
    if (!a->is_static) {
        float new_va_x = a->vel.get(0, 0) + px * inv_m_a;
        float new_va_y = a->vel.get(1, 0) + py * inv_m_a;
        float new_omega_a = a->ang_vel.get(0, 0) + (ra_x * py - ra_y * px) * inv_I_a;
        a->vel = Tensor(std::vector<float>{new_va_x, new_va_y}, true);
        a->ang_vel = Tensor(std::vector<float>{new_omega_a}, true);
    }
    if (!b->is_static) {
        float new_vb_x = b->vel.get(0, 0) - px * inv_m_b;
        float new_vb_y = b->vel.get(1, 0) - py * inv_m_b;
        float new_omega_b = b->ang_vel.get(0, 0) - (rb_x * py - rb_y * px) * inv_I_b;
        b->vel = Tensor(std::vector<float>{new_vb_x, new_vb_y}, true);
        b->ang_vel = Tensor(std::vector<float>{new_omega_b}, true);
    }
    
    // Friction impulse
    float v_rel_t = v_rel_x * m.tangent[0] + v_rel_y * m.tangent[1];
    float delta_jt = -v_rel_t * m.tangent_mass[idx];
    
    // Clamp friction by Coulomb's law
    float max_friction = m.friction * cp.normal_impulse;
    float old_tangent = cp.tangent_impulse;
    cp.tangent_impulse = std::max(-max_friction, std::min(old_tangent + delta_jt, max_friction));
    delta_jt = cp.tangent_impulse - old_tangent;
    
    float tx = delta_jt * m.tangent[0];
    float ty = delta_jt * m.tangent[1];
    
    if (!a->is_static) {
        float new_va_x = a->vel.get(0, 0) + tx * inv_m_a;
        float new_va_y = a->vel.get(1, 0) + ty * inv_m_a;
        float new_omega_a = a->ang_vel.get(0, 0) + (ra_x * ty - ra_y * tx) * inv_I_a;
        a->vel = Tensor(std::vector<float>{new_va_x, new_va_y}, true);
        a->ang_vel = Tensor(std::vector<float>{new_omega_a}, true);
    }
    if (!b->is_static) {
        float new_vb_x = b->vel.get(0, 0) - tx * inv_m_b;
        float new_vb_y = b->vel.get(1, 0) - ty * inv_m_b;
        float new_omega_b = b->ang_vel.get(0, 0) - (rb_x * ty - rb_y * tx) * inv_I_b;
        b->vel = Tensor(std::vector<float>{new_vb_x, new_vb_y}, true);
        b->ang_vel = Tensor(std::vector<float>{new_omega_b}, true);
    }
}

void Engine::solve_velocity_constraints() {
    for (ContactManifold* m : contact_manager.get_manifolds()) {
        for (int i = 0; i < m->point_count; ++i) {
            apply_contact_impulse(*m, i);
        }
    }
}

void Engine::solve_position_constraints() {
    for (ContactManifold* m : contact_manager.get_manifolds()) {
        Body* a = m->body_a;
        Body* b = m->body_b;
        
        for (int i = 0; i < m->point_count; ++i) {
            ContactPoint& cp = m->points[i];
            
            if (cp.penetration <= 0.001f) continue;  // Slop
            
            float correction = (cp.penetration - 0.001f) * 0.2f;  // Baumgarte
            
            if (!a->is_static && !b->is_static) {
                float ax = a->pos.get(0, 0) + m->normal[0] * correction * 0.5f;
                float ay = a->pos.get(1, 0) + m->normal[1] * correction * 0.5f;
                a->pos = Tensor(std::vector<float>{ax, ay}, true);
                float bx = b->pos.get(0, 0) - m->normal[0] * correction * 0.5f;
                float by = b->pos.get(1, 0) - m->normal[1] * correction * 0.5f;
                b->pos = Tensor(std::vector<float>{bx, by}, true);
            } else if (!a->is_static) {
                float ax = a->pos.get(0, 0) + m->normal[0] * correction;
                float ay = a->pos.get(1, 0) + m->normal[1] * correction;
                a->pos = Tensor(std::vector<float>{ax, ay}, true);
            } else if (!b->is_static) {
                float bx = b->pos.get(0, 0) - m->normal[0] * correction;
                float by = b->pos.get(1, 0) - m->normal[1] * correction;
                b->pos = Tensor(std::vector<float>{bx, by}, true);
            }
        }
    }
}

// ============================================================================
// Main Update Loop
// ============================================================================

void Engine::update() {
    float sub_dt = dt / static_cast<float>(substeps);
    
    for (int step = 0; step < substeps; ++step) {
        // 0. Apply motor forces
        for (Body* b : bodies) {
            b->apply_motor_forces();
        }
        
        // 1. Apply gravity
        for (Body* b : bodies) {
            apply_gravity(b, sub_dt);
        }
        
        // 2. Integrate positions and velocities
        for (Body* b : bodies) {
            integrate(b, sub_dt);
        }
        
        // 3. Collision detection and response using OLD working system
        // Dynamic vs Dynamic
        for (size_t i = 0; i < bodies.size(); ++i) {
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                resolve_collision(bodies[i], bodies[j]);
            }
        }
        
        // Dynamic vs Static (colliders)
        for (Body* b : bodies) {
            for (Body* c : colliders) {
                resolve_collision(b, c);
            }
        }
    }
    
    // Clear garbage collectors
    for (Body* b : bodies) {
        b->garbage_collector.clear();
    }
}

// ============================================================================
// Rendering
// ============================================================================

void Engine::render_bodies() {
    // Render colliders (static geometry) in gray - filled
    for (Body* c : colliders) {
        for (const auto& shape : c->shapes) {
            if (shape.type == Shape::BOX) {
                renderer->draw_box_filled(c->get_x(), c->get_y(), 
                                   shape.width, shape.height, 
                                   c->get_rotation(),
                                   0.4f, 0.4f, 0.4f);  // Gray
            }
        }
    }
    
    // Render dynamic bodies in default color - filled
    for (Body* b : bodies) {
        for (const auto& shape : b->shapes) {
            if (shape.type == Shape::BOX) {
                renderer->draw_box_filled(b->get_x(), b->get_y(), 
                                   shape.width, shape.height, 
                                   b->get_rotation());
            }
        }
    }
}

bool Engine::step() {
    if (!renderer->process_events()) {
        return false;
    }
    renderer->clear();
    update();
    render_bodies();
    renderer->present();
    
    // Frame rate limiting based on dt
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(dt * 1000000)));
    return true;
}
