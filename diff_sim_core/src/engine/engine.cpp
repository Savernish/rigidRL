#include "engine/engine.h"
#include "renderer/sdl_renderer.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

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

Body* Engine::add_collider(float x, float y, float width, float height, float rotation) {
    Body* c = Body::create_static(x, y, width, height, rotation);
    colliders.push_back(c);
    return c;
}

void Engine::clear_colliders() {
    for (Body* c : colliders) {
        delete c;
    }
    colliders.clear();
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
    b->step(sub_dt);
}

// ============================================================================
// Collision Detection: Box vs Box (SAT-based)
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
// Collision Response: Impulse-based
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
                float pen, nx, ny, cx, cy;
                if (detect_box_box(a, sa, b, sb, pen, nx, ny, cx, cy)) {
                    // Position correction (push apart fully)
                    float slop = 0.01f;  // Allow small penetration to avoid jitter
                    float correction = std::max(pen - slop, 0.0f);
                    
                    if (!a->is_static && !b->is_static) {
                        float total_mass = a->mass.get(0, 0) + b->mass.get(0, 0);
                        float ratio_a = b->mass.get(0, 0) / total_mass;
                        float ratio_b = a->mass.get(0, 0) / total_mass;
                        
                        float new_ax = a->pos.get(0, 0) - nx * correction * ratio_a;
                        float new_ay = a->pos.get(1, 0) - ny * correction * ratio_a;
                        std::vector<float> new_pos_a = {new_ax, new_ay};
                        a->pos = Tensor(new_pos_a, true);
                        
                        float new_bx = b->pos.get(0, 0) + nx * correction * ratio_b;
                        float new_by = b->pos.get(1, 0) + ny * correction * ratio_b;
                        std::vector<float> new_pos_b = {new_bx, new_by};
                        b->pos = Tensor(new_pos_b, true);
                    } else if (!a->is_static) {
                        float new_ax = a->pos.get(0, 0) - nx * correction;
                        float new_ay = a->pos.get(1, 0) - ny * correction;
                        std::vector<float> new_pos_a = {new_ax, new_ay};
                        a->pos = Tensor(new_pos_a, true);
                    } else if (!b->is_static) {
                        float new_bx = b->pos.get(0, 0) + nx * correction;
                        float new_by = b->pos.get(1, 0) + ny * correction;
                        std::vector<float> new_pos_b = {new_bx, new_by};
                        b->pos = Tensor(new_pos_b, true);
                    }
                    
                    // Apply impulse
                    apply_impulse(a, b, nx, ny, cx, cy);
                }
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
        // 1. Apply gravity and integrate (update velocities and positions)
        for (Body* b : bodies) {
            apply_gravity(b, sub_dt);
            integrate(b, sub_dt);
        }
        
        // 2. Collision detection and response (after integration)
        // This corrects positions and velocities for penetration
        
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
    // Render colliders (static geometry) in gray
    for (Body* c : colliders) {
        for (const auto& shape : c->shapes) {
            if (shape.type == Shape::BOX) {
                renderer->draw_box(c->get_x(), c->get_y(), 
                                   shape.width, shape.height, 
                                   c->get_rotation(),
                                   0.4f, 0.4f, 0.4f);  // Gray
            }
        }
    }
    
    // Render dynamic bodies in default color
    for (Body* b : bodies) {
        for (const auto& shape : b->shapes) {
            if (shape.type == Shape::BOX) {
                renderer->draw_box(b->get_x(), b->get_y(), 
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
    return true;
}
