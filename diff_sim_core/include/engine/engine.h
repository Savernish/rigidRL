#ifndef ENGINE_H
#define ENGINE_H

#include <vector>
#include "renderer/renderer.h"
#include "engine/body.h"
#include "engine/tensor.h"
#include "engine/contact.h"

class Engine {
    Renderer* renderer;
    std::vector<Body*> bodies;       // Dynamic bodies
    std::vector<Body*> colliders;    // Static colliders (ground, walls, etc.)
    
    // Contact management for sequential impulse solver
    ContactManager contact_manager;
    
    float dt;
    int substeps;
    float gravity_x;
    float gravity_y;
    
    // Solver settings
    int velocity_iterations = 8;
    int position_iterations = 3;

public:
    Engine(int width = 800, int height = 600, float scale = 50.0f, 
           float dt = 0.016f, int substeps = 10);
    ~Engine();
    
    // Body management
    void add_body(Body* b);
    
    // Static geometry (ground, walls, platforms)
    Body* add_collider(float x, float y, float width, float height, float rotation = 0.0f);
    void clear_colliders();
    
    // Environment
    void set_gravity(float x, float y);
    
    // Simulation
    void update();          // Physics step only
    void render_bodies();   // Render all bodies + colliders
    bool step();            // Full frame: events + physics + render
    
    // Accessor
    Renderer* get_renderer() { return renderer; }

private:
    // Physics helpers
    void apply_gravity(Body* b, float sub_dt);
    void integrate(Body* b, float sub_dt);
    
    // Collision detection
    bool detect_collision(Body* a, Body* b, ContactManifold& manifold);
    void find_incident_face(float* v, Body* b, float nx, float ny);
    int clip_segment_to_line(float* v_out, float* v_in, float nx, float ny, float offset);
    
    // Sequential impulse solver
    void detect_all_collisions();
    void warm_start();
    void solve_velocity_constraints();
    void solve_position_constraints();
    void apply_contact_impulse(ContactManifold& m, int point_index);
    
    // Legacy (kept for compatibility during transition)
    void resolve_collision(Body* a, Body* b);
    bool detect_box_box(Body* a, const Shape& sa, Body* b, const Shape& sb,
                        float& pen_depth, float& nx, float& ny, float& cx, float& cy);
    int detect_box_box_multi(Body* a, const Shape& sa, Body* b, const Shape& sb,
                              float& pen_depth, float& nx, float& ny,
                              float* contacts_x, float* contacts_y, float* contacts_pen);
    void apply_impulse(Body* a, Body* b, float nx, float ny, float cx, float cy);
};

#endif // ENGINE_H
