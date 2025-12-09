#ifndef ENGINE_H
#define ENGINE_H

#include <vector>
#include "renderer/renderer.h"
#include "engine/body.h"
#include "engine/tensor.h"

class Engine {
    Renderer* renderer;
    std::vector<Body*> bodies;       // Dynamic bodies
    std::vector<Body*> colliders;    // Static colliders (ground, walls, etc.)
    
    float dt;
    int substeps;
    float gravity_x;
    float gravity_y;

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
    
    // Impulse-based collision
    void resolve_collision(Body* a, Body* b);
    bool detect_box_box(Body* a, const Shape& sa, Body* b, const Shape& sb,
                        float& pen_depth, float& nx, float& ny, float& cx, float& cy);
    void apply_impulse(Body* a, Body* b, float nx, float ny, float cx, float cy);
};

#endif // ENGINE_H
