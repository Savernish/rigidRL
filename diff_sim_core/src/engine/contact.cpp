#include "engine/contact.h"
#include "engine/body.h"
#include <cmath>

void ContactManifold::compute_mass() {
    if (!body_a || !body_b) return;
    
    // Get inverse masses and inertias
    float inv_m_a = body_a->is_static ? 0.0f : 1.0f / body_a->mass.get(0, 0);
    float inv_m_b = body_b->is_static ? 0.0f : 1.0f / body_b->mass.get(0, 0);
    float inv_I_a = body_a->is_static ? 0.0f : 1.0f / body_a->inertia.get(0, 0);
    float inv_I_b = body_b->is_static ? 0.0f : 1.0f / body_b->inertia.get(0, 0);
    
    float ax = body_a->pos.get(0, 0);
    float ay = body_a->pos.get(1, 0);
    float bx = body_b->pos.get(0, 0);
    float by = body_b->pos.get(1, 0);
    
    for (int i = 0; i < point_count; ++i) {
        ContactPoint& p = points[i];
        
        // Vector from body center to contact point
        float ra_x = p.position[0] - ax;
        float ra_y = p.position[1] - ay;
        float rb_x = p.position[0] - bx;
        float rb_y = p.position[1] - by;
        
        // Cross products for normal direction
        float ra_cross_n = ra_x * normal[1] - ra_y * normal[0];
        float rb_cross_n = rb_x * normal[1] - rb_y * normal[0];
        
        // Effective mass for normal constraint
        float k_normal = inv_m_a + inv_m_b + 
                         ra_cross_n * ra_cross_n * inv_I_a +
                         rb_cross_n * rb_cross_n * inv_I_b;
        
        normal_mass[i] = (k_normal > 0) ? 1.0f / k_normal : 0.0f;
        
        // Cross products for tangent direction
        float ra_cross_t = ra_x * tangent[1] - ra_y * tangent[0];
        float rb_cross_t = rb_x * tangent[1] - rb_y * tangent[0];
        
        // Effective mass for friction constraint
        float k_tangent = inv_m_a + inv_m_b + 
                          ra_cross_t * ra_cross_t * inv_I_a +
                          rb_cross_t * rb_cross_t * inv_I_b;
        
        tangent_mass[i] = (k_tangent > 0) ? 1.0f / k_tangent : 0.0f;
    }
}

// ============================================================================
// ContactManager Implementation
// ============================================================================

ContactManifold* ContactManager::get_or_create(Body* a, Body* b) {
    ContactKey key{a, b};
    
    auto it = manifold_cache.find(key);
    if (it != manifold_cache.end()) {
        // Existing manifold - mark as still active
        it->second.was_touching = it->second.touching;
        return &it->second;
    }
    
    // Create new manifold
    ContactManifold manifold;
    manifold.body_a = a;
    manifold.body_b = b;
    
    // Combine material properties
    manifold.friction = std::sqrt(a->friction * b->friction);
    manifold.restitution = std::max(a->restitution, b->restitution);
    
    auto result = manifold_cache.insert({key, manifold});
    return &result.first->second;
}

ContactManifold* ContactManager::find(Body* a, Body* b) {
    ContactKey key{a, b};
    auto it = manifold_cache.find(key);
    return (it != manifold_cache.end()) ? &it->second : nullptr;
}

void ContactManager::begin_frame() {
    // Mark all manifolds as not touching (will be updated during collision detection)
    for (auto& pair : manifold_cache) {
        pair.second.was_touching = pair.second.touching;
        pair.second.touching = false;
    }
    active_manifolds.clear();
}

void ContactManager::end_frame() {
    // Remove manifolds that are no longer touching
    for (auto it = manifold_cache.begin(); it != manifold_cache.end();) {
        if (!it->second.touching) {
            it = manifold_cache.erase(it);
        } else {
            // Add to active list for solving
            active_manifolds.push_back(&it->second);
            ++it;
        }
    }
}

void ContactManager::clear() {
    manifold_cache.clear();
    active_manifolds.clear();
}
