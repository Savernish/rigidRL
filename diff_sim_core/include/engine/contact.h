#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <array>

// Forward declaration
class Body;

// Maximum contact points per manifold (4 for box-box)
constexpr int MAX_CONTACT_POINTS = 4;

/**
 * ContactPoint - Single contact point within a manifold
 * 
 * Stores both local coordinates (for warm starting) and
 * accumulated impulses for sequential impulse solving.
 */
struct ContactPoint {
    // Contact position in local space of each body
    float local_a[2] = {0, 0};
    float local_b[2] = {0, 0};
    
    // World-space contact position (computed each frame)
    float position[2] = {0, 0};
    
    // Penetration depth at this contact
    float penetration = 0;
    
    // Accumulated impulses (for warm starting)
    float normal_impulse = 0;
    float tangent_impulse = 0;
    
    // Feature ID for matching contacts across frames
    // High bits: face index, Low bits: vertex/edge index
    uint32_t feature_id = 0;
};

/**
 * ContactManifold - Collection of contact points between two bodies
 * 
 * Represents the collision between two bodies with 1-4 contact points.
 * Persists across frames for warm starting.
 */
struct ContactManifold {
    // Bodies involved in the contact
    Body* body_a = nullptr;
    Body* body_b = nullptr;
    
    // Contact normal (world space, points from A to B)
    float normal[2] = {0, 1};
    
    // Contact tangent (perpendicular to normal, for friction)
    float tangent[2] = {1, 0};
    
    // Contact points
    std::array<ContactPoint, MAX_CONTACT_POINTS> points;
    int point_count = 0;
    
    // Combined material properties
    float friction = 0.5f;
    float restitution = 0.0f;
    
    // For tracking persistent contacts
    bool touching = false;
    bool was_touching = false;
    
    // Effective mass (precomputed for solver)
    float normal_mass[MAX_CONTACT_POINTS] = {0};
    float tangent_mass[MAX_CONTACT_POINTS] = {0};
    
    /**
     * Update tangent vector from normal
     */
    void compute_tangent() {
        tangent[0] = -normal[1];
        tangent[1] = normal[0];
    }
    
    /**
     * Precompute effective masses for the constraint solver
     */
    void compute_mass();
};

/**
 * ContactKey - Unique identifier for a body pair
 * Used as key in the contact cache
 */
struct ContactKey {
    Body* a;
    Body* b;
    
    bool operator==(const ContactKey& other) const {
        return (a == other.a && b == other.b) || (a == other.b && b == other.a);
    }
};

// Hash function for ContactKey
struct ContactKeyHash {
    size_t operator()(const ContactKey& key) const {
        // Order-independent hash
        uintptr_t p1 = reinterpret_cast<uintptr_t>(key.a);
        uintptr_t p2 = reinterpret_cast<uintptr_t>(key.b);
        if (p1 > p2) std::swap(p1, p2);
        return std::hash<uintptr_t>()(p1) ^ (std::hash<uintptr_t>()(p2) << 1);
    }
};

#include <unordered_map>
#include <vector>

/**
 * ContactManager - Manages contact manifolds across frames
 * 
 * Handles:
 * - Contact persistence for warm starting
 * - Adding/removing contacts as collisions start/end
 * - Providing manifolds to the solver
 */
class ContactManager {
public:
    // Get or create a manifold for a body pair
    ContactManifold* GetOrCreate(Body* pBodyA, Body* pBodyB);
    
    // Find existing manifold (returns nullptr if not found)
    ContactManifold* Find(Body* pBodyA, Body* pBodyB);
    
    // Mark all manifolds as potentially stale
    void BeginFrame();
    
    // Remove manifolds that weren't updated this frame
    void EndFrame();
    
    // Get all active manifolds for solving
    std::vector<ContactManifold*>& GetManifolds() { return m_ActiveManifolds; }
    
    // Clear all contacts
    void Clear();
    
private:
    std::unordered_map<ContactKey, ContactManifold, ContactKeyHash> m_ManifoldCache;
    std::vector<ContactManifold*> m_ActiveManifolds;
};
