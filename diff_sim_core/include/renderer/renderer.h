#ifndef RENDERER_H
#define RENDERER_H

class Renderer {
public:
    Renderer(int width, int height, float scale) : width(width), height(height), scale(scale) {}
    virtual ~Renderer() = default;

    // Window properties
    int get_width() const { return width; }
    int get_height() const { return height; }
    float get_scale() const { return scale; }

    // Window management
    virtual void clear() = 0;
    virtual void present() = 0;
    virtual bool process_events() = 0; // Return false if user requested closed

    // Drawing primitives
    // Coordinates are in simulation space (meters), renderer handles scaling
    virtual void draw_box(float x, float y, float w, float h, float rotation, 
                          float r=1.0f, float g=1.0f, float b=1.0f) = 0;
    
    virtual void draw_box_filled(float x, float y, float w, float h, float rotation,
                                  float r=1.0f, float g=1.0f, float b=1.0f) = 0;

    virtual void draw_line(float x1, float y1, float x2, float y2, float r=1.0f, float g=1.0f, float b=1.0f) = 0;

    virtual void draw_circle(float centreX, float centreY, float radius, float r=1.0f, float g=1.0f, float b=1.0f) = 0;

    //virtual void draw_triangle(float x1, float y1, float x2, float y2, float x3, float y3, float r=1.0f, float g=1.0f, float b=1.0f) = 0;
    
protected:
    float scale;
    int width, height;
};

#endif // RENDERER_H
