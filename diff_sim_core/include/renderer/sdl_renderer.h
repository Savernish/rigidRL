#ifndef SDL_RENDERER_H
#define SDL_RENDERER_H

#include "renderer.h"
#include <SDL2/SDL.h>

class SDLRenderer : public Renderer {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;

    // Coordinate conversion
    int to_screen_x(float sim_x);
    int to_screen_y(float sim_y);

public:
    SDLRenderer(int width=800, int height=600, float scale_factor=50.0f);
    ~SDLRenderer();

    void clear() override;
    void present() override;
    bool process_events() override;
    void draw_box(float x, float y, float w, float h, float rotation,
                  float r, float g, float b) override;
    void draw_box_filled(float x, float y, float w, float h, float rotation,
                  float r, float g, float b) override;
    void draw_circle(float centerX, float centerY, float radius,
                  float r, float g, float b) override;
    void draw_line(float x1, float y1, float x2, float y2, 
                  float r, float g, float b) override;

};

#endif // SDL_RENDERER_H
