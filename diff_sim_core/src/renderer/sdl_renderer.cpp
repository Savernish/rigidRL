#include "renderer/sdl_renderer.h"
#include <iostream>
#include <cmath>

SDLRenderer::SDLRenderer(int width, int height, float scale_factor) 
    : Renderer(width, height, scale_factor)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return;
    }

    window = SDL_CreateWindow("rigidRL Physics Engine", 
                              SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 
                              width, height, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
    }
}

SDLRenderer::~SDLRenderer() {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void SDLRenderer::clear() {
    SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255); // Dark Gray
    SDL_RenderClear(renderer);
}

void SDLRenderer::present() {
    SDL_RenderPresent(renderer);
}

bool SDLRenderer::process_events() {
    SDL_Event e;
    while (SDL_PollEvent(&e) != 0) {
        if (e.type == SDL_QUIT) {
            return false;
        }
    }
    return true;
}

int SDLRenderer::to_screen_x(float sim_x) {
    return width / 2 + (int)(sim_x * scale);
}

int SDLRenderer::to_screen_y(float sim_y) {
    return height - 50 - (int)(sim_y * scale); // Ground at bottom, Y-up
}

void SDLRenderer::draw_box(float x, float y, float w, float h, float rotation, float r, float g, float b) {
    SDL_SetRenderDrawColor(renderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);

    // Calculate corners
    // Local corners: (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
    // Rotate by theta around (x, y)
    float cos_t = std::cos(rotation);
    float sin_t = std::sin(rotation);

    float hw = w / 2.0f;
    float hh = h / 2.0f;

    float local_x[] = {-hw, hw, hw, -hw};
    float local_y[] = {-hh, -hh, hh, hh};

    SDL_Point points[5]; // 5 points to close loop

    for (int i = 0; i < 4; ++i) {
        float rot_x = local_x[i] * cos_t - local_y[i] * sin_t;
        float rot_y = local_x[i] * sin_t + local_y[i] * cos_t;
        
        points[i].x = to_screen_x(x + rot_x);
        points[i].y = to_screen_y(y + rot_y);
    }
    points[4] = points[0];

    SDL_RenderDrawLines(renderer, points, 5);
}

void SDLRenderer::draw_line(float x1, float y1, float x2, float y2, float r, float g, float b) {
    SDL_SetRenderDrawColor(renderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);
    SDL_RenderDrawLine(renderer, to_screen_x(x1), to_screen_y(y1), to_screen_x(x2), to_screen_y(y2));
}

// void SDLRenderer::draw_circle(float x, float y, float radius, float r, float g, float b) {
//     SDL_SetRenderDrawColor(renderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);
//     SDL_RenderDrawCircle(renderer, to_screen_x(x), to_screen_y(y), radius);
// }

