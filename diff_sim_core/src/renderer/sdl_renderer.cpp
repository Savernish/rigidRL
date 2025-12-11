#include "renderer/sdl_renderer.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <climits>

SDLRenderer::SDLRenderer(int width, int height, float scale_factor) 
    : Renderer(width, height, scale_factor), window(nullptr), renderer(nullptr)
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
        SDL_Quit();
        return;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        // Try software renderer as fallback
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
        if (!renderer) {
            std::cerr << "Software renderer also failed! SDL_Error: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            window = nullptr;
            SDL_Quit();
            return;
        }
        std::cout << "Using software renderer as fallback" << std::endl;
    }
}

SDLRenderer::~SDLRenderer() {
    if (renderer) {
        SDL_DestroyRenderer(renderer);
    }
    if (window) {
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
}

void SDLRenderer::clear() {
    if (!renderer) return;
    SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255); // Dark Gray
    SDL_RenderClear(renderer);
}

void SDLRenderer::present() {
    if (!renderer) return;
    SDL_RenderPresent(renderer);
}

bool SDLRenderer::process_events() {
    if (!window) return false;  // No window means we should quit
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
    if (!renderer) return;
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

void SDLRenderer::draw_box_filled(float x, float y, float w, float h, float rotation, float r, float g, float b) {
    if (!renderer) return;
    SDL_SetRenderDrawColor(renderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);

    float cos_t = std::cos(rotation);
    float sin_t = std::sin(rotation);
    float hw = w / 2.0f;
    float hh = h / 2.0f;

    float local_x[] = {-hw, hw, hw, -hw};
    float local_y[] = {-hh, -hh, hh, hh};

    // Get screen coordinates of corners
    int screen_x[4], screen_y[4];
    int min_y = INT_MAX, max_y = INT_MIN;
    
    for (int i = 0; i < 4; ++i) {
        float rot_x = local_x[i] * cos_t - local_y[i] * sin_t;
        float rot_y = local_x[i] * sin_t + local_y[i] * cos_t;
        screen_x[i] = to_screen_x(x + rot_x);
        screen_y[i] = to_screen_y(y + rot_y);
        min_y = std::min(min_y, screen_y[i]);
        max_y = std::max(max_y, screen_y[i]);
    }

    // Scanline fill: for each row, find intersections with polygon edges
    for (int scanY = min_y; scanY <= max_y; ++scanY) {
        std::vector<int> intersections;
        
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            int y1 = screen_y[i], y2 = screen_y[j];
            int x1 = screen_x[i], x2 = screen_x[j];
            
            if ((y1 <= scanY && y2 > scanY) || (y2 <= scanY && y1 > scanY)) {
                int intersectX = x1 + (scanY - y1) * (x2 - x1) / (y2 - y1);
                intersections.push_back(intersectX);
            }
        }
        
        std::sort(intersections.begin(), intersections.end());
        
        for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
            SDL_RenderDrawLine(renderer, intersections[i], scanY, intersections[i+1], scanY);
        }
    }
}

void SDLRenderer::draw_line(float x1, float y1, float x2, float y2, float r, float g, float b) {
    if (!renderer) return;
    SDL_SetRenderDrawColor(renderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);
    SDL_RenderDrawLine(renderer, to_screen_x(x1), to_screen_y(y1), to_screen_x(x2), to_screen_y(y2));
}

void SDLRenderer::draw_circle(float centreX, float centreY, float radius, float r, float g, float b) {
    if (!renderer) return;
    // Convert world center to screen coordinates
    int screenCenterX = to_screen_x(centreX);
    int screenCenterY = to_screen_y(centreY);
    int screenRadius = static_cast<int>(radius * scale);
    
    const int32_t diameter = (screenRadius * 2);

    int32_t x = (screenRadius - 1);
    int32_t y = 0;
    int32_t tx = 1;
    int32_t ty = 1;
    int32_t error = (tx - diameter);

    SDL_SetRenderDrawColor(renderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);

    while (x >= y)
    {
        //  Each of the following renders an octant of the circle
        SDL_RenderDrawPoint(renderer, screenCenterX + x, screenCenterY - y);
        SDL_RenderDrawPoint(renderer, screenCenterX + x, screenCenterY + y);
        SDL_RenderDrawPoint(renderer, screenCenterX - x, screenCenterY - y);
        SDL_RenderDrawPoint(renderer, screenCenterX - x, screenCenterY + y);
        SDL_RenderDrawPoint(renderer, screenCenterX + y, screenCenterY - x);
        SDL_RenderDrawPoint(renderer, screenCenterX + y, screenCenterY + x);
        SDL_RenderDrawPoint(renderer, screenCenterX - y, screenCenterY - x);
        SDL_RenderDrawPoint(renderer, screenCenterX - y, screenCenterY + x);

        if (error <= 0)
        {
            ++y;
            error += ty;
            ty += 2;
        }

        if (error > 0)
        {
            --x;
            tx += 2;
            error += (tx - diameter);
        }
    }
}


