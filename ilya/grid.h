#pragma once

typedef struct {
    double x;
    double y;
} Point;

//Простая равномерная сетка, которую будем считать на лету
Point uniform_grid(Point starting_point, int x_offset, int y_offset, double h);