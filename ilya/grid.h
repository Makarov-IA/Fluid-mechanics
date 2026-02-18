#pragma once

struct Point {
    double x;
    double y;
};

Point uniform_grid(Point starting_point, int x_offset, int y_offset, double h);