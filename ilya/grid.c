#include <grid.h>

//Простая равномерная сетка, которую будем считать на лету
Point uniform_grid(Point starting_point, int x_offset, int y_offset, double h) {
    return (Point){starting_point.x + h*x_offset, starting_point.y + h*y_offset};
}