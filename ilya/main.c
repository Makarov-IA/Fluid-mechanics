#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <grid.h>

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) < (b) ? (b) : (a))

int main(int argc, char *argv[]) {
    //Пока не трогаю аргументы: скорее всего не буду использовать соло, а как функцию, вызываемую из .py скрипта
    //TODO: интегрировать как python функию
    Point starting_point;
    double Lx, Ly, hx, hy;
    int Nx, Ny;
    double (*f)(Point);
    //TODO: Parse here
    Nx = 100; Ny = 100;
    Lx = 1.0; Ly = 1.0;
    hx = Lx/Nx; hy = Ly/Ny; 


}