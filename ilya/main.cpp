#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <grid.h>

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) < (b) ? (b) : (a))

int main(int argc, char *argv[]) {
    //TODO: интегрировать как python функию
    Point sp;
    double Lx, Ly, hx, hy;
    int Nx, Ny;
    double (*f)(Point);
    //TODO: парсить тут
    Nx = 100; Ny = 100;
    Lx = 1.0; Ly = 1.0;
    hx = Lx/Nx; hy = Ly/Ny; 


}