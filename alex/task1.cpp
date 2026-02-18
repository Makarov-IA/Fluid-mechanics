#include <iostream>
#include <vector>
#include <cmath>
#include "./Eigen/Dense"
#include "./Eigen/Sparse"

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using VectorXd = Eigen::VectorXd;

int main() {
    
    int N = 50;
    double h = 1.0 / (N + 1);
    int total_points = N * N;

    std::cout << "Размерность сетки: " << N << "x" << N << std::endl;
    std::cout << "Шаг h: " << h << std::endl;
    std::cout << "Общее число неизвестных: " << total_points << std::endl;

    VectorXd b = VectorXd::Zero(total_points);

    // Список триплетов для построения разреженной матрицы A (i, j, value)
    // Для 5-точечной схемы у нас максимум 5 ненулевых элементов на строку
    std::vector<Triplet> tripletList;
    tripletList.reserve(total_points * 5);

    // Коэффициенты разностной схемы
    // u_xx ~ (u(i+1,j) - 2u(i,j) + u(i-1,j)) / h^2
    // u_yy ~ (u(i,j+1) - 2u(i,j) + u(i,j-1)) / h^2
    // Уравнение: u_xx - u_yy = f
    // (u(i+1,j) - 2u(i,j) + u(i-1,j) - u(i,j+1) + 2u(i,j) - u(i,j-1)) / h^2 = f
    // Упрощаем: (u(i+1,j) + u(i-1,j) - u(i,j+1) - u(i,j-1)) / h^2 = f
    // Заметьте: коэффициенты при u(i,j) сокращаются (-2 + 2 = 0)!
    // Это характерно для волнового уравнения (гиперболический тип) в стационарной постановке.
    
    double inv_h2 = 1.0 / (h * h);

    // Функция f(x, y). Замените на свою функцию.
    auto f_func = [](double x, double y) {
        return std::sin(M_PI * x) * std::sin(M_PI * y); // Пример
    };

    // Заполнение матрицы и вектора
    for (int j = 0; j < N; ++j) {       // Индекс по Y (строки сетки)
        for (int i = 0; i < N; ++i) {   // Индекс по X (столбцы сетки)
            
            int idx = j * N + i; // Линейный индекс неизвестного u(i, j)
            
            double x = (i + 1) * h;
            double y = (j + 1) * h;

            // Правая часть
            b(idx) = f_func(x, y) * inv_h2; // Умножаем на h^2, так как мы перенесли его в знаменатель коэффициентов

            // Центральная точка: коэффициент 0 (так как -2/h^2 от u_xx и +2/h^2 от -u_yy дают 0)
            // Но для устойчивости численных методов иногда добавляют регуляризацию, 
            // однако строго по схеме: coeff_center = 0.
            tripletList.push_back(Triplet(idx, idx, 0.0)); 

            // Сосед слева (i-1, j) -> коэффициент +1/h^2
            if (i > 0) {
                tripletList.push_back(Triplet(idx, idx - 1, inv_h2));
            } else {
                // Граничное условие u(0, y) = 0, вклад в правую часть равен 0, ничего не добавляем
            }

            // Сосед справа (i+1, j) -> коэффициент +1/h^2
            if (i < N - 1) {
                tripletList.push_back(Triplet(idx, idx + 1, inv_h2));
            } else {
                // Граничное условие u(1, y) = 0
            }

            // Сосед снизу (i, j-1) -> коэффициент -1/h^2 (знак минус из-за -u_yy)
            if (j > 0) {
                tripletList.push_back(Triplet(idx, idx - N, -inv_h2));
            } else {
                // Граничное условие u(x, 0) = 0
            }

            // Сосед сверху (i, j+1) -> коэффициент -1/h^2
            if (j < N - 1) {
                tripletList.push_back(Triplet(idx, idx + N, -inv_h2));
            } else {
                // Граничное условие u(x, 1) = 0
            }
        }
    }

    // Создание разреженной матрицы
    SparseMatrix A(total_points, total_points);
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // Опционально: сжатие памяти (удаление явно сохраненных нулей, если они есть)
    A.makeCompressed();

    std::cout << "Матрица A создана. Ненулевых элементов: " << A.nonZeros() << std::endl;
    std::cout << "Размер матрицы: " << A.rows() << "x" << A.cols() << std::endl;

    // --- Далее данные готовы для передачи в решатель ---
    
    // Пример использования прямого решателя для разреженных матриц (SimplicialLDLT)
    // Внимание: Матрица для уравнения u_xx - u_yy не является положительно определенной,
    // поэтому LDLT может не подойти. Лучше использовать LU или GMRES.
    
    Eigen::SparseLU<SparseMatrix> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Разложение матрицы не удалось!" << std::endl;
        return -1;
    }

    VectorXd u = solver.solve(b);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Решение системы не удалось!" << std::endl;
        return -1;
    }

    std::cout << "Система решена успешно." << std::endl;
    std::cout << "Пример значения в центре области: " << u(total_points / 2) << std::endl;

    // Здесь можно добавить код для вывода данных в файл для визуализации
    // Например, в формате CSV или простом текстовом виде
    
    /*
    std::ofstream outfile("solution.txt");
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            outfile << u(j * N + i) << " ";
        }
        outfile << "\n";
    }
    outfile.close();
    */

    return 0;
}