#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <legendre_zero_point.hpp>
#include <functional>
#include <TRG.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1

using std::cin;
using std::cout;
using std::cerr;
using std::string;

// m x n 行列を出力
template<typename T>
void print_matrix(T *matrix, MKL_INT m, MKL_INT n, MKL_INT lda, const string &message) {
    cout << '\n' << message << '\n';
    REP(i, m) {
        REP(j, n) {
            cout << std::scientific << std::setprecision(5) << (matrix[i * lda + j] >= 0 ? " " : "")
                 << matrix[i * lda + j]
                 << ' ';
        }
        cout << '\n';
    }
    cout << '\n';
}

double
Trace(double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N, MKL_INT *order) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

//    std::vector<double> x = math::solver::legendre_zero_point(n_node);
    std::vector<double> x(n_node);
//    std::vector<double> p(n_node);
    std::vector<double> w(n_node);

    std::ifstream GL_node;
    GL_node.open("GL-node.txt", std::ios::in);
    for (int i = 0; GL_node >> x[i] >> w[i]; ++i);
    GL_node.close();

//    REP(i, n_node) {
//        p[i] = std::legendre(n_node - 1, x[i]);
//    }
//    REP(i, n_node) {
//        w[i] = 2 * (1 - x[i] * x[i]) / (n_node * n_node * p[i] * p[i]);
//    }

    // initialize tensor network : max index size is D_cut
    std::vector<std::vector<std::vector<std::vector<double>>>> T(D_cut, std::vector<std::vector<std::vector<double>>>(D_cut, std::vector<std::vector<double>>(D_cut, std::vector<double>(D_cut))));
    std::function<double(double, double, double, double)> f = [=](double theta1, double phi1, double theta2,
                                                                  double phi2) {
        std::function<double(double)> s = [=](double theta) { return std::sin(M_PI * theta / 2); };
        std::function<double(double)> c = [=](double theta) { return std::cos(M_PI * theta / 2); };
        return std::exp(K * (s(theta1) * s(theta2) + c(theta1) * c(theta2) * std::cos(M_PI * (phi1 - phi2))));
    };
    double M[n_node * n_node * n_node * n_node];
    REP4(theta1, phi1, theta2, phi2, n_node) {
                    M[n_node * n_node * n_node * theta1 + n_node * n_node * phi1 + n_node * theta2 + phi2] = f(
                            x[theta1], x[phi1], x[theta2], x[phi2]);
                }
    double U[n_node * n_node * n_node * n_node];
    double VT[n_node * n_node * n_node * n_node];
    double sigma[n_node * n_node];
    double buffer[n_node * n_node];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n_node * n_node, n_node * n_node, M, n_node * n_node, sigma,
                                  U, n_node * n_node, VT, n_node * n_node, buffer);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    double A[n_node][n_node][D], B[n_node][n_node][D];
    REP(k, D) {
        double s = std::sqrt(sigma[k]);
        REP(i, n_node)
            REP(j, n_node) {
                A[i][j][k] = U[n_node * n_node * n_node * i + n_node * n_node * j + k] * s;
                B[i][j][k] = VT[n_node * n_node * k + n_node * i + j] * s;
            }
    }
    REP4(i, j, k, l, D) {
                    REP(theta, n_node)
                        REP(phi, n_node) {
                            T[i][j][k][l] += A[theta][phi][i] * A[theta][phi][j] * B[theta][phi][k] * B[theta][phi][l] *
                                             w[theta] * w[phi] * std::cos(M_PI * x[theta] / 2);
                        }
                }

    REP(n, N) {
        // Tを 1~10 に丸め込む
        double _min = std::abs(T[0][0][0][0]);
        REP4(i, j, k, l, D) {
                        if (std::abs(T[i][j][k][l]) > 0)
                            _min = std::min(_min, std::abs(T[i][j][k][l]));
                    }
        auto o = static_cast<MKL_INT>(std::floor(std::log10(_min)));
        REP4(i, j, k, l, D) {
                        REP(t, std::abs(o)) {
                            if (o > 0) {
                                T[i][j][k][l] /= 10;
                            } else {
                                T[i][j][k][l] *= 10;
                            }
                        }
                    }
        order[n] = o;

        TRG::solver(D, D_cut, T);
    }

    double Z = 0;
    REP(i, D)
        REP(j, D) {
            Z += T[i][j][i][j];
        }

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    cout << "計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << '\n';
    return Z;
}

int main() {
    /* inputs */
    MKL_INT N = 20;     // volume : 2^N
    MKL_INT n_node = 32;  // n_node
    MKL_INT D_cut = 32; // bond dimension

    /* calculation */
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    const string fileName =
            "gauss_quadrature_node" +
            std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) +
            ".txt";
    std::ofstream dataFile;
    dataFile.open(fileName, std::ios::trunc);
    double K_start = 5;
    double K_end = 10.01;
    double K = K_start; // inverse temperature
    while (K <= K_end) {
        cout << "K = " << std::fixed << std::setprecision(1) << K << "   ";
        MKL_INT order[N];
        double Z = log(Trace(K, D_cut, n_node, N, order));
        REP(i, N) Z /= 2; // 体積で割る
        REP(i, N) {
            double tmp = order[i] * log(10);
            REP(j, i) tmp /= 2;
            Z += tmp;
        }
        Z += log(M_PI / 8);
        dataFile << std::setprecision(10) << K << '\t' << Z << '\n';
        K += MESH;
    }
    dataFile.close();
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";

    return 0;
}
