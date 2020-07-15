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
#include <HOTRG.hpp>
#include <tensor.hpp>
#include <gsl/gsl_specfunc.h>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1
#define LINF 1e300

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void initTensor(const double K, const int &n_node, const int &D_cut, int &D, Tensor &T) {
    std::vector<double> x = math::solver::legendre_zero_point(n_node);
//    std::vector<double> x(n_node);
    std::vector<double> p(n_node);
    std::vector<double> w(n_node);

//    std::ifstream GL_node;
//    GL_node.open("GL-node.txt", std::ios::in);
//    for (int i = 0; GL_node >> x[i] >> w[i]; ++i);
//    GL_node.close();

    REP(i, n_node) {
        p[i] = gsl_sf_legendre_Pl(n_node - 1, x[i]);
    }
    REP(i, n_node) {
        w[i] = 2 * (1 - x[i] * x[i]) / (n_node * n_node * p[i] * p[i]);
    }

    std::function<double(double, double, double, double)> f = [=](double theta1, double phi1, double theta2, double phi2) {
        std::function<double(double)> s = [=](double theta) { return std::sin(M_PI * theta / 2); };
        std::function<double(double)> c = [=](double theta) { return std::cos(M_PI * theta / 2); };
        return std::exp(K * (s(theta1) * s(theta2) + c(theta1) * c(theta2) * std::cos(M_PI * (phi1 - phi2))));
    };
    auto M = new double[n_node * n_node * n_node * n_node];
    REP4(theta1, phi1, theta2, phi2, n_node) {
                    M[n_node * n_node * n_node * theta1 + n_node * n_node * phi1 + n_node * theta2 + phi2] = f(
                            x[theta1], x[phi1], x[theta2], x[phi2]);
                }
    auto U = new double[n_node * n_node * n_node * n_node];
    auto VT = new double[n_node * n_node * n_node * n_node];
    auto sigma = new double[n_node * n_node];
    auto buffer = new double[n_node * n_node - 1];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n_node * n_node, n_node * n_node, M, n_node * n_node,
                                  sigma, U, n_node * n_node, VT, n_node * n_node, buffer);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    REP(k, D) {
        double s = std::sqrt(sigma[k]);
        REP(i, n_node)REP(j, n_node) {
                U[n_node * n_node * n_node * i + n_node * n_node * j + k] *= s;
                VT[n_node * n_node * k + n_node * i + j] *= s;
            }
    }
    double sum;
    REP4(i, j, k, l, D) {
                    sum = 0;
                    REP(theta, n_node)REP(phi, n_node) {
                            const double a = U[n_node * n_node * n_node * theta + n_node * n_node * phi + i];
                            const double b = U[n_node * n_node * n_node * theta + n_node * n_node * phi + j];
                            const double c = VT[n_node * n_node * k + n_node * theta + phi];
                            const double d = VT[n_node * n_node * l + n_node * theta + phi];
                            sum += a * b * c * d * w[theta] * w[phi] * std::cos(M_PI * x[theta] / 2);
                        }
                    T(i, j, k, l) = sum;
                }
    delete[] M;
    delete[] U;
    delete[] VT;
    delete[] sigma;
    delete[] buffer;
}

int normalization(Tensor &T) {
    const int Dx = T.GetDx();
    const int Dy = T.GetDy();
    // Tを 1~10 に丸め込む
    double _min = LINF;
    double _max = 0;
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    const double t = T(i, j, k, l);
                    if (std::abs(t) > 0) {
                        _min = std::min(_min, std::abs(t));
                        _max = std::max(_max, std::abs(t));
                    }
                }
//    cout << std::scientific << std::setprecision(2) << _min << ' ' << _max << '\n';
    auto o = static_cast<MKL_INT>(std::floor((std::log10(_min) + std::log10(_max)) / 2));
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    if (o > 0) {
                        REP(t, std::abs(o)) T(i, j, k, l) /= 10;
                    } else {
                        REP(t, std::abs(o)) T(i, j, k, l) *= 10;
                    }
                }
    return o;
}

void Trace(double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N, std::ofstream &file) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    Tensor T(D, D, D_cut, D_cut);
    initTensor(K, n_node, D_cut, D, T);

    auto order = new MKL_INT[N];
    MKL_INT Dx = D, Dy = D;

    for (int n = 1; n <= N; ++n) {
        order[n - 1] = normalization(T);

        if (n <= 2 / N) { // compression along x-axis
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compression along y-axis
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        double Tr = 0;
        REP(i, Dx) {
            REP(j, Dy) {
                Tr += T(i, j, i, j);
            }
        }
        Tr = std::log(Tr);
        REP(i, n) Tr /= 2; // 体積で割る
        REP(i, n) {
            double tmp = order[i] * std::log(10);
            REP(j, i) tmp /= 2;
            Tr += tmp;
        }
        Tr += std::log(M_PI / 8);
        file << '\t' << std::fixed << std::setprecision(10) << Tr;
        cout << '\t' << std::fixed << std::setprecision(10) << Tr << std::flush;
    }
    delete[] order;
    file << '\n';
    cout << '\n';
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    cout << "計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << '\n';
}

int main() {
    /* inputs */
    MKL_INT N = 20;     // volume : 2^N
    MKL_INT n_node = 32;  // n_node
    MKL_INT D_cut = 16; // bond dimension

    double K_start = 0.1;
    double K_end = 4.01;
    double K = K_start; // inverse temperature

    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    start = std::chrono::system_clock::now();
    fileName = "gauss_quadrature_HOTRG_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    while (K <= K_end) {
        cout << "K = " << std::fixed << std::setprecision(1) << K << " : " << std::flush;
        dataFile << std::setprecision(1) << K;
        Trace(K, D_cut, n_node, N, dataFile);
        K += MESH;
    }
    dataFile.close();
    end = std::chrono::system_clock::now();
    cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    /* vs D_cut */
//    for (D_cut = 8; D_cut <= 24; D_cut += 4) {
//        K = K_start;
//        start = std::chrono::system_clock::now();
//        fileName =
//                "gauss_quadrature_HOTRG_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        while (K <= K_end) {
//            cout << "K = " << std::fixed << std::setprecision(1) << K << " : " << std::flush;
//            dataFile << std::setprecision(1) << K;
//            Trace(K, D_cut, n_node, N/*, dataFile*/);
//            K += MESH;
//        }
//        dataFile.close();
//        end = std::chrono::system_clock::now();
//        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
//    }

    return 0;
}
