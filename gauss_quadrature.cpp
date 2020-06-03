#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <vector>
#include <mkl.h>
#include <fstream>
#include "libraries/include/legendre_zero_point.hpp"

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1

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
TRG(double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N, MKL_INT *order) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

//    std::vector<double> x = math::solver::legendre_zero_point(n_node);
    std::vector<double> x(n_node);
//    std::vector<double> p(n_node);
    std::vector<double> w(n_node);

    std::ifstream GL_node;
    GL_node.open("GL-node.txt", std::ios::in);
    for (int i = 0; GL_node >> x[i] >> w[i]; ++i) {
        cout << x[i] << ' ' << w[i] << '\n';
    }
    GL_node.close();

//    REP(i, n_node) {
//        p[i] = std::legendre(n_node - 1, x[i]);
//    }
//    REP(i, n_node) {
//        w[i] = 2 * (1 - x[i] * x[i]) / (n_node * n_node * p[i] * p[i]);
//    }

    // initialize tensor network : max index size is D_cut
    double T[D_cut][D_cut][D_cut][D_cut];
    REP4(i, j, k, l, D_cut) {
                    T[i][j][k][l] = 0;
                }
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
    double u[n_node * n_node * n_node * n_node];
    double vt[n_node * n_node * n_node * n_node];
    double sig[n_node * n_node];
    double buffer[n_node * n_node];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n_node * n_node, n_node * n_node, M, n_node * n_node, sig,
                                  u, n_node * n_node, vt, n_node * n_node, buffer);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    double A[n_node][n_node][D], B[n_node][n_node][D];
    REP(k, D) {
        double s = std::sqrt(sig[k]);
        REP(i, n_node)
            REP(j, n_node) {
                A[i][j][k] = u[n_node * n_node * n_node * i + n_node * n_node * j + k] * s;
                B[i][j][k] = vt[n_node * n_node * k + n_node * i + j] * s;
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

        MKL_INT D_new = std::min(D * D, D_cut);
        double Ma[D * D * D * D], Mb[D * D * D * D]; // Ma = M(ij)(kl)  Mb = M(jk)(li)
        REP(i, D * D * D * D) {
            Ma[i] = 0;
            Mb[i] = 0;
        }
        REP4(i, j, k, l, std::min(D, D_cut)) {
                        Ma[l + D * k + D * D * j + D * D * D * i] = T[i][j][k][l];
                        Mb[i + D * l + D * D * k + D * D * D * j] = T[i][j][k][l];
                    }
        double sigma[D * D];
        double U[D * D * D * D], VH[D * D * D * D];
        double S1[D][D][D_new], S2[D][D][D_new], S3[D][D][D_new], S4[D][D][D_new];
        double superb[D * D];
        if (n < 0) {
            print_matrix(Ma, D * D, D * D, D * D, "Ma");
//            print_matrix(Mb, D * D, D * D, D * D, "Mb");
        }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, Ma, D * D, sigma, U, D * D, VH, D * D,
                              superb); // Ma = U * sigma * VH
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            return 1;
        }
        if (n < 0) {
            print_matrix(sigma, 1, D * D, D * D, "sigma");
            print_matrix(U, D * D, D * D, D * D, "U");
//            print_matrix(VH, D * D, D * D, D * D, "VH");
//            double US[D * D * D * D];
//            REP4(i, j, k, l, D) {
//                            US[i + D * j + D * D * k + D * D * D * l] =
//                                    U[i + D * j + D * D * k + D * D * D * l] * sigma[i + D * j];
//                        }
//            double USVH[D * D * D * D];
//            REP(i, D * D * D * D) {
//                USVH[i] = 0;
//            }
//            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D * D, D * D, D * D, 1, US, D * D, VH, D * D, 0,
//                        USVH, D * D);
//            print_matrix(USVH, D * D, D * D, D * D, "U * sigma * VH");
        }
        REP(i, D) {
            REP(j, D) {
                REP(k, D_new) {
                    double s = sqrt(sigma[k]);
                    S1[i][j][k] = s * U[k + D * D * j + D * D * D * i];
                    S3[i][j][k] = s * VH[j + D * i + D * D * k];
                }
            }
        }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, Mb, D * D, sigma, U, D * D, VH, D * D, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            return 1;
        }
        if (n < 0) {
            print_matrix(sigma, 1, D * D, D * D, "sigma");
            print_matrix(U, D * D, D * D, D * D, "U");
//            print_matrix(VH, D * D, D * D, D * D, "VH");
        }
        REP(i, D) {
            REP(j, D) {
                REP(k, D_new) {
                    double s = sqrt(sigma[k]);
                    S2[i][j][k] = s * U[k + D * D * j + D * D * D * i];
                    S4[i][j][k] = s * VH[j + D * i + D * D * k];
                }
            }
        }

        double X12[D_new][D_new][D][D], X34[D_new][D_new][D][D];
        REP(i, D_new) {
            REP(j, D_new) {
                REP(b, D) {
                    REP(d, D) {
                        X12[i][j][b][d] = 0;
                        X34[i][j][b][d] = 0;
                        REP(a, D) {
                            X12[i][j][b][d] += S1[a][d][i] * S2[b][a][j];
                            X34[i][j][b][d] += S3[a][b][i] * S4[d][a][j];
                        }
                    }
                }
            }
        }

        REP4(i, j, k, l, D_new) {
                        T[i][j][k][l] = 0;
                        REP(b, D) {
                            REP(d, D) {
                                T[i][j][k][l] += X12[k][l][b][d] * X34[i][j][b][d];
                            }
                        }
                    }

        // 更新
        D = D_new;
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
            "gauss_quadrature_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) +
            ".txt";
    std::ofstream dataFile;
    dataFile.open(fileName, std::ios::trunc);
    double K_start = 4.0;
    double K_end = 4.01;
    double K = K_start; // inverse temperature
    while (K <= K_end) {
        cout << "K = " << K << "   ";
        MKL_INT order[N];
        double Z = log(TRG(K, D_cut, n_node, N, order));
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
