#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <functional>
#include <legendre_zero_point.hpp>
#include <tensor.hpp>
#include <impure_tensor.hpp>
#include <HOTRG.hpp>
#include <gsl/gsl_specfunc.h>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N)REP(j, N)REP(k, N)REP(l, N)
#define REPxy(i, j, k, l, Dx, Dy) REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy)

#define MESH 1e-1
#define LINF 1e300

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void initTensor(const double &K, const int &n_node, const int &D_cut, int &D, Tensor &T, ImpureTensor &IMT) {
    std::vector<double> x = math::solver::legendre_zero_point(n_node);
//    std::vector<double> x(n_node);
    std::vector<double> p(n_node);
    std::vector<double> w(n_node);

//    std::ifstream GL_node;
//    GL_node.open("GL-node.txt", std::ios::in);
//    for (int i = 0; GL_node >> x[i] >> w[i]; ++i);
//    GL_node.close();

    REP(i, n_node) p[i] = gsl_sf_legendre_Pl(n_node - 1, x[i]);
    REP(i, n_node) w[i] = 2 * (1 - x[i] * x[i]) / (n_node * n_node * p[i] * p[i]);

    std::function<double(double, double, double, double)>
            f = [=](double theta1, double phi1, double theta2, double phi2) {
        std::function<double(double)> s = [=](double theta) { return std::sin(M_PI * theta / 2); };
        std::function<double(double)> c = [=](double theta) { return std::cos(M_PI * theta / 2); };
        return std::exp(K * (s(theta1) * s(theta2) + c(theta1) * c(theta2) * std::cos(M_PI * (phi1 - phi2))));
    };
    auto M = new double[n_node * n_node * n_node * n_node];
    REP4(theta1, phi1, theta2, phi2, n_node) {
                    M[n_node * n_node * n_node * theta1 + n_node * n_node * phi1 + n_node * theta2 + phi2]
                            = f(x[theta1], x[phi1], x[theta2], x[phi2]);
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
        REP(i, n_node)
            REP(j, n_node) {
                U[n_node * n_node * n_node * i + n_node * n_node * j + k] *= s;
                VT[n_node * n_node * k + n_node * i + j] *= s;
            }
    }
    REP4(i, j, k, l, D) {
                    double sum = 0;
                    double sum1 = 0;
                    double sum2 = 0;
                    double sum3 = 0;
                    REP(theta, n_node) {
                        const double cosTheta = std::cos(M_PI * x[theta] / 2);
                        const double sinTheta = std::sin(M_PI * x[theta] / 2);
                        REP(phi, n_node) {
                            const double cosPhi = std::cos(M_PI * x[phi]);
                            const double sinPhi = std::sin(M_PI * x[phi]);
                            const double a = U[n_node * n_node * n_node * theta + n_node * n_node * phi + i];
                            const double b = U[n_node * n_node * n_node * theta + n_node * n_node * phi + j];
                            const double c = VT[n_node * n_node * k + n_node * theta + phi];
                            const double d = VT[n_node * n_node * l + n_node * theta + phi];
                            const double t = a * b * c * d * w[theta] * w[phi] * cosTheta;
                            sum += t;
                            sum1 += t * cosTheta * cosPhi;
                            sum2 += -t * cosTheta * sinPhi;
                            sum3 += -t * sinTheta;
                        }
                    }
                    T(i, j, k, l) = sum;
                    IMT.tensors[0](i, j, k, l) = sum1;
                    IMT.tensors[1](i, j, k, l) = sum2;
                    IMT.tensors[2](i, j, k, l) = sum3;
                }
    delete[] M;
    delete[] U;
    delete[] VT;
    delete[] sigma;
    delete[] buffer;
}

int normalization(Tensor &T, ImpureTensor &originIMT, std::vector<ImpureTensor> &IMTs) {
    double _min = LINF;
    double _max = 0;
    int Dx = T.GetDx();
    int Dy = T.GetDy();
    bool isAllMerged = true;
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    double t = std::abs(T(i, j, k, l));
                    if (t > 0) {
                        _min = std::min(_min, t);
                        _max = std::max(_max, t);
                    }
                }
    for (ImpureTensor &IMT : IMTs) {
        if (!IMT.isMerged) {
            isAllMerged = false;
            continue;
        }
        for (Tensor &tensor : IMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            double t = std::abs(tensor(i, j, k, l));
                            if (t > 0) {
                                _min = std::min(_min, t);
                                _max = std::max(_max, t);
                            }
                        }
        }
    }
    if (!isAllMerged) {
        for (Tensor &tensor : originIMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            double t = std::abs(tensor(i, j, k, l));
                            if (t > 0) {
                                _min = std::min(_min, t);
                                _max = std::max(_max, t);
                            }
                        }
        }
    }
    auto o = static_cast<MKL_INT>(std::floor((std::log10(_min) + std::log10(_max)) / 2));
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    if (o > 0) {
                        REP(t, std::abs(o)) T(i, j, k, l) /= 10;
                    } else {
                        REP(t, std::abs(o)) T(i, j, k, l) *= 10;
                    }
                }
    if (!isAllMerged) {
        for (Tensor &tensor : originIMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            if (o > 0) {
                                REP(t, std::abs(o)) tensor(i, j, k, l) /= 10;
                            } else {
                                REP(t, std::abs(o)) tensor(i, j, k, l) *= 10;
                            }
                        }
        }
    }
    for (ImpureTensor &IMT : IMTs) {
        if (!IMT.isMerged) continue;
        for (Tensor &tensor : IMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            if (o > 0) {
                                REP(t, std::abs(o)) tensor(i, j, k, l) /= 10;
                            } else {
                                REP(t, std::abs(o)) tensor(i, j, k, l) *= 10;
                            }
                        }
        }
    }
    return o;
}

void Trace(const int n_data_point, double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N, std::ofstream &file) {
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    Tensor T(D, D, D_cut, D_cut);
    ImpureTensor originIMT(D, D, D_cut, D_cut);

    initTensor(K, n_node, D_cut, D, T, originIMT);

    std::vector<ImpureTensor> IMTs(n_data_point);

    auto order = new int[N];
    MKL_INT Dx = D, Dy = D;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        order[n - 1] = normalization(T, originIMT, IMTs);

        if (n <= N / 2) { // compress along x-axis
            cout << " compress along x-axis :" << std::flush;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            if (n <= n_data_point) {
                int d = 1;
                REP(i, n - 1) d *= 2;
                IMTs[n - 1] = ImpureTensor(d, originIMT);
                IMTs[n - 1].isMerged = true;
                for (int i = 0; i < 3; ++i) {
                    HOTRG::contractionX(D_cut, IMTs[n - 1].tensors[i], originIMT.tensors[i], U, "left");
                }
                for (int i = 0; i < n - 1; ++i) {
                    for (auto &tensor : IMTs[i].tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
                for (Tensor &tensor : originIMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            } else {
                isMerged = true;
                for (int i = 0; i < n_data_point; ++i) {
                    for (auto &tensor : IMTs[i].tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            }
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compress along y-axis
            cout << " compress along y-axis :" << std::flush;
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            for (int i = 0; i < n_data_point; ++i) {
                for (auto &tensor : IMTs[i].tensors) HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
            }
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        if (!isMerged) {
            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            cout << " 計算時間 " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << '\n';
            continue;
        }

        double Tr = 0;
        REP(i, Dx)REP(j, Dy) Tr += T(i, j, i, j);

        for (ImpureTensor &IMT : IMTs) {
            double Tr1 = 0, Tr2 = 0, Tr3 = 0;
            REP(i, Dx)REP(j, Dy) {
                    Tr1 += IMT.tensors[0](i, j, i, j);
                    Tr2 += IMT.tensors[1](i, j, i, j);
                    Tr3 += IMT.tensors[2](i, j, i, j);
                }
            double res = (Tr1 + Tr2 + Tr3) / Tr;
            IMT.corrs.push_back(res);
            cout << '\t' << std::fixed << std::setprecision(10) << res << std::flush;
        }
        cout << '\n';
    }
    for (ImpureTensor &IMT : IMTs) {
        file << IMT.distance;
        for (double corr : IMT.corrs) {
            file << '\t' << std::fixed << std::setprecision(10) << corr << std::flush;
        }
        file << '\n';
    }
    delete[] order;
}

int main() {
    /* inputs */
    MKL_INT N = 16;     // volume : 2^N
    MKL_INT n_node = 32;  // n_node
    MKL_INT D_cut = 12; // bond dimension
    double K = 1.9; // inverse temperature
    int n_data_point = 7; // number of d. d = 1, 2, 4, 8, 16, 32, 64, ...

    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
//    start = std::chrono::system_clock::now();
//    fileName = "new_2point_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
//    dataFile.open(fileName, std::ios::trunc);
//    Trace(n_data_point, K, D_cut, n_node, N, dataFile);
//    dataFile.close();
//    end = std::chrono::system_clock::now();
//    cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    /* vs D_cut */
    for (D_cut = 8; D_cut <= 20; D_cut += 4) {
        start = std::chrono::system_clock::now();
        cout << "---------- " << D_cut << " ----------\n";
        fileName = "new_2point_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
        dataFile.open(fileName, std::ios::trunc);
        Trace(n_data_point, K, D_cut, n_node, N, dataFile);
        dataFile.close();
        end = std::chrono::system_clock::now();
        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
    }

    /* vs n_node */
//    for (n_node = 8; n_node <= 32; n_node += 8) {
//        start = std::chrono::system_clock::now();
//        cout << "---------- " << n_node << " ----------\n";
//        fileName = "2point_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(n_data_point, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        end = std::chrono::system_clock::now();
//        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
//    }
    return 0;
}
