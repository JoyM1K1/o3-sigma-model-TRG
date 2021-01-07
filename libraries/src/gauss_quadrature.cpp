#include "../include/gauss_quadrature.hpp"
#include "../include/legendre_zero_point.hpp"
#include <iostream>
#include <vector>
#include <functional>
#include <gsl/gsl_specfunc.h>
#include <cmath>
#include <mkl.h>
//#include <fstream>

#define REP(i, N) for (int i = 0; i < (N); ++i)

void GaussQuadrature::initTensor(const double &K, const int &n_node, const int &D_cut, BaseTensor &T) {
    const int D = std::min(n_node * n_node, D_cut);
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

    BaseTensor M(n_node);
    std::function<double(int)> si = [&](int theta) { return std::sin(M_PI * x[theta] / 2); };
    std::function<double(int)> co = [&](int theta) { return std::cos(M_PI * x[theta] / 2); };
    M.forEach([&](int theta1, int phi1, int theta2, int phi2, double *m) {
        *m = std::exp(K * (si(theta1) * si(theta2) + co(theta1) * co(theta2) * std::cos(M_PI * (x[phi1] - x[phi2]))));
    });
    auto U = new double[n_node * n_node * n_node * n_node];
    auto VT = new double[n_node * n_node * n_node * n_node];
    auto sigma = new double[n_node * n_node];
    auto buffer = new double[n_node * n_node - 1];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n_node * n_node, n_node * n_node, M.GetMatrix(), n_node * n_node,
                                  sigma, U, n_node * n_node, VT, n_node * n_node, buffer);
    if (info > 0) {
        std::cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    REP(k, D) {
        double s = std::sqrt(sigma[k]);
        REP(i, n_node)REP(j, n_node) {
            U[n_node * n_node * n_node * i + n_node * n_node * j + k] *= s;
            VT[n_node * n_node * k + n_node * i + j] *= s;
        }
    }
    BaseTensor X(D, D, n_node, n_node);
    BaseTensor Y(n_node, n_node, D, D);
    X.forEach([&](int i, int j, int a, int b, double *t) {
        *t = U[n_node * n_node * n_node * a + n_node * n_node * b + i] * U[n_node * n_node * n_node * a + n_node * n_node * b + j] * w[a] * w[b] * co(a) * M_PI / 8;
    });
    Y.forEach([&](int a, int b, int k, int l, double *t) {
        *t = VT[n_node * n_node * k + n_node * a + b] * VT[n_node * n_node * l + n_node * a + b];
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D * D, D * D, n_node * n_node, 1, X.GetMatrix(),
            n_node * n_node, Y.GetMatrix(), D * D, 0, T.GetMatrix(), D * D);
    delete[] U;
    delete[] VT;
    delete[] sigma;
    delete[] buffer;
}

template<class Tensor>
void GaussQuadrature::initTensorWithImpure(const double &K, const int &n_node, const int &D_cut, const int &D, Tensor &T, BaseImpureTensor<Tensor> &IMT) {
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

    std::function<double(int)> si = [&](int theta) { return std::sin(M_PI * x[theta] / 2); };
    std::function<double(int)> co = [&](int theta) { return std::cos(M_PI * x[theta] / 2); };

    BaseTensor M(n_node);
    M.forEach([&](int theta1, int phi1, int theta2, int phi2, double *m) {
        *m = std::exp(K * (si(theta1) * si(theta2) + co(theta1) * co(theta2) * std::cos(M_PI * (x[phi1] - x[phi2]))));
    });
    auto U = new double[n_node * n_node * n_node * n_node];
    auto VT = new double[n_node * n_node * n_node * n_node];
    auto sigma = new double[n_node * n_node];
    auto buffer = new double[n_node * n_node - 1];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n_node * n_node, n_node * n_node, M.GetMatrix(), n_node * n_node,
                                  sigma, U, n_node * n_node, VT, n_node * n_node, buffer);
    if (info > 0) {
        std::cerr << "The algorithm computing SVD failed to converge.\n";
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
    BaseTensor X(D, D, n_node, n_node);
    BaseTensor Y(n_node, n_node, D, D);
    /* pure tensor */
    X.forEach([&](int i, int j, int a, int b, double *t) {
        *t = U[n_node * n_node * n_node * a + n_node * n_node * b + i] * U[n_node * n_node * n_node * a + n_node * n_node * b + j] * w[a] * w[b] * co(a) * M_PI / 8;
    });
    Y.forEach([&](int a, int b, int k, int l, double *t) {
        *t = VT[n_node * n_node * k + n_node * a + b] * VT[n_node * n_node * l + n_node * a + b];
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D * D, D * D, n_node * n_node, 1, X.GetMatrix(),
            n_node * n_node, Y.GetMatrix(), D * D, 0, T.GetMatrix(), D * D);
    /* impure tensor x */
    X.forEach([&](int i, int j, int a, int b, double *t) {
        *t = U[n_node * n_node * n_node * a + n_node * n_node * b + i] * U[n_node * n_node * n_node * a + n_node * n_node * b + j] * w[a] * w[b] * co(a) * co(a) * std::cos(M_PI * x[b]) * M_PI / 8;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D * D, D * D, n_node * n_node, 1, X.GetMatrix(),
            n_node * n_node, Y.GetMatrix(), D * D, 0, IMT.tensors[0].GetMatrix(), D * D);
    /* impure tensor y */
    X.forEach([&](int i, int j, int a, int b, double *t) {
        *t = - U[n_node * n_node * n_node * a + n_node * n_node * b + i] * U[n_node * n_node * n_node * a + n_node * n_node * b + j] * w[a] * w[b] * co(a) * co(a) * std::sin(M_PI * x[b]) * M_PI / 8;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D * D, D * D, n_node * n_node, 1, X.GetMatrix(),
            n_node * n_node, Y.GetMatrix(), D * D, 0, IMT.tensors[1].GetMatrix(), D * D);
    /* impure tensor z */
    X.forEach([&](int i, int j, int a, int b, double *t) {
        *t = - U[n_node * n_node * n_node * a + n_node * n_node * b + i] * U[n_node * n_node * n_node * a + n_node * n_node * b + j] * w[a] * w[b] * co(a) * si(a) * M_PI / 8;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D * D, D * D, n_node * n_node, 1, X.GetMatrix(),
            n_node * n_node, Y.GetMatrix(), D * D, 0, IMT.tensors[2].GetMatrix(), D * D);
    delete[] U;
    delete[] VT;
    delete[] sigma;
    delete[] buffer;
}

template void GaussQuadrature::initTensorWithImpure(const double &K, const int &n_node, const int &D_cut, const int &D, TRG::Tensor &T, BaseImpureTensor<TRG::Tensor> &IMT);
template void GaussQuadrature::initTensorWithImpure(const double &K, const int &n_node, const int &D_cut, const int &D, HOTRG::Tensor &T, BaseImpureTensor<HOTRG::Tensor> &IMT);
