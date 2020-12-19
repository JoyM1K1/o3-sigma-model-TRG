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

    REP(i, n_node) {
        p[i] = gsl_sf_legendre_Pl(n_node - 1, x[i]);
    }
    REP(i, n_node) {
        w[i] = 2 * (1 - x[i] * x[i]) / (n_node * n_node * p[i] * p[i]);
    }

    std::function<double(double, double, double, double)> f = [&](double theta1, double phi1, double theta2, double phi2) {
        std::function<double(double)> s = [&](double theta) { return std::sin(M_PI * theta / 2); };
        std::function<double(double)> c = [&](double theta) { return std::cos(M_PI * theta / 2); };
        return std::exp(K * (s(theta1) * s(theta2) + c(theta1) * c(theta2) * std::cos(M_PI * (phi1 - phi2))));
    };
    BaseTensor M(n_node);
    M.forEach([&](int theta1, int phi1, int theta2, int phi2, double *m) {
        *m = f(x[theta1], x[phi1], x[theta2], x[phi2]);
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
    double sum;
    T.forEach([&](int i, int j, int k, int l, double *t) {
        sum = 0;
        REP(theta, n_node)REP(phi, n_node) {
                const double a = U[n_node * n_node * n_node * theta + n_node * n_node * phi + i];
                const double b = U[n_node * n_node * n_node * theta + n_node * n_node * phi + j];
                const double c = VT[n_node * n_node * k + n_node * theta + phi];
                const double d = VT[n_node * n_node * l + n_node * theta + phi];
                sum += a * b * c * d * w[theta] * w[phi] * std::cos(M_PI * x[theta] / 2);
            }
        *t = sum * M_PI / 8;
    });
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

    std::function<double(double, double, double, double)>
            f = [=](double theta1, double phi1, double theta2, double phi2) {
        std::function<double(double)> s = [=](double theta) { return std::sin(M_PI * theta / 2); };
        std::function<double(double)> c = [=](double theta) { return std::cos(M_PI * theta / 2); };
        return std::exp(K * (s(theta1) * s(theta2) + c(theta1) * c(theta2) * std::cos(M_PI * (phi1 - phi2))));
    };
    BaseTensor M(n_node);
    M.forEach([&](int theta1, int phi1, int theta2, int phi2, double *m) {
        *m = f(x[theta1], x[phi1], x[theta2], x[phi2]);
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
    T.forEach([&](int i, int j, int k, int l, double *t) {
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
                const double p = a * b * c * d * w[theta] * w[phi] * cosTheta;
                sum += p;
                sum1 += p * cosTheta * cosPhi;
                sum2 += -p * cosTheta * sinPhi;
                sum3 += -p * sinTheta;
            }
        }
        *t = sum * M_PI / 8;
        IMT.tensors[0](i, j, k, l) = sum1 * M_PI / 8;
        IMT.tensors[1](i, j, k, l) = sum2 * M_PI / 8;
        IMT.tensors[2](i, j, k, l) = sum3 * M_PI / 8;
    });
    delete[] U;
    delete[] VT;
    delete[] sigma;
    delete[] buffer;
}

template void GaussQuadrature::initTensorWithImpure(const double &K, const int &n_node, const int &D_cut, const int &D, TRG::Tensor &T, BaseImpureTensor<TRG::Tensor> &IMT);
template void GaussQuadrature::initTensorWithImpure(const double &K, const int &n_node, const int &D_cut, const int &D, HOTRG::Tensor &T, BaseImpureTensor<HOTRG::Tensor> &IMT);
