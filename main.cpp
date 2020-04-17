#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <complex>
#include <mkl.h>
#include <mkl_scalapack.h>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) \
    REP(i, N)               \
    REP(j, N)               \
    REP(k, N)               \
    REP(l, N)

using namespace std;

// m x n 行列を出力
template <typename T>
void print_matrix(T *matrix, MKL_INT m, MKL_INT n, MKL_INT lda, string message)
{
    cout << '\n'
         << message << '\n';
    REP(i, m)
    {
        REP(j, n)
        {
            cout << scientific << setprecision(5) << (matrix[i * lda + j] >= 0 ? " " : "") << matrix[i * lda + j] << ' ';
        }
        cout << '\n';
    }
    cout << '\n';
}

double Z_SVD(double K, MKL_INT D_cut, MKL_INT N, MKL_INT &order)
{
    chrono::system_clock::time_point start = chrono::system_clock::now();
    // index dimension
    MKL_INT D = 2;

    // initialize tensor network : max index size is D_cut
    double T[D_cut][D_cut][D_cut][D_cut];
    REP4(i, j, k, l, D_cut)
    {
        T[i][j][k][l] = 0;
    }
    REP4(i, j, k, l, D)
    {
        T[i][j][k][l] = pow((tanh(K)), (i + j + k + l) / 2) * ((i + j + k + l + 1) % 2);
    }

    REP(n, N)
    {
        // Tを1以下に丸め込む
        MKL_INT c = 0;
        REP4(i, j, k, l, D)
        {
            if (T[i][j][k][l] <= 10)
                continue;
            MKL_INT count = 1;
            double t = T[i][j][k][l] / 10;
            while (t > 10)
            {
                t /= 10;
                count++;
            }
            c = max(c, count);
        }
        cout << '\n';
        order += c;
        cout << n + 1 << "-th order : " << order << '\n';

        MKL_UINT64 div = pow(10, c);
        REP4(i, j, k, l, D)
        {
            T[i][j][k][l] /= div;
        }

        MKL_INT D_new = min(D * D, D_cut);
        double Ma[D * D * D * D], Mb[D * D * D * D]; // Ma = M(ij)(kl)  Mb = M(jk)(li)
        REP(i, D * D * D * D)
        {
            Ma[i] = 0;
            Mb[i] = 0;
        }
        REP4(i, j, k, l, min(D, D_cut))
        {
            Ma[l + D * k + D * D * j + D * D * D * i] = T[i][j][k][l];
            Mb[i + D * l + D * D * k + D * D * D * j] = T[i][j][k][l];
        }
        double sigma[D * D];
        double U[D * D * D * D], VH[D * D * D * D];
        double S1[D][D][D_new], S2[D][D][D_new], S3[D][D][D_new], S4[D][D][D_new];
        double superb[D * D];
        if (n < 0)
        {
            print_matrix(Ma, D * D, D * D, D * D, "Ma");
            print_matrix(Mb, D * D, D * D, D * D, "Mb");
        }
        MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, Ma, D * D, sigma, U, D * D, VH, D * D, superb); // Ma = U * sigma * VH
        if (info > 0)
        {
            cout << "The algorithm computing SVD failed to converge.\n";
            return 1;
        }
        if (n < 0)
        {
            print_matrix(sigma, 1, D * D, D * D, "sigma");
            print_matrix(U, D * D, D * D, D * D, "U");
            print_matrix(VH, D * D, D * D, D * D, "VH");
            double US[D * D * D * D];
            REP4(i, j, k, l, D)
            {
                US[i + D * j + D * D * k + D * D * D * l] = U[i + D * j + D * D * k + D * D * D * l] * sigma[i + D * j];
            }
            double USVH[D * D * D * D];
            REP(i, D * D * D * D)
            {
                USVH[i] = 0;
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D * D, D * D, D * D, 1, US, D * D, VH, D * D, 0, USVH, D * D);
            print_matrix(USVH, D * D, D * D, D * D, "U * sigma * VH");
        }
        REP(i, D)
        {
            REP(j, D)
            {
                REP(k, D_new)
                {
                    double s = sqrt(sigma[k]);
                    S1[i][j][k] = s * U[k + D * D * j + D * D * D * i];
                    S3[i][j][k] = s * VH[j + D * i + D * D * k];
                }
            }
        }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, Mb, D * D, sigma, U, D * D, VH, D * D, superb);
        if (info > 0)
        {
            cout << "The algorithm computing SVD failed to converge.\n";
            return 1;
        }
        if (n < 0)
        {
            print_matrix(sigma, 1, D * D, D * D, "sigma");
            print_matrix(U, D * D, D * D, D * D, "U");
            print_matrix(VH, D * D, D * D, D * D, "VH");
        }
        REP(i, D)
        {
            REP(j, D)
            {
                REP(k, D_new)
                {
                    double s = sqrt(sigma[k]);
                    S2[i][j][k] = s * U[k + D * D * j + D * D * D * i];
                    S4[i][j][k] = s * VH[j + D * i + D * D * k];
                }
            }
        }

        double T_new[D_cut][D_cut][D_cut][D_cut];

        double X12[D_new][D_new][D][D], X34[D_new][D_new][D][D];
        REP(i, D_new)
        {
            REP(j, D_new)
            {
                REP(b, D)
                {
                    REP(d, D)
                    {
                        X12[i][j][b][d] = 0;
                        X34[i][j][b][d] = 0;
                        REP(a, D)
                        {
                            X12[i][j][b][d] += S1[a][d][i] * S2[b][a][j];
                            X34[i][j][b][d] += S3[a][b][i] * S4[d][a][j];
                        }
                    }
                }
            }
        }

        REP4(i, j, k, l, D_new)
        {
            T_new[i][j][k][l] = 0;
            REP(b, D)
            {
                REP(d, D)
                {
                    T_new[i][j][k][l] += X12[k][l][b][d] * X34[i][j][b][d];
                }
            }
        }

        // 更新
        D = D_new;
        order *= 2;
        REP4(i, j, k, l, D)
        {
            T[i][j][k][l] = T_new[i][j][k][l];
        }
    }

    double Z = 0;
    REP(i, D)
    REP(j, D)
    {
        Z += T[i][j][i][j];
    }

    chrono::system_clock::time_point end = chrono::system_clock::now();
    cout << "計算時間 : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << '\n';
    return Z;
}

int main()
{
    // inputs
    double K;      // inverse temperature
    MKL_INT D_cut; // bond dimension
    MKL_INT N;     // repeat count

    // K = log(sqrt(2) + 1) / 2; // critical value
    // D_cut = 8;

    cout << "input K : ";
    cin >> K;
    cout << "input bond dimension : ";
    cin >> D_cut;
    cout << "input N : ";
    cin >> N;
    cout << '\n';

    MKL_INT order = 0;

    double Z = log(Z_SVD(K, D_cut, N * 2, order));
    double V = pow(4, N);

    Z = log(2) + 2 * log(cosh(K)) + order * log(10) / V + Z / V;

    cout << fixed << setprecision(9) << "log(Z)/V : " << Z << '\n';

    return 0;
}
