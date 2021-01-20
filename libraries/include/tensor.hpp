#ifndef O3_SIGMA_MODEL_TENSOR_HPP
#define O3_SIGMA_MODEL_TENSOR_HPP

#include <functional>
#include <cassert>
#include <tuple>
#include <array>

#define REP(i, N) for (int i = 0; i < (N); ++i)

class BaseTensor {
private:
    int Di{0}, Dj{0}, Dk{0}, Dl{0}, D_max{0};
public:
    double *array{nullptr};
    long long int order{0};

    BaseTensor() = default;

    explicit BaseTensor(int D);

    BaseTensor(int D, int D_max);

    BaseTensor(int Di, int Dj, int Dk, int Dl);

    BaseTensor(int Di, int Dj, int Dk, int Dl, int D_max);

    BaseTensor(BaseTensor &rhs);

    ~BaseTensor();

    int GetDx() const;

    int GetDy() const;

    int GetDi() const;

    int GetDj() const;

    int GetDk() const;

    int GetDl() const;

    int GetD_max() const;

    void UpdateDx(int Dx);

    void UpdateDy(int Dy);

    void SetDi(int Di_);

    void SetDj(int Dj_);

    void SetDk(int Dk_);

    void SetDl(int Dl_);

    BaseTensor &operator=(const BaseTensor &rhs);

    const double &operator()(int i, int j, int k, int l) const;

    double &operator()(int i, int j, int k, int l);

    void forEach(const std::function<void(double *)> &f);

    void forEach(const std::function<void(int, int, int, int, double *)> &f);

    long long int normalization(int c);

    double trace();
};

template<int N>
class tensor {
private:
    int D[N]{};
    int D_prod[N]{};
    std::array<int, N> *indices{nullptr};
    int array_size{0};
public:
    double *array{nullptr};

    tensor();

    template<typename ...Args>
    tensor(Args ...args);

    ~tensor();

    template<typename ...Args>
    const double &operator()(Args ...args) const;

    template<typename ...Args>
    double &operator()(Args ...args);

    void forEach(const std::function<void(double *, std::array<int, N> *)> &f);
};

template<int N>
inline tensor<N>::tensor() {
    array_size = 0;
    array[0] = 0;
    for (auto &d : D) d = 0;
    for (auto &d : D_prod) d = 0;
}

template<int N>
template<typename ...Args>
inline tensor<N>::tensor(Args ...args) {
    static_assert(sizeof...(args) == N, "Initializer list size must be equal to tensor rank.");
    int i = 0;
    int buffer = 1;
    for (auto &d : {args...}) {
        D[i] = d;
        buffer *= d;
        i++;
    }
    array_size = buffer;
    array = new double[array_size];
    indices = new std::array<int, N>[array_size];
    i = 0;
    for (auto &d : {args...}) {
        buffer /= d;
        D_prod[i] = buffer;
        i++;
    }
    REP(a, array_size) array[a] = 0;

    REP(n, N) {
        REP(a, array_size) {
            indices[a][n] = (a / D_prod[n]) % D[n];
        }
    }
}

template<int N>
inline tensor<N>::~tensor() {
    delete[] array;
    delete[] indices;
}

template<int N>
template<typename ...Args>
inline const double &tensor<N>::operator()(Args ...args) const {
    static_assert(sizeof...(args) == N, "operator '()' args' size must be equal to tensor rank.");
    int index = 0;
    int i = 0;
    for (auto &leg : {args...}) {
        assert(typeid(leg) == typeid(int));
        assert(0 <= leg && leg <= D[i]);
        index += D_prod[i] * leg;
        i++;
    }
    return array[index];
}

template<int N>
template<typename ...Args>
inline double &tensor<N>::operator()(Args ...args) {
    static_assert(sizeof...(args) == N, "operator '()' args' size must be equal to tensor rank.");
    int index = 0;
    int i = 0;
    for (auto &leg : {args...}) {
        assert(typeid(leg) == typeid(int));
        assert(0 <= leg && leg <= D[i]);
        index += D_prod[i] * leg;
        i++;
    }
    return array[index];
}

template<int N>
inline void tensor<N>::forEach(const std::function<void(double *, std::array<int, N> *)> &f) {
    REP(i, array_size) {
        f(&array[i], &indices[i]);
    }
}

#endif //O3_SIGMA_MODEL_TENSOR_HPP
