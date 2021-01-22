#include <iostream>
#include <cmath>
#include "../include/frac.hpp"

frac::frac(long long n, long long d) {
    num = n;
    den = d;
    reduce();
}

double frac::toDouble() const {
    return static_cast<double>(num) / static_cast<double>(den);
}

long long frac::gcd(long long x, long long y) const {
    x = std::abs(x);
    y = std::abs(y);
    return y ? gcd(y, x % y) : x;
}

void frac::reduce() {
    if (num == 0) {
        den = 1;
        return;
    }
    if (den < 0) { // 分母は常に正
        den *= -1;
        num *= -1;
    }
    long long g = gcd(std::abs(num), std::abs(den));
    num /= g;
    den /= g;
}

frac frac::abs(frac a) {
    frac res(std::abs(a.num), std::abs(a.den));
    res.reduce();
    return res;
}

frac frac::sign() const {
    frac res(num, den);
    res.reduce();
    return (res / frac::abs(res));
}

frac frac::max(frac a, frac b) {
    return a > b ? a : b;
}

frac frac::min(frac a, frac b) {
    return a < b ? a : b;
}

//void frac::swap(frac &a, frac &b) {
//    frac tmp = a;
//    a = b;
//}

frac frac::cleanSquareRoot() const {
    if (num <= 0 || den <= 0)
        return frac(0);
    for (long long i = 0; i * i <= num; ++i)
        for (long long j = 0; j * j <= den; ++j) {
            if (i * i == num && j * j == den)
                return frac(i, j);
        }
    return frac(0);
}

bool frac::isSquare() const {
    frac res = cleanSquareRoot();
    return res * res == *this;
}

frac &frac::operator=(const frac &rhs) {
    if (this == &rhs)
        return *this;
    (*this).den = rhs.den;
    (*this).num = rhs.num;
    (*this).reduce();
    return *this;
}

frac &frac::operator=(const long long rhs) {
    num = rhs;
    den = 1;
    return *this;
}

bool frac::operator==(const frac &rhs) const {
    frac A(num, den);
    frac B(rhs.num, rhs.den);
    A.reduce();
    B.reduce();
    return A.num == B.num && A.den == B.den;
}

bool frac::operator==(const long long rhs) const {
    frac R(rhs);
    return *this == R;
}

bool frac::operator!=(const frac &rhs) const {
    return !(*this == rhs);
}

bool frac::operator!=(const long long rhs) const {
    frac R(rhs);
    return *this != R;
}

bool frac::operator<(const frac &rhs) const {
    long long g = gcd(den, rhs.den);
    long long res = num * (rhs.den / g) - rhs.num * (den / g);
    return res < 0;
}

bool frac::operator<(const long long rhs) const {
    return *this < frac(rhs);
}

bool frac::operator>=(const frac &rhs) const {
    return !(*this < rhs);
}

bool frac::operator>=(const long long rhs) const {
    return *this >= frac(rhs);
}

bool frac::operator>(const frac &rhs) const {
    return *this >= rhs && *this != rhs;
}

bool frac::operator>(const long long rhs) const {
    return *this > frac(rhs);
}

bool frac::operator<=(const frac &rhs) const {
    return !(*this > rhs);
}

bool frac::operator<=(const long long rhs) const {
    return *this <= frac(rhs);
}

frac frac::operator+(const frac &rhs) const {
    long long g = gcd(den, rhs.den);
    long long l = den / g;
    long long r = rhs.den / g;
    frac res(num * r + rhs.num * l, g * l * r);
    res.reduce();
    return res;
}

frac frac::operator+(const long long rhs) const {
    frac L(num, den), R(rhs);
    frac res = L + R;
    res.reduce();
    return res;
}

frac frac::operator-(const frac &rhs) const {
    long long g = gcd(den, rhs.den);
    long long l = den / g;
    long long r = rhs.den / g;
    frac res(num * r - rhs.num * l, g * l * r);
    res.reduce();
    return res;
}

frac frac::operator-(const long long rhs) const {
    frac L(num, den), R(rhs);
    frac res = L - R;
    res.reduce();
    return res;
}

frac frac::operator*(const frac &rhs) const {
    frac res(num * rhs.num, den * rhs.den);
    res.reduce();
    return res;
}

frac frac::operator*(const long long rhs) const {
    frac L(num, den), R(rhs);
    frac res = L * R;
    res.reduce();
    return res;
}

frac frac::operator/(const frac &rhs) const {
    frac res(num * rhs.den, den * rhs.num);
    res.reduce();
    return res;
}

frac frac::operator/(const long long rhs) const {
    frac L(num, den), R(rhs);
    frac res = L / R;
    res.reduce();
    return res;
}

frac frac::operator%(const long long rhs) const {
    frac res(num % (rhs * den), den);
    res.reduce();
    return res;
}

frac &frac::operator+=(const frac &rhs) {
    return *this = *this + rhs;
}

frac &frac::operator+=(const long long rhs) {
    return *this = *this + rhs;
}

frac &frac::operator-=(const frac &rhs) {
    return *this = *this - rhs;
}

frac &frac::operator-=(const long long rhs) {
    return *this = *this - rhs;
}

frac &frac::operator*=(const frac &rhs) {
    return *this = *this * rhs;
}

frac &frac::operator*=(const long long rhs) {
    return *this = *this * rhs;
}

frac &frac::operator/=(const frac &rhs) {
    return *this = *this / rhs;
}

frac &frac::operator/=(const long long rhs) {
    return *this = *this / rhs;
}

frac &frac::operator++() {
    return *this += 1;
}

frac &frac::operator--() {
    return *this -= 1;
}

frac operator-(const frac &unary) {
    frac res(-unary.num, unary.den);
    res.reduce();
    return res;
}

std::ostream &operator<<(std::ostream &out, const frac &rhs) {
    out << rhs.num;
    if (rhs.den != 1) {
        out << '/' << rhs.den;
    }
    return out;
}
