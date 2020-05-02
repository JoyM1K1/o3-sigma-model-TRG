//
// Created by Joy on 2020/04/27.
//

#ifndef O3_SIGMA_MODEL_FRAC_HPP
#define O3_SIGMA_MODEL_FRAC_HPP

#include <iostream>
#include <cmath>

class frac {
public:
    long long num, den; // numerator, denominator
    explicit frac(long long n = 0, long long d = 1); // Constructor

    double toDouble() const;

    long long gcd(long long x, long long y) const;

    void reduce();

    frac sign() const;

    static frac abs(frac a);

    static frac max(frac a, frac b);

    static frac min(frac a, frac b);

    frac cleanSquareRoot() const;

    bool isSquare() const;

    // --------------- operator ---------------
    frac &operator=(const frac &rhs);

    frac &operator=(const long long rhs);

    bool operator==(const frac &rhs) const;

    bool operator==(const long long rhs) const;

    bool operator!=(const frac &rhs) const;

    bool operator!=(const long long rhs) const;

    bool operator<(const frac &rhs) const;

    bool operator<(const long long rhs) const;

    bool operator>=(const frac &rhs) const;

    bool operator>=(const long long rhs) const;

    bool operator>(const frac &rhs) const;

    bool operator>(const long long rhs) const;

    bool operator<=(const frac &rhs) const;

    bool operator<=(const long long rhs) const;

    frac operator+(const frac &rhs) const;

    frac operator+(const long long rhs) const;

    frac operator-(const frac &rhs) const;

    frac operator-(const long long rhs) const;

    frac operator*(const frac &rhs) const;

    frac operator*(const long long rhs) const;

    frac operator/(const frac &rhs) const;

    frac operator/(const long long rhs) const;

    frac &operator+=(const frac &rhs);

    frac &operator+=(const long long rhs);

    frac &operator-=(const frac &rhs);

    frac &operator-=(const long long rhs);

    frac &operator*=(const frac &rhs);

    frac &operator*=(const long long rhs);

    frac &operator/=(const frac &rhs);

    frac &operator/=(const long long rhs);

    frac &operator++();

    frac &operator--();

    friend frac operator-(const frac &unary);

    friend std::ostream &operator<<(std::ostream &out, const frac &rhs);

};

#endif //O3_SIGMA_MODEL_FRAC_HPP
