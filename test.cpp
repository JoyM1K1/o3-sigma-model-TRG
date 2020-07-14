//
// Created by Joy on 2020/06/11.
//

#include <iostream>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

int main() {
    std::cout << std::legendre(4, 1) << '\n';
}