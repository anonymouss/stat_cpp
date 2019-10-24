#include <cstdio>

#include "Math.h"
#include "Types.h"

#define TestName "Math"
#define ENTER printf("\n=== Run test " TestName " ===\n\n");
#define EXIT printf("\n=== Exit test " TestName " ===\n\n");

int main() {
    ENTER;

    auto dispMat = [](stat::Mat<int> mat) {
        for (auto i = 0; i < mat.size(); ++i) {
            for (auto j = 0; j < mat[i].size(); ++j) { printf("%d, ", mat[i][j]); }
            printf("\n");
        }
        printf("\n");
    };

    stat::Vec<int> x1{3, 3}, x2{4, 3}, x3{1, 1};
    stat::Mat<int> X{x1, x2, x3};
    auto g = stat::gram(X);
    dispMat(g);

    stat::Vec<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto mu = stat::mean(v);
    auto sigma = stat::stdev(v);
    auto gaussian = stat::gaussian_prob(5, mu, sigma);
    dispMat({v});
    printf("mu = %f, sigma = %f, gaussian probability of 5 = %f\n", mu, sigma, gaussian);

    EXIT;
}