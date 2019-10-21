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

    EXIT;
}