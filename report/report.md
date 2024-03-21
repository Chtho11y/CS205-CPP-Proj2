<center><h1>Project2:Matrix Multiplication</h1></center>

**Name**: 周益贤(Zhou Yixian)

**SID**: 12211825

## Part1 项目分析

项目要求分别用 C 语言和 Java 分别实现矩阵乘法，比较效率并进行分析。

朴素的矩阵乘法复杂度为 $O(n^3)$, 虽然现在已有复杂度更优的算法，如 $O(n^{2.89})$ 的 Strassen 算法，但是由于各方面限制，朴素的矩阵乘法反而在大部分情况效率更优。因此本次项目将分别使用 Java 和 C 语言，尝试实现更快的朴素矩阵乘法，并分析限制矩阵乘法效率的因素具体是什么。

测试主要分为两方面：

1. 分别用两种语言实现相同的优化手段，比较它们执行相同算法的运行效率。
2. 使用两种语言所允许的任何优化手段，比较它们能提供的最佳效率。

测试中 Java 不使用第三方矩阵乘法库，因为它们可能就是用 C 或 C++ 实现的。

为了便于测试，此次比较的矩阵均为方阵，且大小均为 2 的整数次方。

#### 1. 测试平台

+ 处理器：AMD R9-3900X

     下列数据来自 CPU-Z.

     + 基础频率：3.8 GHz， 最大频率：4.6 GHz，经观察测试时频率通常在 4.2 GHz 左右。
     + 核心数：12，线程数：24.
     + 缓存：L1(数据): 384KB, L2: 6MB, L3: 64MB.

+ 内存：32GB 3200 MHz (1600MHz x2) 

     + 通道：2x64 bits

+ 操作系统：Windows10 22H2

#### 2. 分析指标

为了更好地分析矩阵乘法的性能，除了统计用时，这里额外采用两个指标：

1. 时间对数

   根据复杂度理论，当矩阵大小 $n$ 足够大时，低阶项可忽略，计算用时近似正比与 $n^3$，即
   $$
   \begin{align}
   T &= k n^3\\
   \ln T &= \ln k + 3\ln n 
   \end{align}
   $$
   可以发现 $\ln T$ 正比于 $\ln n$，直线拟合后比较 $\ln k$ 即可反映出效率差异。

   ~~同时也避免一些样例用时过长画不下~~

   事实上这种办法遇到了一些问题，实际数据分析时会指出。

2. 每秒浮点运算次数 $\text{FLOPS}$

   $\text{FLOPS}$ 是常用的衡量计算速度的指标。

   朴素的矩阵乘法采用公式：$C_{i,j} = \sum\limits_{k=1}^n A_{i,k}\cdot B_{k,j}$，需要 $n^3$ 次乘法和 $n^3$ 次加法，即实际进行了 $2n^3$ 次浮点运算。

   故计算速度 $v = \dfrac {2\cdot 10^{-6}n^3 } {T}$ ，其中 $T$ 单位为毫秒，$v$ 单位为十亿次浮点运算每秒，即 $\text{GFLOPS}$.

#### 3. 机器性能估计

该项目中的实测数据来源于 AIDA64 benchmark.

通过比较理论性能与实际表现，可以估计运算瓶颈。

1. CPU 性能：

   该 CPU 共有 $12$ 核心，支持 avx2 指令集，不支持 avx512 指令集，频率按 $4.2$ GHz 计算。

   矩阵乘法中可用 fmadd 指令，同时对 $8$ 个单精度浮点数完成一次乘法和一个加法，花费 0.5 个时钟周期。

   故该 CPU 的理论性能为：
   $$
   v = 12\times 4.2 \times 8 \times 2 / 0.5 = 1612.8 \text{ GFLOPS}
   $$
   实测 CPU 的单精度浮点性能为 $1578$ GFLOPS.

   由于 CPU 的频率不固定，实际性能可能有波动，但不会有太大偏差。

2. 内存性能：

   内存频率为 $3200$ MHz，通道为2x64bits，内存带宽为 $3.2\times 2\times 64/8 = 51.2 \text{GBps}$，即每秒最多传输 $12.8\text{G}$ 个单精度浮点数。

   实测内存读取速率为 $47.7$ GBps，写入速率为 $45.9$ GBps. 相当于 $11.5-12\text{G}$ 个单精度浮点数。

   L1 - L3 缓存的速率均接近甚至超过 $1$ TBps，不太可能构成瓶颈，不作讨论。

测试均为极其理想的情况，然而实际运算过程中 CPU 和内存会相互制约，结果可能比二者理论性能都要差。

## Part2 测试结果及分析

注：C 语言使用一维数组模拟二维数组，即位于 $(i,j)$ 的元素实际下标为 $i \cdot cols + j$.

#### 1. 普通乘法

普通乘法按照公式，使用三重循环直接计算：

C 语言：

```c
mat_ptr mat_mul_very_naive(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t));
    create_mat(res, a->rows, b->cols);
    clear_mat(res);
    int N = a->rows, M = a->cols, K = b->cols;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < K; j++)
            for(int k = 0; k < M; k++)
                res->data[i * K + j] += a->data[i * M + k] * b->data[k * K + j];
    return res;
}
```

Java：

```java
Matrix multiply_naive(Matrix mat){
        int N = getRow();
        int M = getCol();
        int K = mat.getCol();
        Matrix res = new Matrix(N, K);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                for (int k = 0; k < M; k++) {
                    res.data[i][j] += data[i][k] * mat.data[k][j];
                }
            }
        }
        return res;
    }
```

用时：



