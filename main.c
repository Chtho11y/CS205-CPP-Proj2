#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<stdalign.h>
#include<xmmintrin.h>
#include<immintrin.h>
#include<math.h>
#include<unistd.h>
// #include "include/cblas.h"

double get_clock(){
    struct timespec tim;
    clock_gettime(CLOCK_MONOTONIC, &tim);
    return tim.tv_sec * 1e3 + tim.tv_nsec / 1e6;
}

typedef float value_type;

typedef struct _matrix{
    value_type* data;
    size_t rows;
    size_t cols;
}mat_t[1], *mat_ptr;

#define log_info(str, ...)\
    fprintf(stderr, str,##__VA_ARGS__)
#define log_save(str, ...)\
    printf(str,##__VA_ARGS__)

void create_mat(mat_ptr mat, size_t rows, size_t cols){
    mat->data = malloc(rows * cols * sizeof(value_type));
    mat->rows = rows;
    mat->cols = cols;
}


void clear_mat(mat_ptr mat){
    memset(mat->data, 0, mat->rows * mat->cols * sizeof(value_type));
}

void destroy_mat(mat_ptr mat){
    free(mat->data);
}

void free_mat(mat_ptr mat){
    destroy_mat(mat);
    free(mat);
}


void load_mat(mat_ptr mat){
    int n = mat->cols * mat->rows;
    for(int i = 0; i < n; i++)
        mat->data[i] = (float)rand() / RAND_MAX;
}

void load_std_mat(mat_ptr mat){
    char buf[20];
    sprintf(buf, "data/%d.ans", (int)mat->cols);
    FILE *in = fopen(buf, "r");
    int n = mat->cols * mat->rows;
    for(int i = 0; i < n; i++)
        fscanf(in, "%f", mat->data + i);
}

void save_mat(mat_ptr mat, FILE *file){
    for(int i = 0; i < mat->rows; i++){
        for(int j = 0; j < mat->cols; j++)
            fprintf(file, "%f ", mat->data[i * mat->cols + j]);
        fprintf(file, "\n");
    }
    fflush(file);
}

value_type cmp_mat(mat_ptr mat1, mat_ptr mat2){
    value_type res = 0.0f;
    int n = mat1->rows * mat1->cols;
    for(int i = 0; i < n; i++)
        res = fmax(res, fabs(mat1->data[i] - mat2->data[i]));
    return res;
}

mat_ptr trans_mat(mat_ptr mat){
    mat_ptr res = malloc(sizeof(mat_t));
    int N = mat->rows;
    int M = mat->cols;
    create_mat(res, M, N);
    #pragma omp parallel for
    for(int i = 0; i < N; i++)
        for(int j = 0; j < M; j++)
            res->data[j * N + i] = mat->data[i * M + j];
    return res;
}

mat_ptr mat_mul_naive(mat_ptr a, mat_ptr b){
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

mat_ptr mat_mul_reorder(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t));
    create_mat(res, a->rows, b->cols);
    clear_mat(res);
    int N = a->rows, M = a->cols, K = b->cols;
    for(int i = 0; i < N; i++)
        for(int k = 0; k < M; k++)
            for(int j = 0; j < K; j++)
                res->data[i * K + j] += a->data[i * M + k] * b->data[k * K + j];
    return res;
}

mat_ptr mat_mul_trans(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t)), bt = trans_mat(b);
    create_mat(res, a->rows, b->cols);
    clear_mat(res);
    int N = a->rows, M = a->cols, K = b->rows;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < K; j++){
            value_type *res_ptr = &res->data[i * K + j];
            value_type *a_ptr = &a->data[i * M];
            value_type *bt_ptr = &bt->data[j * M];
            for(int k = 0; k < M; k++)
                *res_ptr += a_ptr[k] * bt_ptr[k];
        }
    free_mat(bt);
    return res;
}

mat_ptr mat_mul_openmp(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t)), bt = trans_mat(b);
    create_mat(res, a->rows, b->cols);
    clear_mat(res);
    int N = a->rows, M = a->cols, K = b->rows;

    #pragma omp parallel for
    for(int i = 0; i < N; i++)
        for(int j = 0; j < K; j++){
            value_type sum = 0;
            value_type *a_ptr = &a->data[i * M];
            value_type *bt_ptr = &bt->data[j * M];

            #pragma omp simd reduction(+: sum)
            for(int k = 0; k < M; k++)
                sum += a_ptr[k] * bt_ptr[k];
            res->data[i * K + j] = sum;
        }
    free_mat(bt);
    return res;
}

mat_ptr mat_mul_parallel(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t)), bt = trans_mat(b);
    create_mat(res, a->rows, b->cols);
    clear_mat(res);
    int N = a->rows, M = a->cols, K = b->rows;

    #pragma omp parallel for
    for(int i = 0; i < N; i++)
        for(int j = 0; j < K; j++){
            value_type sum = 0;
            value_type *a_ptr = &a->data[i * M];
            value_type *bt_ptr = &bt->data[j * M];

            for(int k = 0; k < M; k++)
                sum += a_ptr[k] * bt_ptr[k];
            res->data[i * K + j] = sum;
        }
    free_mat(bt);
    return res;
}

mat_ptr mat_mul_openmp_reorder(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t));
    create_mat(res, a->rows, b->cols);
    clear_mat(res);
    int N = a->rows, M = a->cols, K = b->cols;
    #pragma omp parallel for
    for(int i = 0; i < N; i++)
        for(int k = 0; k < M; k++){
            value_type v = a->data[i * M + k];
            int ib = i * K;
            int ik = k * K;
            #pragma omp simd
            for(int j = 0; j < K; j++)
                res->data[ib + j] += v * b->data[ik + j];
        }
    return res;
}

mat_ptr mat_mul_simd(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t)), bt = trans_mat(b);
    create_mat(res, a->rows, b->cols);
    clear_mat(res);
    int N = a->rows, M = a->cols, K = b->rows;

    for(int i = 0; i < N; i++)
        for(int j = 0; j < K; j++){
            __m256 sum = _mm256_setzero_ps();
            value_type *a_ptr = &a->data[i * M];
            value_type *bt_ptr = &bt->data[j * M];
            for(int k = 0; k < M; k += 8)
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr + k), _mm256_loadu_ps(bt_ptr + k), sum);
            value_type tmp = 0;
            for(int i = 0; i < 8; ++i)
                tmp += sum[i];
            res->data[i * K + j] = tmp;
        }
    free_mat(bt);
    return res;
}

mat_ptr mat_mul_simd_reorder(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t));
    create_mat(res, a->rows, b->cols);
    clear_mat(res);
    int N = a->rows, M = a->cols, K = b->rows;

    for(int i = 0; i < N; i++)
        for(int k = 0; k < M; k++){
            int ib = i * K;
            int ik = k * K;
            
            __m256 val = _mm256_set1_ps(a->data[i * M + k]);

            for(int j = 0; j < K; j += 8){
                value_type* p = res->data + ib + j;
                _mm256_storeu_ps(p , 
                    _mm256_fmadd_ps(val, _mm256_loadu_ps(b->data + ik + j), _mm256_loadu_ps(p))
                );
            }
    }
    return res;
}

#define BS 64

mat_ptr mat_mul_block(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t));
    create_mat(res, a->rows, b->cols);
    clear_mat(res);

    alignas(64) float A[BS][BS];
    alignas(64) float B[BS][BS];
    alignas(64) float C[BS][BS];

    // float A[BS][BS];
    // float B[BS][BS];
    // float C[BS][BS];

    int N = a->rows, M = a->cols, K = b->cols;

    #pragma omp parallel for private(A, B, C)
    for(int bi = 0; bi < N; bi +=BS){
        for(int bk = 0; bk < M; bk += BS){
            
            for(int i = 0; i < BS; ++i){
                value_type* a_ptr  = a->data + (i + bi) * M + bk;
                #pragma omp simd
                for(int k = 0; k < BS; ++k)
                    A[i][k] = a_ptr[k];
            }

            for(int bj = 0; bj < K; bj += BS){
                memset(C, 0, sizeof(C));
                for(int k = 0; k < BS; ++k){
                    value_type* b_ptr = b->data + (k + bk) * K + bj;
                    #pragma omp simd
                    for(int j = 0; j < BS; ++j)
                        B[k][j] = b_ptr[j];
                }
                for(int i = 0; i < BS; ++i)
                    for(int k = 0; k < BS; ++k){
                        value_type v = A[i][k];
                        #pragma omp simd
                        for(int j = 0; j < BS; ++j)
                            C[i][j] += v * B[k][j];
                    }
                
                for(int i = 0; i < BS; ++i){
                    value_type* res_ptr = res->data + (i + bi) * K + bj;
                    #pragma omp simd
                    for(int j = 0; j < BS; ++j)
                        res_ptr[j] += C[i][j];
                }
            }
        }
    }
    
    return res;
}

mat_ptr mat_mul_block_fast(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t));
    create_mat(res, a->rows, b->cols);
    clear_mat(res);

    alignas(64) float A[BS][BS];
    alignas(64) float B[BS][BS];
    alignas(64) float C[BS][BS];

    int N = a->rows, M = a->cols, K = b->cols;

    #pragma omp parallel for private(A, B, C)
    for(int bi = 0; bi < N; bi += BS){
        for(int bk = 0; bk < M; bk += BS){
            
            for(int i = 0; i < BS; ++i){
                value_type* a_ptr  = a->data + (i + bi) * M + bk;
                #pragma omp simd
                for(int k = 0; k < BS; ++k)
                    A[i][k] = a_ptr[k];
            }

            for(int bj = 0; bj < K; bj += BS){
                memset(C, 0, sizeof(C));
                for(int k = 0; k < BS; ++k){
                    value_type* b_ptr = b->data + (k + bk) * K + bj;
                    #pragma omp simd
                    for(int j = 0; j < BS; ++j)
                        B[k][j] = b_ptr[j];
                }

                // for(int i = 0; i < BS; ++i)
                //     for(int k = 0; k < BS; k += 4){
                //         value_type v1 = A[i][k];
                //         value_type v2 = A[i][k + 1];
                //         value_type v3 = A[i][k + 2];
                //         value_type v4 = A[i][k + 3];
                //         value_type *cptr = C[i];
                //         #pragma omp simd
                //         for(int j = 0; j < BS; ++j){
                //             value_type t = 0;
                //             t += v1 * B[k][j];
                //             t += v2 * B[k + 1][j];
                //             t += v3 * B[k + 2][j];
                //             t += v4 * B[k + 3][j];
                //             cptr[j] += t;
                //         }
                //     }

                for(int i = 0; i < BS; i += 2)
                    for(int k = 0; k < BS; k += 4){
                        value_type v01 = A[i][k], v02 = A[i][k + 1], v03 = A[i][k + 2], v04 = A[i][k + 3];
                        value_type v11 = A[i + 1][k], v12 = A[i + 1][k + 1], v13 = A[i + 1][k + 2], v14 = A[i + 1][k + 3];
                        #pragma omp simd
                        for(int j = 0; j < BS; ++j){
                            value_type t1 = 0 , t2 = 0;
                            // value_type b1 = B[k][j], b2 = B[k + 1][j], b3 = B[k + 2][j], b4 = B[k + 3][j];

                            t1 += v01 * B[k][j];
                            t2 += v11 * B[k][j];
                            
                            t1 += v02 * B[k + 1][j];
                            t2 += v12 * B[k + 1][j];

                            t1 += v03 * B[k + 2][j];
                            t2 += v13 * B[k + 2][j];

                            t1 += v04 * B[k + 3][j];
                            t2 += v14 * B[k + 3][j];

                            C[i][j] += t1;
                            C[i + 1][j] += t2;
                        }
                    }
                
                for(int i = 0; i < BS; ++i){
                    value_type* res_ptr = res->data + (i + bi) * K + bj;
                    #pragma omp simd
                    for(int j = 0; j < BS; ++j)
                        res_ptr[j] += C[i][j];
                }
            }
        }
    }
    
    return res;
}

// mat_ptr mat_mul_openblas(mat_ptr a, mat_ptr b){
//     mat_ptr res = malloc(sizeof(mat_t));
//     create_mat(res, a->rows, b->cols);
//     clear_mat(res);

//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
//         a->rows, b->cols, a->cols, 
//         1.0, a->data, a->cols, 
//         b->data, b->cols, 0.0, 
//         res->data, res->cols);

//     return res;
// }

// #define SLOW_TEST

#ifdef SLOW_TEST

#define START_P 128
#define STEP 128
#define END_P 2048
#define TEST_CNT 3

#else

#define START_P 256
#define STEP 256
#define END_P 4096
#define TEST_CNT 3

#endif

#define STEP_CNT (((END_P - START_P) / STEP) + 1)

__attribute__((constructor))
void init(){
    log_info("C matmul start. RANGE = [%d, %d], STEP = %d\n", START_P, END_P, STEP);
    srand(time(0));
}

__attribute__((destructor))
void finish(){
    log_info("C matmul end.\n");
}

const char *test_name;
double tot_tim[STEP_CNT];
mat_t std;

void begin_testcase(const char *name){
    test_name = name;
    log_info("Test case '%s' begin.\n", name);
    for(int i = 0; i < STEP_CNT; ++i){
        tot_tim[i] = 0;
    }
}

void pretty_print(double v){
    char buf[20];
    sprintf(buf, "(%.2f)", v);
    log_info("%9s", buf);
}

#define BENCHMARK(name)\
    begin_testcase(#name);\
    for(int i = START_P, id = 0; i <= END_P; i += STEP, ++id){\
        create_mat(mat_a, i, i);\
        create_mat(mat_b, i, i);\
        log_info("##%02d %4d = ", id, i);\
        for(int tid = 0; tid <= TEST_CNT; ++tid){\
            load_mat(mat_a);\
            load_mat(mat_b);\
            double st = get_clock();\
            res = mat_mul_##name(mat_a, mat_b);\
            double ed = get_clock();\
            if(tid){\
                tot_tim[id] += ed - st;\
                log_info("%9.2f", ed - st);\
            }else{\
                pretty_print(ed - st);\
            }\
            free_mat(res);\
        }\
        tot_tim[id] /= 3;\
        log_info(" => %9.2f ms(%.2f GFLOPS)\n", tot_tim[id], 2e-6 * i * i * i / tot_tim[id]);\
        destroy_mat(mat_a);\
        destroy_mat(mat_b);\
    }\
    end_testcase()\
    
int n, m, k;

void end_testcase(){
    printf("%15s ",test_name);
    for(int i = 0; i < STEP_CNT; ++i){
        log_save("%.2f%c", tot_tim[i],", "[i == STEP_CNT - 1]);
    }
    log_save("\n");
    sleep(3);
}

int main(){
    mat_t mat_a, mat_b;
    mat_ptr res;
    log_save("%15s", "N");
    for(int i = START_P; i <= END_P; i += STEP){
        log_save("%9d", i);
    }
    log_save("\n");

#ifdef SLOW_TEST

    // BENCHMARK(naive);
    // BENCHMARK(trans);
    // BENCHMARK(reorder);
    // BENCHMARK(simd);
    // BENCHMARK(simd_reorder);
    // BENCHMARK(block);
    // BENCHMARK(openmp_reorder);

#else

    BENCHMARK(openmp);
    BENCHMARK(parallel);
    BENCHMARK(openmp_reorder);
    BENCHMARK(block);
    BENCHMARK(block_fast);
#endif

}