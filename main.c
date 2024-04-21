#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<stdalign.h>
#include<xmmintrin.h>
#include<immintrin.h>
#include<math.h> 

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


FILE *data_in;

__attribute__((constructor))
void init(){
    log_info("C matmul start.\n");
    srand(time(0));

    // data_in = fopen("data/data.in", "r");
}

__attribute__((destructor))
void finish(){
    log_info("C matmul end.\n");
    //fclose(data);
}

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

void load_mat(mat_ptr mat){
    int n = mat->cols * mat->rows;
    for(int i = 0; i < n; i++)
        scanf("%f", mat->data + i);
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

mat_ptr mat_mul_naive(mat_ptr a, mat_ptr b){
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
    destroy_mat(bt);
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
    destroy_mat(bt);
    return res;
}

mat_ptr mat_mul_openmp2(mat_ptr a, mat_ptr b){
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
                _mm256_storeu_ps(res->data + ib + j , 
                    _mm256_fmadd_ps(val, _mm256_loadu_ps(b->data + ik + j), _mm256_loadu_ps(res->data + ib + j))
                );
            }
    }
    return res;
}

#define BS 32

mat_ptr mat_mul_block(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t));
    create_mat(res, a->rows, b->cols);
    clear_mat(res);

    alignas(64) float A[BS][BS];
    alignas(64) float B[BS][BS];
    alignas(64) float C[BS][BS];

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
                
                // for(int i = 0; i < BS; i += 4)
                //     for(int k = 0; k < BS; k += 4){
                //         value_type v01 = A[i][k], v02 = A[i][k + 1], v03 = A[i][k + 2], v04 = A[i][k + 3];
                //         value_type v11 = A[i + 1][k], v12 = A[i + 1][k + 1], v13 = A[i + 1][k + 2], v14 = A[i + 1][k + 3];
                //         value_type v21 = A[i + 2][k], v22 = A[i + 2][k + 1], v23 = A[i + 2][k + 2], v24 = A[i + 2][k + 3];
                //         value_type v31 = A[i + 3][k], v32 = A[i + 3][k + 1], v33 = A[i + 3][k + 2], v34 = A[i + 3][k + 3];
                //         #pragma omp simd
                //         for(int j = 0; j < BS; ++j){
                //             value_type t1 = 0 , t2 = 0, t3 = 0, t4 = 0;
                //             value_type b1 = B[k][j], b2 = B[k + 1][j], b3 = B[k + 2][j], b4 = B[k + 3][j];

                //             t1 += v01 * b1;
                //             t2 += v11 * b1;
                //             t3 += v21 * b1;
                //             t4 += v31 * b1;
                            
                //             t1 += v02 * b2;
                //             t2 += v12 * b2;
                //             t3 += v22 * b2;
                //             t4 += v32 * b2;

                //             t1 += v03 * b3;
                //             t2 += v13 * b3;
                //             t3 += v23 * b3;
                //             t4 += v33 * b3;
                            
                //             t1 += v04 * b4;
                //             t2 += v14 * b4;
                //             t3 += v24 * b4;
                //             t4 += v34 * b4;

                //             C[i][j] += t1;
                //             C[i + 1][j] += t2;
                //             C[i + 2][j] += t3;
                //             C[i + 3][j] += t4;
                //         }
                //     }

                for(int i = 0; i < BS; i += 2)
                    for(int k = 0; k < BS; k += 4){
                        value_type v01 = A[i][k], v02 = A[i][k + 1], v03 = A[i][k + 2], v04 = A[i][k + 3];
                        value_type v11 = A[i + 1][k], v12 = A[i + 1][k + 1], v13 = A[i + 1][k + 2], v14 = A[i + 1][k + 3];
                        #pragma omp simd
                        for(int j = 0; j < BS; ++j){
                            value_type t1 = 0 , t2 = 0;
                            value_type b1 = B[k][j], b2 = B[k + 1][j], b3 = B[k + 2][j], b4 = B[k + 3][j];

                            t1 += v01 * b1;
                            t2 += v11 * b1;
                            
                            t1 += v02 * b2;
                            t2 += v12 * b2;

                            t1 += v03 * b3;
                            t2 += v13 * b3;

                            t1 += v04 * b4;
                            t2 += v14 * b4;

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

const char *test_name;
double begin_cl;
mat_t std;

void begin_testcase(const char *name){
    test_name = name;
    begin_cl = get_clock();
    log_info("Test case %s begin.\n", name);
}

int n, m, k;

void end_testcase(mat_ptr res){
    double duration = (get_clock() - begin_cl);
    double eps = cmp_mat(res, std);
    double gflops = (double)n * n * n * 2e-6 / duration;
    log_info("Test case '%s' end in %.2f ms, max delta = %.6f, GFLOPS = %.2f\n", test_name, duration, eps, gflops);
    log_save("Test case '%s': %.2f ms, GFLOPS = %.2f, delta = %.6f\n", test_name, duration, gflops, eps);
}

int main(){

    freopen("data/4096.txt", "r", stdin);
    mat_t mat_a, mat_b;
    
    begin_testcase("load matrix");
        scanf("%d", &n);
        m = k = n;    
        create_mat(mat_a, n, m);
        create_mat(mat_b, m, k);
        create_mat(std, n, k);
        load_mat(mat_a);
        load_mat(mat_b);
        load_std_mat(std);
    end_testcase(std);

    // begin_testcase("very naive mul");
    //     mat_ptr res1 = mat_mul_very_naive(mat_a, mat_b);
    // end_testcase(res1);

    // begin_testcase("naive mul");
    //     mat_ptr res2 = mat_mul_naive(mat_a, mat_b);
    // end_testcase(res2);

    // begin_testcase("transpose mul");
    //     mat_ptr res3 = mat_mul_trans(mat_a, mat_b);
    // end_testcase(res3);

    // begin_testcase("simd mul");
    //     mat_ptr res4 = mat_mul_simd(mat_a, mat_b);
    // end_testcase(res4);

//     begin_testcase("openmp2 mul");
//         mat_ptr res5 = mat_mul_openmp2(mat_a, mat_b);
//     end_testcase(res5);    

    begin_testcase("block mul");
        mat_ptr res6 = mat_mul_block_fast(mat_a, mat_b);
    end_testcase(res6);
}