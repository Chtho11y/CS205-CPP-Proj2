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

#define START_P 512
#define STEP 512
#define END_P 8192
#define TEST_CNT 3

const int STEP_CNT = ((END_P - START_P) / STEP) + 1;

const char *test_name;
double tot_tim[STEP_CNT];
mat_t std;

void begin_testcase(const char *name){
    test_name = name;
    log_info("Test case %s begin.\n", name);
    for(int i = 0; i < STEP_CNT; ++i){
        tot_tim[i] = 0;
    }
}

#define BENCHMARK(name, proc)\
    begin_testcase(#name);\
    for(int i = START_P, id = 0; i <= END_P; i += STEP, ++id){\
        create_mat(mat_a, i, i);\
        create_mat(mat_b, i, i);\
        for(int tid = 0; tid <= TEST_CNT; ++tid){\
            load_mat(mat_a);\
            load_mat(mat_b);\
            double st = get_clock();\
            res = name##proc(mat_a, mat_b);\
            double ed = get_clock();\
            if(tid)\
                tot_tim[tid] += ed - st;\
            free_mat(res);\
        }\
        tot_tim[tid] /= 3;\
        log_info("## %d = %9.2f ms\n", tot_tim[tid]);\
        destroy_mat(mat_a);\
        destroy_mat(mat_b);\
    }\
    end_testcase()\
    
int n, m, k;

void end_testcase(){
    printf("%15s ",test_name);
    for(int i = 0; i < STEP_CNT; ++i){
        log_save("%9.2f", tot_tim[i]);
    }
    log_save("\n");
}

int main(){
    freopen("bin.txt", "w", stdout);
    mat_t mat_a, mat_b;
    mat_ptr res;
    log_save("%15s", "N");
    for(int i = START_P; i <= END_P; i += STEP){
        log_save("%9d", i);
    }
    log_save("\n");
}
    