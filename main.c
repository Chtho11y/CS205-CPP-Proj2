#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<stdalign.h>
#include<xmmintrin.h>
#include<immintrin.h>
#include<math.h> 

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

mat_ptr mat_mul_simd(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t)), bt = trans_mat(b);
    create_mat(res, a->rows, b->cols);
    clear_mat(res);
    int N = a->rows, M = a->cols, K = b->rows;

    #pragma omp parallel for
    for(int i = 0; i < N; i++)
        for(int j = 0; j < K; j++){
            __m256 sum = _mm256_setzero_ps();
            value_type *a_ptr = &a->data[i * M];
            value_type *bt_ptr = &bt->data[j * M];

            for(int k = 0; k < M; k += 8)
                sum = _mm256_fmadd_ps(_mm256_load_ps(a_ptr + k), _mm256_loadu_ps(bt_ptr + k), sum);
            int tmp = 0;
            for(int i = 0; i < 8; ++i)
                tmp += sum[i];
            res->data[i * K + j] = tmp;
        }
    destroy_mat(bt);
    return res;
}

mat_ptr mat_mul_block(mat_ptr a, mat_ptr b){
    mat_ptr res = malloc(sizeof(mat_t)), bt = trans_mat(b);
    create_mat(res, a->rows, b->cols);
    clear_mat(res);

    int N = a->rows, M = a->cols, K = b->rows;
    
    destroy_mat(bt);
    return res;
}

const char *test_name;
int begin_cl;
mat_ptr std;

void begin_testcase(const char *name){
    test_name = name;
    begin_cl = clock();
    log_info("Test case %s begin.\n", name);
}

void end_testcase(){
    int duration = clock() - begin_cl;
    log_info("Test case '%s' end in %d ms\n", test_name, duration);
    log_save("Test case '%s': %d ms\n", test_name, duration);
}

int main(){
    mat_t mat_a, mat_b;
    
    begin_testcase("load matrix");
        int n, m, k;
        scanf("%d", &n);
        m = k = n;    
        create_mat(mat_a, n, m);
        create_mat(mat_b, m, k);
        create_mat(std, n, k);
        load_mat(mat_a);
        load_mat(mat_b);
        load_std_mat(std);
    end_testcase();

    // begin_testcase("very naive mul");
    //     mat_ptr res1 = mat_mul_very_naive(mat_a, mat_b);
    // end_testcase();

    // begin_testcase("transpose mul");
    //     mat_ptr res2 = mat_mul_trans(mat_a, mat_b);
    // end_testcase();

    begin_testcase("openmp mul");
        mat_ptr res3 = mat_mul_openmp(mat_a, mat_b);
    end_testcase();

    begin_testcase("simd mul");
        mat_ptr res4 = mat_mul_simd(mat_a, mat_b);
    end_testcase();    
    log_info("max eps = %.6f\n", cmp_mat(res3, std));
}