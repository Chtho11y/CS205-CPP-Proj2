package main;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.*;

class Matrix{
    float[][] data;

    Matrix(int n, int m){
        data = new float[n][m];
    }

    void clear(){
        for(float[] arr: data){
            Arrays.fill(arr, 0);
        }
    }

    void get(Scanner sc){
        int N = getRow();
        int M = getCol();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++){
                data[i][j] = sc.nextFloat();
            }
        }
    }

    void rand(){
        int N = getRow();
        int M = getCol();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                data[i][j] = (float) (Math.random() * 2 - 1);
            }
        }
    }

    Matrix transpose(){
        int N = getRow();
        int M = getCol();
        Matrix res = new Matrix(M, N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                res.data[j][i] = data[i][j];
            }
        }
        return res;
    }

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

    Matrix multiply_reorder(Matrix mat){
        int N = getRow();
        int M = getCol();
        int K = mat.getCol();
        Matrix res = new Matrix(N, K);
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < M; k++){
                for (int j = 0; j < K; j++){
                    res.data[i][j] += data[i][k] * mat.data[k][j];
                }
            }
        }
        return res;
    }

    Matrix multiply_trans(Matrix mat){
        Matrix mt = mat.transpose();
        int N = getRow();
        int M = getCol();
        int K = mat.getCol();
        Matrix res = new Matrix(N, K);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                for (int k = 0; k < M; k++) {
                    res.data[i][j] += data[i][k] * mt.data[j][k];
                }
            }
        }
        return res;
    }

    Matrix multiply_thread(Matrix mat) throws ExecutionException, InterruptedException {
        Matrix mt = mat.transpose();
        int N = getRow();
        int M = getCol();
        int K = mat.getCol();
        Matrix res = new Matrix(N, K);
        ExecutorService executor = Executors.newFixedThreadPool(24);
        ArrayList<Future<?>> futures = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            var C = res.data[i];
            var A = data[i];
            var ftr = executor.submit(()->{
                for (int j = 0; j < K; j++) {
                    var B = mt.data[j];
                    for (int k = 0; k < M; k++) {
                        C[j] += A[k] * B[k];
                    }
                }
            });
            futures.add(ftr);
        }

        executor.shutdown();
        for (Future<?> future: futures) {
            future.get();
        }
        return res;
    }

    Matrix multiply_thread_reorder(Matrix mat) throws ExecutionException, InterruptedException {
        int N = getRow();
        int M = getCol();
        int K = mat.getCol();
        Matrix res = new Matrix(N, K);
        ExecutorService executor = Executors.newFixedThreadPool(24);
        ArrayList<Future<?>> futures = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            var C = res.data[i];
            var A = data[i];
            var ftr = executor.submit(()->{
                for (int k = 0; k < M; k++){
                    var B = mat.data[k];
                    for (int j = 0; j < K; j++){
                        C[j] += A[k] * B[j];
                    }
                }
            });
            futures.add(ftr);
        }

        executor.shutdown();
        for (Future<?> future: futures) {
            future.get();
        }
        return res;
    }

    int getRow(){
        return data.length;
    }

    int getCol(){
        return data[0].length;
    }

    double cmp(Matrix mat){
        double res = 0;
        for (int i = 0; i < mat.getCol(); i++) {
            for (int j = 0; j < mat.getRow(); j++) {
                res = Math.max(res, Math.abs(mat.data[i][j] - data[i][j]));
            }
        }
        return res;
    }
}

public class Main {
    static String name;
    static long st_cl;

    static Matrix std;

    static int n;

    static void testcase_begin(String name){
        Main.name = name;
        st_cl = System.currentTimeMillis();
        System.err.printf("testcase %s begin\n", name);
    }

    static void testcase_end(Matrix res){
        long end_cl = System.currentTimeMillis();
        long dur = end_cl - st_cl;
        double eps = std.cmp(res);
        double gflops = 2e-6 * n * n * n / dur;
        System.err.printf("testcase %s end in %d ms, GFLOPS = %.2f, delta = %.6f\n", name, dur, gflops, eps);
        System.out.printf("testcase %s: %d ms, GFLOPS = %.2f, delta:%.6f\n", name, dur, gflops, eps);
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException, IOException {
        System.err.println("java GEMM begin.");
        testcase_begin("mat init");
        Scanner sc = new Scanner(System.in);
        n = sc.nextInt();
        Matrix mat1, mat2;
        mat1 = new Matrix(n, n);
        mat2 = new Matrix(n, n);
        std = new Matrix(n, n);
        mat1.get(sc);
        mat2.get(sc);
        try(FileInputStream file = new FileInputStream("data/"+ n + ".ans")){
            Scanner fl = new Scanner(file);
            std.get(fl);
        }
        testcase_end(std);

//        testcase_begin("very naive GEMM");
//        var res1 = mat1.multiply_naive(mat2);
//        testcase_end(res1);

        testcase_begin("reorder");
        var res2 = mat1.multiply_reorder(mat2);
        testcase_end(res2);
//
//        testcase_begin("transpose");
//        var res3 = mat1.multiply_trans(mat2);
//        testcase_end(res3);

        testcase_begin("multi thread");
        var res4 = mat1.multiply_thread(mat2);
        testcase_end(res4);

        testcase_begin("multi thread2");
        var res5 = mat1.multiply_thread_reorder(mat2);
        testcase_end(res5);

        System.err.println("java GEMM end.");
    }
}