#include<iostream>
#include<fstream>
#include<filesystem>
#include<string>
#include<random>
#include<ctime>

int main(){
    std::uniform_real_distribution<float> gen_float(-1,1);
    std::default_random_engine gen(time(0));

    // for(size_t n = (1<<8); n <= (1<<13); n <<= 1){
    int n = 64;
        // std::string name = "data/" + std::to_string(n) + ".txt";
        std::string name = "data.txt";
        std::ofstream out(name);

        out << n << std::endl;
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < n; ++j){
                // out<< gen_float(gen) << " ";
                out<<gen()%100<<" ";
            }
            out << "\n";
        }
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < n; ++j){
                // out<< gen_float(gen) << " ";
                out<<gen()%100<<" ";
            }
            out << "\n";
        }

        out.flush();
        out.close();
    // }
}