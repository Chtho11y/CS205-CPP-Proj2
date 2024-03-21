#include<iostream>
#include<fstream>
#include<string>
#include<ctime>

std::string file_in(int n){
    return " <data/" + std::to_string(n) + ".txt";
}

std::string file_cout(int n){
    return " >result/res_c_" + std::to_string(n) + ".txt";
}

std::string file_jout(int n){
    return " >result/res_java_" + std::to_string(n) + ".txt";
}

std::string file_nout(){
    return " >bin.txt";
}

int main(){
    for(int i = (1<<8); i <= (1<<13); i <<= 1){
        std::cout << "N = " << i << std::endl;
        std::string cmd_c = "main.exe" + file_in(i) + file_nout();
        system(cmd_c.c_str());
        // std::string cmd_j = "java -jar main.jar" + file_in(i) + file_jout(i);
        // system(cmd_j.c_str());
    }
    // system("main.exe < data/4.txt");
    // system("main.exe >result/res_c.txt");
    // system("java -jar main.jar >result/res_java.txt");
}