#include<iostream>
#include<fstream>
#include<string>
#include<ctime>

int main(){
    for(int i = (1<<8); i <= (1<<13); i <<= 1){
        std::cout << "N = " << i << std::endl;
        std::string cmd_c = "main.exe <data/" + std::to_string(i) + ".txt";
        system(cmd_c.c_str());
    }
    // system("main.exe < data/4.txt");
    // system("main.exe >result/res_c.txt");
    // system("java -jar main.jar >result/res_java.txt");
}