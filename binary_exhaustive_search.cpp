#undef _GLIBCXX_DEBUG                // disable run-time bound checking, etc
#pragma GCC optimize("Ofast,inline") // Ofast = O3,fast-math,allow-store-data-races,no-protect-parens

#pragma GCC target("bmi,bmi2,lzcnt,popcnt")                      // bit manipulation
#pragma GCC target("movbe")                                      // byte swap
#pragma GCC target("aes,pclmul,rdrnd")                           // encryption
#pragma GCC target("avx,avx2,f16c,fma,sse3,ssse3,sse4.1,sse4.2") // SIMD

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <bitset>
#include <bits/stdc++.h>
#include <ctime>

/*
 *   g++ -O3 -o binary_exhaustive_search binary_exhaustive_search.cpp
 */
//using namespace std;

#include "binary_exhaustive_search.h"


void test00()
{

    int potencia = 0;
    int bits     = 64;
    int vecinos  = 1000;
    int base_vecinos = 1000;

    //
    clock_t begin = 0;
    clock_t end = 0;
    clock_t total_time = 0;

    std::string dir_base = "data/";
    std::string files_dir = "";
    std::string class_file_train = "";
    std::string vector_file_train = "";

    std::string class_file_val = "";
    std::string vector_file_val = "";
    
    std::vector<std::string> dbs = {"resnet101_hdf_imagenet"};

    // 
    int d = 0
    potencia = 10;
    //potencia = pts_pows[d];
    printf("====\n====\n");
    printf("DB: %s\n\n", dbs[0].c_str());
    printf("bits: %e\n", pow(2, potencia));

    // crear indice
    ES * es = new ES(potencia, bits, vecinos, base_vecinos);

    es->iniciar_indice();
    //
    //files_dir = base_db + dbs[d];
    files_dir = dbs[0];

    //
    class_file_train  = dir_base+"corr_mtrx_0"+std::to_string(d)+"_"+files_dir+"_train/txt_classes.txt";  // +std::to_string(id)
    vector_file_train = dir_base+"corr_mtrx_0"+std::to_string(d)+"_"+files_dir+"_train/txt_vectors.txt";  // +std::to_string(id)
    es->crear_indice(class_file_train, vector_file_train);

    //
    es->baseFile = "index_files";
    es->db_name  = "0"+std::to_string(d)+"_"+files_dir;  // std::to_string(id)+
    class_file_val  = dir_base+"corr_mtrx_0"+std::to_string(d)+"_"+files_dir+"_val/txt_classes.txt";  // +std::to_string(id)
    vector_file_val = dir_base+"corr_mtrx_0"+std::to_string(d)+"_"+files_dir+"_val/txt_vectors.txt";  // +std::to_string(id)
    es->ejecutar_indice(class_file_val, vector_file_val);

    //
    es->limpiar_memoria();
}

// 
int main(int argc, char** argv)
{
    test00();

    return 0;
}
