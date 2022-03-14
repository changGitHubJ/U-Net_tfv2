// import math
// import matplotlib.pyplot as plt
// import numpy as np
// import os
// import tensorflow as tf

// from PIL import Image

// import common as c

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <cmath>
#define _USE_MATH_DEFINES
using namespace std;

#define TRAIN_DATA_SIZE 120
#define TEST_DATA_SIZE 30
#define IMG_SIZE 256
#define OUTPUT_SIZE 256*256
#define CATEGORY 6

vector<string> FILENAMES = {"/home/username/DAGM/Class1_def/",
                            "/home/username/DAGM/Class2_def/",
                            "/home/username/DAGM/Class3_def/",
                            "/home/username/DAGM/Class4_def/",
                            "/home/username/DAGM/Class5_def/",
                            "/home/username/DAGM/Class6_def/"};

vector<float> weight = {0.9, 1.0, 0.9, 0.9, 0.9, 0.8};

vector<string> split(string str, char del) {
    int first = 0;
    int last = str.find_first_of(del);
    vector<string> result;
    if(last < 0) return result;
 
    while (first < str.size())
    {
        string subStr(str, first, last - first);
 
        result.push_back(subStr);
 
        first = last + 1;
        last = str.find_first_of(del, first);
 
        if (last == string::npos) {
            last = str.size();
        }
    }
 
    return result;
}

vector<vector<int>> read2darray_i(string filename, int row, int col)
{
    vector<vector<int>> array(row);

    ifstream ifs(filename.c_str());
    if(!ifs.is_open())
    {
        cout << "cannot open " << filename << endl;
        return array;
    }
    
    for(int i = 0; i < row; i++)
    {
        string line;
        getline(ifs, line);
        vector<string> vals = split(line, ',');
        for(int j = 0; j < col; j++)
        {
            array[i].push_back(stoi(vals[j]));
        }
    }

    ifs.close();

    return array;
}

int main(int argc, char* argv[])
{
    int cntTrain = 0;
    int cntTest = 0;
    vector<string> lineTrainAll;
    vector<string> lineTestAll;
    for(int n = 0; n < CATEGORY; n++)
    {
        vector<string> lineTrain;
        vector<string> lineTest;
        for(int k = 0; k < TRAIN_DATA_SIZE + TEST_DATA_SIZE; k++)
        {
            string filename = FILENAMES[n] + to_string(k + 1) + ".txt";
            printf("%s\n", filename.c_str());
            vector<vector<int>> img = read2darray_i(filename.c_str(), 256, 256);
            
            if(k < TRAIN_DATA_SIZE)
            {
                string line = to_string(cntTrain);
                string lineAll = to_string(cntTrain);
                for(int i = 0; i < IMG_SIZE; i++)
                {
                    for(int j = 0; j < IMG_SIZE; j++)
                    {
                        char buf[64];
                        sprintf(buf, ",%d", img[i][j]);
                        line += buf;
                        lineAll += buf;
                    }
                }
                line += "\n";
                lineAll += "\n";
                lineTrain.push_back(line);
                lineTrainAll.push_back(lineAll);
                cntTrain++;
            }
            else
            {
                string line = to_string(cntTest);
                string lineAll = to_string(cntTest);
                for(int i = 0; i < IMG_SIZE; i++)
                {
                    for(int j = 0; j < IMG_SIZE; j++)
                    {
                        char buf[64];
                        sprintf(buf, ",%d", img[i][j]);
                        line += buf;
                        lineAll += buf;
                    }
                }
                line += "\n";
                lineAll += "\n";
                lineTest.push_back(line);
                lineTestAll.push_back(lineAll);
                cntTest++;
            }
        }
        struct stat statBuf;
        mode_t mode;
        mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
        if(stat("./data", &statBuf) != 0)   
        {
            mkdir("./data", mode);
        }
        string ofilename = "./data/trainImage256_" + to_string(n) + ".txt";
        ofstream ofs0(ofilename);
        for(int i = 0; i < lineTrain.size(); i++)
        {
            ofs0 << lineTrain[i] << flush;
        }
        ofs0.close();
        ofilename = "./data/testImage256_" + to_string(n) + ".txt";
        ofstream ofs1(ofilename);
        for(int i = 0; i < lineTest.size(); i++)
        {
            ofs1 << lineTest[i] << flush;
        }
        ofs1.close();    
    }
    ofstream ofs2("./data/trainImage256_100.txt");
    for(int i = 0; i < lineTrainAll.size(); i++)
    {
        ofs2 << lineTrainAll[i] << flush;
    }
    ofs2.close();
    ofstream ofs3("./data/testImage256_100.txt");
    for(int i = 0; i < lineTestAll.size(); i++)
    {
        ofs3 << lineTestAll[i] << flush;
    }
    ofs3.close();

    // label
    vector<vector<string>> Labels;
    for(int n = 0; n < CATEGORY; n++)
    {
        string filename = FILENAMES[n] + "labels.txt";
        ifstream label(filename);
        cout << "reading " << filename << endl;
        vector<string> lines;
        for(int i = 0; i < 150; i++)
        {
            string line;
            getline(label, line);
            lines.push_back(line);
        }
        label.close();
        Labels.push_back(lines);
    }
    vector<vector<float>> lineTrainAllLab;
    vector<vector<float>> lineTestAllLab;
    for(int n = 0; n < CATEGORY; n++)
    {
        cout << "labeling of category" << n << endl;
        vector<vector<float>> lineTrain;
        vector<vector<float>> lineTest;
        for(int k = 0; k < TRAIN_DATA_SIZE + TEST_DATA_SIZE; k++)
        {
            double* x = new double[512];
            double* y = new double[512];
            for(int i = 0; i < IMG_SIZE; i++)
            {
                x[i] = 1 + 2*i;
                y[i] = 1 + 2*i;
            }
            vector<string> val = split(Labels[n][k], '\t');
            int num = stoi(val[0]) - 1;
            float mjr = stof(val[1]);
            float mnr = stof(val[2]);
            float rot = stof(val[3]);
            float cnx = stof(val[4]);
            float cny = stof(val[5]);

            // inverse rotate pixels
            vector<float> label(OUTPUT_SIZE + 1, 0.0);
            vector<float> labelAll(OUTPUT_SIZE*CATEGORY + 1, 0);
            label[0] = num; // index
            labelAll[0] = num; // index
            for(int i = 0; i < IMG_SIZE; i++)
            {
                for(int j = 0; j < IMG_SIZE; j++)
                {
                    float dist = sqrt((x[i] - cnx)*(x[i] - cnx) + (y[j] - cny)*(y[j] - cny));
                    float xTmp = (x[i] - cnx) * cos(-rot) - (y[j] - cny) * sin(-rot);
                    float yTmp = (x[i] - cnx) * sin(-rot) + (y[j] - cny) * cos(-rot);
                    float ang = atan2(yTmp, xTmp);
                    double distToEllipse = sqrt((mjr * cos(ang))*(mjr * cos(ang)) + (mnr * sin(ang))*(mnr * sin(ang)));
                    if(dist < distToEllipse)
                    {
                        label[j*IMG_SIZE + i + 1] = 1.0; // defection
                        labelAll[(j*IMG_SIZE + i)*CATEGORY + n + 1] = weight[n]; // defection
                    }
                    else
                    {
                        label[j*IMG_SIZE + i + 1] = 0.0;
                        labelAll[(j*IMG_SIZE + i)*CATEGORY + n + 1] = 0.0;
                    }
                }
            }
            if(k < TRAIN_DATA_SIZE)
            {
                lineTrain.push_back(label);
                lineTrainAllLab.push_back(labelAll);
            }
            else
            {
                lineTest.push_back(label);
                lineTestAllLab.push_back(labelAll);
            }
            delete[] x;
            delete[] y;
        }
        string ofilename = "./data/trainLabel256_" + to_string(n) + ".txt";
        ofstream ofs4(ofilename);
        for(int i = 0; i < lineTrain.size(); i++)
        {
            ofs4 << lineTrain[i][0];
            for(int j = 1; j < lineTrain[i].size(); j++)
            {
                ofs4 << "," << lineTrain[i][j];
            }
            ofs4 << endl;
        }
        ofs4.close();
        ofilename = "./data/testLabel256_" + to_string(n) + ".txt";
        ofstream ofs5(ofilename);
        for(int i = 0; i < lineTest.size(); i++)
        {
            ofs5 << lineTest[i][0];
            for(int j = 1; j < lineTest[i].size(); j++)
            {
                ofs5 << "," << lineTest[i][j];
            }
            ofs5 << endl;
        }
        ofs5.close();
    }
    
    // normalize

    ofstream ofs6("./data/trainLabel256_100.txt");
    for(int i = 0; i < lineTrainAllLab.size(); i++)
    {
        ofs6 << lineTrainAllLab[i][0];
        for(int j = 1; j < lineTrainAllLab[i].size(); j++)
        {
            ofs6 << "," << lineTrainAllLab[i][j];
        }
        ofs6 << endl;
    }
    ofs6.close();
    ofstream ofs7("./data/testLabel256_100.txt");
    for(int i = 0; i < lineTestAllLab.size(); i++)
    {
        ofs7 << lineTestAllLab[i][0];
        for(int j = 1; j < lineTestAllLab[i].size(); j++)
        {
            ofs7 << "," << lineTestAllLab[i][j];
        }
        ofs7 << endl;
    }
    ofs7 << endl;
    
    return 0;
}