#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

    vector<string> possible_labels = {"left","keep","right"};


    /**
    * Constructor
    */
    GNB();

    /**
    * Destructor
    */
    virtual ~GNB();

    void train(vector<vector<double> > data, vector<string>  labels);

    string predict(vector<double> coords);

private:
    vector<double> getMeanAndVariance(vector<vector<double>> dataSet);
    template <typename T>
    T pdf(T x, T m, T s);
    double getPosterior(vector<double> coords, vector<double> meansAndVariances);

    vector<double> leftMV;
    vector<double> keepMV;
    vector<double> rightMV;
    double total;

};

#endif



