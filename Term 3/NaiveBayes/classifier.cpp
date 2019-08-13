#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <stdlib.h>     /* fabs */
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels) {

    /*
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d,
            s_dot and d_dot.
          - Example : [
                [3.5, 0.1, 5.9, -0.02],
                [8.0, -0.3, 3.0, 2.2],
                ...
            ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
    */

    total = data.size();;
    int index = 0;

    vector<vector<double>> left;
    vector<vector<double>> keep;
    vector<vector<double>> right;

    for(auto const& label: labels) {
        if (label == "left") {
            left.push_back(data[index]);
        }
        if (label == "keep") {
            keep.push_back(data[index]);
        }
        if (label == "right") {
            right.push_back(data[index]);
        }
        index++;
    }

    leftMV = getMeanAndVariance(left);
    keepMV = getMeanAndVariance(keep);
    rightMV = getMeanAndVariance(right);
}

string GNB::predict(vector<double> coords) {
    /*
        Once trained, this method is called and expected to return
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """
        # TODO - complete this
    */

    double posterior_left = getPosterior(coords, leftMV);
    double posterior_keep = getPosterior(coords, keepMV);
    double posterior_right = getPosterior(coords, rightMV);

    int turnIndex = 1;
    if (posterior_left > posterior_keep && posterior_left > posterior_right) turnIndex = 0;
    if (posterior_right > posterior_keep && posterior_right > posterior_left) turnIndex = 2;
    if (posterior_keep >= posterior_left && posterior_keep >= posterior_left) turnIndex = 1;

    return this->possible_labels[turnIndex];
}

double GNB::getPosterior(vector<double> coords, vector<double> meansAndVariances) {
    double P_s = pdf(coords[0], meansAndVariances[0], meansAndVariances[4]);
    double P_d = pdf(coords[1], meansAndVariances[1], meansAndVariances[5]);
    double P_s_dot = pdf(coords[2], meansAndVariances[2], meansAndVariances[6]);
    double P_d_dot = pdf(coords[3], meansAndVariances[3], meansAndVariances[7]);

    return P_s * P_d * P_s_dot * P_d_dot;
}

vector<double> GNB::getMeanAndVariance(vector<vector<double>> dataSet) {
    double s_sum = 0.0;
    double d_sum = 0.0;
    double s_dot_sum = 0.0;
    double d_dot_sum = 0.0;
    int numberOfItems = dataSet.size();

    for(auto const& vehicle_data: dataSet) {
        s_sum += vehicle_data[0];
        d_sum += vehicle_data[1];
        s_dot_sum += vehicle_data[2];
        d_dot_sum += vehicle_data[3];
    }
    double s_mean = (double)s_sum/numberOfItems;
    double d_mean = (double)d_sum/numberOfItems;
    double s_dot_mean = (double)s_dot_sum/numberOfItems;
    double d_dot_mean = (double)d_dot_sum/numberOfItems;

    double s_variance = 0;
    double d_variance = 0;
    double s_dot_variance = 0;
    double d_dot_variance = 0;

    for(auto const& vehicle_data: dataSet) {
        s_variance += pow(s_mean-vehicle_data[0], 2);
        d_variance += pow(d_mean-vehicle_data[1], 2);
        s_dot_variance += pow(s_dot_mean-vehicle_data[2], 2);
        d_dot_variance += pow(d_dot_mean-vehicle_data[3], 2);
    }
    s_variance /= numberOfItems;
    d_variance /= numberOfItems;
    s_dot_variance /= numberOfItems;
    d_dot_variance /= numberOfItems;

    vector<double> values;
    values.push_back(s_mean);
    values.push_back(d_mean);
    values.push_back(s_dot_mean);
    values.push_back(d_dot_mean);
    values.push_back(s_variance);
    values.push_back(d_variance);
    values.push_back(s_dot_variance);
    values.push_back(d_dot_variance);

    return values;
}

template <typename T>
T GNB::pdf(T x, T m, T s) {
    static const T inv_sqrt_2pi_s = 1/sqrt(2*M_PI*s);
    T a = pow(x - m, 2) / s;
    return inv_sqrt_2pi_s * exp(-a);
}
