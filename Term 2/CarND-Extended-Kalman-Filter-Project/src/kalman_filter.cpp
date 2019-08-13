#include "kalman_filter.h"
#include <math.h>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;

  x_size = x_.size();
  I = MatrixXd::Identity(x_size, x_size);
  Ht = H_.transpose();
}

void KalmanFilter::Predict() {
   x_ = F_ * x_;
   MatrixXd Ft = F_.transpose();
   P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // for measurement updates with lidar, we can use the H matrix for calculating y, S, K and P.
  // y=z−Hx
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::h_func() {

    double px = x_[0];
    double py = x_[1];
    double vx = x_[2];
    double vy = x_[3];
    double x1 = 0.0;
    double x2 = 0.0;
    double x3 = 0.0;

    if (px < 0.0000001 && py < 0.000001) {
        px = 0.001;
        py = 0.001;
    }
    x1 = sqrt(px*px+py*py);
    x2 = atan2(py, px);

    // make sure x2 is within -pi +py
    while (x2 < M_PI*-1.0) {
        x2 += M_PI;
    }

    while (x2 > M_PI) {
        x2 -= M_PI;
    }

    if (fabs(x1) >= 0.0001) {
        x3 = (px*vx+py*vy)/x1;
    } else {
        // should never go here
        x3 = 0.0;
        std::cout << "Logic error?" << std::endl;
    }

    VectorXd result(3);
    result << x1, x2, x3;
    return result;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // for radar, Hj is used to calculate S, K and P.
  // the equation for radar becomes y=z−h(x').

    VectorXd z_pred = h_func();
    Tools tools;

    // Use px and py from z rather than x_
    // convert polar to cartesiaon
    float x_cart = z[0]*cos(z[1]);
    float y_cart = z[0]*sin(z[1]);
    VectorXd xj = VectorXd(4);
    // create xj using px/py from z, vx/vy from x_
    xj << x_cart, y_cart, x_[2], x_[3];

    H_ = tools.CalculateJacobian(xj);
    VectorXd y = z - z_pred;
    Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    // new estimate
    x_ = x_ + (K * y);
    P_ = (I - K * H_) * P_;
}
