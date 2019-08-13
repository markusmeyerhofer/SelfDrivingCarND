#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;
  Ht = H_.transpose();

  x_size = x_.size();
  I = MatrixXd::Identity(x_size, x_size);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3; // TODO!

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7 * M_PI; // 2*M_PI TODO

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ =  0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  previous_timestamp_ = 0;
  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;
  n_z_radar_ = 3;
  n_z_lidar_ = 4;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  time_us_ = 0;

  lambda_ =  3 - n_aug_;
  NIS_radar_ = 0;
  NIS_laser_ = 0;

  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i < 2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2], 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;

  // avoid nan and inf for P_
  while (delta_t > 0.1)
  {
  const double dt = 0.05;
  this->Predict(dt);
  delta_t -= dt;
  }
  this->Predict(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      this->UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      this->UpdateLidar(meas_package);
  }
}

void UKF::Predict(double delta_t) {

  // Generate Sigma Points
  // Predict Sigma Points
  // Predict Mean and Covariance

  this->SigmaPointPrediction(delta_t);
  this->PredictMeanAndCovariance();
}

void UKF::PredictMeanAndCovariance() {

  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i < 2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
      //angle normalization
      if (x_diff(3) > M_PI || x_diff(3) < -M_PI) {
          x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));
      }
      P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  z_ = VectorXd(2);
  z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

  Tools tools;
  VectorXd xj = VectorXd(4);
  xj << z_[0], z_[1], x_[2], x_[3];
  //H_ = tools.CalculateJacobian(xj);
  MatrixXd R_ = MatrixXd(2, 2);
  R_ << 0.0225, 0,
        0, 0.0225;

  z_pred_ = H_ * x_;
  VectorXd y = z_ - z_pred_;
  S_ = H_ * P_ * Ht + R_;
  MatrixXd Si = S_.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

    // new estimate
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;

  NIS_laser_ = z_pred_.transpose() * S_.inverse() * z_pred_;

  //cout << "NIS Lidar: " << NIS_laser_ << endl;
}

MatrixXd UKF::PredictRadarMeasurement() {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    if (p_x == 0 && p_y == 0) {
      Zsig(0,i) = 0.0;
      Zsig(1,i) = 0.0;
      Zsig(2,i) = 0.0;
    } else {
      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
      Zsig(1,i) = atan2(p_y, p_x);
      Zsig(2,i) = (p_x*v1 + p_y*v2)/sqrt(p_x*p_x + p_y*p_y);
    }
  }

  //mean predicted measurement
  z_pred_ = VectorXd(n_z_radar_);
  z_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_ + 1; i++) {
      z_pred_ = z_pred_ + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S_ = MatrixXd(n_z_radar_,n_z_radar_);
  S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred_;

    //angle normalization
    if (z_diff(1)> M_PI || z_diff(1)<-M_PI) {
      z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));
    }

    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_radar_,n_z_radar_);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S_ = S_ + R;

  //print result
  //std::cout << "z_pred: " << std::endl << z_pred_ << std::endl;
  //std::cout << "S: " << std::endl << S_ << std::endl;
  return Zsig;
}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  z_ = VectorXd(3);
  z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];

  MatrixXd Zsig_ = PredictRadarMeasurement();

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    //angle normalization
    if (z_diff(1) > M_PI || z_diff(1)<-M_PI) {
        z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));
    }

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    if (x_diff(3) > M_PI || x_diff(3) < -M_PI) {
        x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));
     }

     Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
   }

   //Kalman gain K;
   MatrixXd K = Tc * S_.inverse();

   //residual
   VectorXd z_diff = z_ - z_pred_;

   //angle normalization
   if (z_diff(1) > M_PI || z_diff(1) < -M_PI) {
       z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));
   }

   //update state mean and covariance matrix
   x_ = x_ + K * z_diff;
   P_ = P_ - K*S_*K.transpose();

   MatrixXd Si = S_.inverse();
   VectorXd x_toRadar = ToRadarSpace(x_);
   x_toRadar << x_[0], x_[1], x_[2];
   VectorXd z_NIS = z_ - x_toRadar;
   NIS_radar_ = z_NIS.transpose()*Si*z_NIS;
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // check for numerical consistency
  Eigen::LLT<MatrixXd> lltOfPaug(P_aug);
  if (lltOfPaug.info() == Eigen::NumericalIssue) {
      // if decomposition fails, we have numerical issues
      std::cout << "LLT failed!" << std::endl;
      //Eigen::EigenSolver<MatrixXd> es(P_aug);
      //cout << "Eigenvalues of P_aug:" << endl << es.eigenvalues() << endl;
      throw std::range_error("LLT failed");
  }

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)   {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(double delta_t) {

  MatrixXd Xsig_aug = MatrixXd(n_x_, 2 * n_aug_ + 1);

  this->AugmentedSigmaPoints(&Xsig_aug);
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

MatrixXd UKF::ToRadarSpace(MatrixXd cartesian) {
  double p_x=0, p_y=0, v=0, yaw=0, rho=0, phi=0;
  p_x = cartesian(0);
  p_y = cartesian(1);
  v = cartesian(2);
  yaw = cartesian(3);
  rho = sqrt(p_x*p_x + p_y*p_y);
  VectorXd radar(3);
  radar(0) = rho;
  phi = atan2(p_y, p_x);
  radar(1) = phi;
  if (fabs(rho) < 0.0001) {
      radar(2) = 0;
  } else {
      radar(2) = (p_x*cos(yaw) + p_y*sin(yaw))*v/rho;
  }
  return radar;
}

MatrixXd UKF::ToLaserSpace(MatrixXd cartesian) {
  double p_x=0, p_y=0, v=0, yaw=0, rho=0, phi=0;
  p_x = cartesian(0);
  p_y = cartesian(1);
  v = cartesian(2);
  yaw = cartesian(3);
  rho = sqrt(p_x*p_x + p_y*p_y);
  VectorXd radar(3);
  radar(0) = rho;
  phi = atan2(p_y, p_x);
  radar(1) = phi;
  if (fabs(rho) < 0.0001) {
      radar(2) = 0;
  } else {
      radar(2) = (p_x*cos(yaw) + p_y*sin(yaw))*v/rho;
  }
  return radar;
}
