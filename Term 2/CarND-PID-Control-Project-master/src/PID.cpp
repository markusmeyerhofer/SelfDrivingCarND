#include "PID.h"

/*
* TODO: Complete the PID class.
*/

PID::PID() {
  p_error = 0.0;
  i_error = 0.0;
  d_error = 0.0;
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
}

void PID::UpdateError(double cte) {
  d_error = cte - p_error;
  i_error += cte;
  p_error = cte;
}

double PID::TotalError() {
  //return i_error;
  return -1*(Kp * p_error + Kd * d_error + Ki * i_error);
}
