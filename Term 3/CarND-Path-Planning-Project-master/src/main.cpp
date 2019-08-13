#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2) {
    return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y) {

    double closestLen = 100000; //large number
    int closestWaypoint = 0;

    for(int i = 0; i < maps_x.size(); i++) {
        double map_x = maps_x[i];
        double map_y = maps_y[i];
        double dist = distance(x,y,map_x,map_y);
        if(dist < closestLen) {
            closestLen = dist;
            closestWaypoint = i;
        }
    }
    return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y) {

    int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

    double map_x = maps_x[closestWaypoint];
    double map_y = maps_y[closestWaypoint];

    double heading = atan2( (map_y-y),(map_x-x) );

    double angle = abs(theta-heading);

    if(angle > pi()/4) {
        closestWaypoint++;
    }
    return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y) {
    int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

    int prev_wp;
    prev_wp = next_wp-1;
    if(next_wp == 0) {
        prev_wp  = maps_x.size()-1;
    }

    double n_x = maps_x[next_wp]-maps_x[prev_wp];
    double n_y = maps_y[next_wp]-maps_y[prev_wp];
    double x_x = x - maps_x[prev_wp];
    double x_y = y - maps_y[prev_wp];

    // find the projection of x onto n
    double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
    double proj_x = proj_norm*n_x;
    double proj_y = proj_norm*n_y;

    double frenet_d = distance(x_x,x_y,proj_x,proj_y);

    //see if d value is positive or negative by comparing it to a center point

    double center_x = 1000-maps_x[prev_wp];
    double center_y = 2000-maps_y[prev_wp];
    double centerToPos = distance(center_x,center_y,x_x,x_y);
    double centerToRef = distance(center_x,center_y,proj_x,proj_y);

    if(centerToPos <= centerToRef) {
        frenet_d *= -1;
    }

    // calculate s value
    double frenet_s = 0;
    for(int i = 0; i < prev_wp; i++) {
        frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
    }

    frenet_s += distance(0,0,proj_x,proj_y);

    return {frenet_s,frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y) {
    int prev_wp = -1;

    while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) )) {
        prev_wp++;
    }

    int wp2 = (prev_wp+1)%maps_x.size();

    double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
    // the x,y,s along the segment
    double seg_s = (s-maps_s[prev_wp]);

    double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
    double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

    double perp_heading = heading-pi()/2;

    double x = seg_x + d*cos(perp_heading);
    double y = seg_y + d*sin(perp_heading);

    return {x,y};
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // the car is initially in the middle lane
  int lane = 1;
  // reference velocity, slightly below maximum allowed speed
  double ref_vel = 0;
  double look_ahead = 800.0;
  int lane_change_damping = 0; // no lane changes for first 2 seconds

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy, &lane_change_damping, &look_ahead, &ref_vel, &lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;

    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

            // Main car's localization Data
            double car_x = j[1]["x"];
            double car_y = j[1]["y"];
            double car_s = j[1]["s"];
            double car_d = j[1]["d"];
            double car_yaw = j[1]["yaw"];
            double car_speed = j[1]["speed"];

            // Previous path data given to the Planner
            auto previous_path_x = j[1]["previous_path_x"];
            auto previous_path_y = j[1]["previous_path_y"];
            // Previous path's end s and d values
            double end_path_s = j[1]["end_path_s"];
            double end_path_d = j[1]["end_path_d"];

            // Sensor Fusion Data, a list of all other cars on the same side of the road.
            auto sensor_fusion = j[1]["sensor_fusion"];

            int prev_size = previous_path_x.size();

            json msgJson;

            vector<double> next_x_vals;
            vector<double> next_y_vals;

            // Avoid collisions

            if (prev_size > 0) {
                car_s = end_path_s;
            }

            bool too_close = false;
            bool avoid_left = false;
            bool avoid_right = false;
            double cost_per_lane[]={0.0, 0.0, 0.0};

            cost_per_lane[0] = 0.0;
            cost_per_lane[1] = 0.0;
            cost_per_lane[2] = 0.0;

            // ###
            // ### BEGIN path planning section
            // ###

            for (int i = 0; i < sensor_fusion.size(); i++) {
                float d = sensor_fusion[i][6];

                double vx = sensor_fusion[i][3];
                double vy = sensor_fusion[i][4];
                double check_speed = sqrt(pow(vx,2) + pow(vy,2));
                double check_car_pred2 = sensor_fusion[i][5];

                if (check_car_pred2 - car_s <= look_ahead) { // disregard cars out of sight
                    check_car_pred2 += ((double)prev_size*.02*check_speed);

                    double cost = 0;

                    if (check_car_pred2 > car_s) { // looking at cars ahead only, disregarding cars behind
                        cost += look_ahead-(check_car_pred2-car_s);
                        int lane_index = -1;
                        if (d < 4) {
                            lane_index = 0; // car in lane 0
                        }
                        if (d > 8) {
                            lane_index = 2; // car in lane 2
                        }
                        if (d >= 4 && d <= 8) {
                            lane_index = 1; // car in lane 1
                        }

                        if (lane != lane_index) { // increase cost if lane change required
                            cost += 1000;
                        }
                        assert(lane_index != -1);

                        if (cost > cost_per_lane[lane_index]) { // set the highest car cost only for each lane
                            cost_per_lane[lane_index] = cost; // add the computed cost to the respective lane
                        }
                    }

                    if (d < (2+4*lane+2) && (d > (2+4*lane -2 ))) {
                        if ((check_car_pred2 > car_s) && (check_car_pred2-car_s < 30)) { // avoid rear-ending
                            too_close = true;
                            cost_per_lane[lane] += 1000;
                        }
                    } else {
                        double distance_s = car_s - check_car_pred2;
                        double distance_d = car_d - d;

                        double distance = sqrt(pow(distance_s, 2) +  pow(distance_d, 2));

                        if (check_car_pred2 > car_s && abs(distance_s) < 30 && check_speed < car_speed) { // check for slower cars ahead
                            if (distance_d > -7.5 && distance_d < -1.5) {avoid_right = true;}
                            if (distance_d < 7.5 && distance_d > 1.5) {avoid_left = true;}
                            cout << "Slow car ahead" << endl;
                        } else
                        if (check_car_pred2 < car_s && abs(distance_s) < 40 && check_speed > car_speed) { // check for faster cars from behind
                            avoid_right = true;
                            avoid_left = true;
                            cout << "Fast car from behind" << endl;
                        }
                        if (abs(distance) < 15) {                                           // don't bump into nearby cars
                            if (distance_d > -7.5 && distance_d < -1.5) {avoid_right = true;}
                            if (distance_d < 7.5 && distance_d > 1.5) {avoid_left = true;}
                            cout << "Watch out for car next lane" << endl;
                        }
                    }
                }
            }

            int cheapest_i = 0;
            double cheapest_lane = 100000;
            for(int i = 0; i < 3; i++)             {
                if(cost_per_lane[i] <  cheapest_lane) {
                    cheapest_i = i;
                    cheapest_lane = cost_per_lane[i];
                }
            }

            //cout << "cost for lanes: " << cost_per_lane[0] << " " << cost_per_lane[1] << " " << cost_per_lane[2] << " cheapest: " << cheapest_i << endl;
            if (avoid_left) {cout << "avoiding left" << endl;}
            if (avoid_right) {cout << "avoiding right" << endl;}
            cout << "Target lane: " << cheapest_i << endl;

            //change lanes only if no car nearby
            lane_change_damping += 1;
            if (cheapest_i < lane) {
                if (!avoid_left && lane_change_damping > 100) { // wait at least 1 second1 before changing lanes again
                    lane -= 1;
                    lane_change_damping = 0; // set back to zero
                }
            }
            if (cheapest_i > lane) {
                if (!avoid_right && lane_change_damping > 100) {
                    lane += 1;
                    lane_change_damping = 0; // set back to zero
                }
            }

            // ###
            // ### END path planning section
            // ###

            // ###
            // ### avoid bumping into cars
            // ###

            if (too_close) {
                ref_vel -= .224;

            } else {
                if (ref_vel < 49.5) {
                   ref_vel += .224;
                }
            }

            // ###
            // ### Compute trajectory points
            // ###

            // create vectors of spaced (x, y) waypoints, evently spaced at 30 m to later interpolate using splines
            vector<double> ptsx;
            vector<double> ptsy;

            // ref x, y and yaw
            // referencing eihter the start point or the previous paths endpoint
            double ref_x = car_x;
            double ref_y = car_y;
            double ref_yaw = deg2rad(car_yaw);

            if (prev_size < 2) { // if previous path is almost empty, use car pos as reference

                // create tangent to car
                double prev_car_x = car_x - cos(car_yaw);
                double prev_car_y = car_y - sin(car_yaw);

                if (prev_car_x > car_x) {
                    ptsx.push_back(car_x);
                    ptsx.push_back(prev_car_x);
                    ptsy.push_back(car_y);
                    ptsy.push_back(prev_car_y);
                } else {
                    if (prev_car_x == car_x) {
                       car_x += 0.0001;
                    }
                    ptsx.push_back(prev_car_x);
                    ptsx.push_back(car_x);
                    ptsy.push_back(prev_car_y);
                    ptsy.push_back(car_y);
                }

            } else { // else use the previous paths's endpoint as starting reference
                // new reference are previous path's endpoints
                ref_x = previous_path_x[prev_size-1];
                ref_y = previous_path_y[prev_size-1];

                double ref_x_prev = previous_path_x[prev_size-2];
                double ref_y_prev = previous_path_y[prev_size-2];
                ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

                // use two points making the path tangent to previous path's endpoints
                if (ref_x_prev > ref_x) {
                    ptsx.push_back(ref_x);
                    ptsx.push_back(ref_x_prev);
                     ptsy.push_back(ref_y);
                    ptsy.push_back(ref_y_prev);
                } else {
                    if (ref_x_prev == ref_x) {
                        ref_x += 0.0001;
                    }
                    ptsx.push_back(ref_x_prev);
                    ptsx.push_back(ref_x);
                    ptsy.push_back(ref_y_prev);
                    ptsy.push_back(ref_y);
                }
            }

            vector<double> next_wp0 = getXY(car_s + 30, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> next_wp1 = getXY(car_s + 60, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> next_wp2 = getXY(car_s + 90, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

            ptsx.push_back(next_wp0[0]);
            ptsx.push_back(next_wp1[0]);
            ptsx.push_back(next_wp2[0]);

            ptsy.push_back(next_wp0[1]);
            ptsy.push_back(next_wp1[1]);
            ptsy.push_back(next_wp2[1]);

            for (int i = 0; i < ptsy.size(); i++) {
                // shift reference angle to 0 degrees
                double shift_x = ptsx[i]-ref_x;
                double shift_y = ptsy[i]-ref_y;

                ptsx[i] = (shift_x * cos(0-ref_yaw) - shift_y * sin(0-ref_yaw));
                ptsy[i] = (shift_x * sin(0-ref_yaw) + shift_y * cos(0-ref_yaw));
            }

            // create a spline

            tk::spline s;

            // set x, y coordinates
            std::sort(ptsx.begin(), ptsx.end()); // avoid assertion error. Should sort y too, but delta x is negligible, so I ignore
            s.set_points(ptsx, ptsy);

            // start with previous path points

            for (int i = 0; i < previous_path_x.size(); i++) {
                next_x_vals.push_back(previous_path_x[i]);
                next_y_vals.push_back(previous_path_y[i]);
            }

            // calculate how to break up spline points to approach target velocity
            double target_x = 30.0;
            double target_y = s(target_x);
            double target_dist = sqrt((target_x*target_x)+(target_y+target_y));
            double x_add_on = 0.0;

            double dist_inc = 0.5;

            // fill up the path to always output 50 points
            for(int i = 0; i < 50-previous_path_x.size(); i++) {

                double N = (target_dist/(.02*ref_vel/2.24));
                double x_point = x_add_on + (target_x/N);
                double y_point = s(x_point);

                x_add_on = x_point;

                double x_ref = x_point;
                double y_ref = y_point;

                // rotate back to global coordinates
                x_point = (x_ref * cos(ref_yaw)-y_ref*sin(ref_yaw));
                y_point = (x_ref * sin(ref_yaw)+y_ref*cos(ref_yaw));

                x_point += ref_x;
                y_point += ref_y;

                next_x_vals.push_back(x_point);
                next_y_vals.push_back(y_point);
            }

            msgJson["next_x"] = next_x_vals;
            msgJson["next_y"] = next_y_vals;

            auto msg = "42[\"control\","+ msgJson.dump()+"]";

            //this_thread::sleep_for(chrono::milliseconds(1000));
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}