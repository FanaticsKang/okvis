/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Sep 3, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file ImuError.cpp
 * @brief Source file for the ImuError class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <thread>

#include <glog/logging.h>

#include <okvis/Parameters.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/ode/ode.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurements and parameters.
ImuError::ImuError(const okvis::ImuMeasurementDeque &imuMeasurements,
                   const okvis::ImuParameters &imuParameters,
                   const okvis::Time &t_0, const okvis::Time &t_1) {
  setImuMeasurements(imuMeasurements);
  setImuParameters(imuParameters);
  setT0(t_0);
  setT1(t_1);

  OKVIS_ASSERT_TRUE_DBG(
      Exception, t_0 >= imuMeasurements.front().timeStamp,
      "First IMU measurement included in ImuError is not old enough!");
  OKVIS_ASSERT_TRUE_DBG(
      Exception, t_1 <= imuMeasurements.back().timeStamp,
      "Last IMU measurement included in ImuError is not new enough!");
}

// Propagates pose, speeds and biases with given IMU measurements.
int ImuError::redoPreintegration(
    const okvis::kinematics::Transformation & /*T_WS*/,
    const okvis::SpeedAndBias &speedAndBiases) const {
  // ensure unique access
  std::lock_guard<std::mutex> lock(preintegrationMutex_);

  // now the propagation
  okvis::Time time = t0_;
  okvis::Time end = t1_;

  // sanity check:
  assert(imuMeasurements_.front().timeStamp <= time);
  if (!(imuMeasurements_.back().timeStamp >= end))
    return -1;  // nothing to do...

  // increments (initialise with identity)
  Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  C_integral_ = Eigen::Matrix3d::Zero();
  C_doubleintegral_ = Eigen::Matrix3d::Zero();
  acc_integral_ = Eigen::Vector3d::Zero();
  acc_doubleintegral_ = Eigen::Vector3d::Zero();

  // cross matrix accumulatrion
  cross_ = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  dalpha_db_g_ = Eigen::Matrix3d::Zero();
  dv_db_g_ = Eigen::Matrix3d::Zero();
  dp_db_g_ = Eigen::Matrix3d::Zero();

  // the Jacobian of the increment (w/o biases)
  P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

  // Eigen::Matrix<double, 15, 15> F_tot;
  // F_tot.setIdentity();

  double Delta_t = 0;
  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements_.begin();
       it != imuMeasurements_.end(); ++it) {
    //可以认为XXX_S_0是前一时刻的IMU数据, XXX_S_1是当前时刻的IMU数据
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements_.end()) {
      nexttime = t1_;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {  //最后一个时刻,对IMU数据进行内插
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t1_;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;
    //提取Start时间附近的IMU数据, 并且进行内插
    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParameters_.sigma_g_c;
    double sigma_a_c = imuParameters_.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParameters_.g_max ||
        fabs(omega_S_0[1]) > imuParameters_.g_max ||
        fabs(omega_S_0[2]) > imuParameters_.g_max ||
        fabs(omega_S_1[0]) > imuParameters_.g_max ||
        fabs(omega_S_1[1]) > imuParameters_.g_max ||
        fabs(omega_S_1[2]) > imuParameters_.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING) << "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParameters_.a_max ||
        fabs(acc_S_0[1]) > imuParameters_.a_max ||
        fabs(acc_S_0[2]) > imuParameters_.a_max ||
        fabs(acc_S_1[0]) > imuParameters_.a_max ||
        fabs(acc_S_1[1]) > imuParameters_.a_max ||
        fabs(acc_S_1[2]) > imuParameters_.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING) << "acc saturation";
    }

    // actual propagation
    // orientation:
    Eigen::Quaterniond dq;
    // The author quit like use average value in IMU DATA. 0.5 * (.)
    const Eigen::Vector3d omega_S_true =
        (0.5 * (omega_S_0 + omega_S_1) - speedAndBiases.segment<3>(3));
        //delta Quaternion
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = ode::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    // dq is from Formation 4 in essay by Kang
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    // Delta_q_初始化为(1,0,0,0)用于迭代计算两时刻间的旋转
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();  //前一时刻的旋转矩阵
    const Eigen::Matrix3d C_1 =
        Delta_q_1.toRotationMatrix();  //当一时刻的旋转矩阵

    const Eigen::Vector3d acc_S_true =
        (0.5 * (acc_S_0 + acc_S_1) - speedAndBiases.segment<3>(6));
    //这是旋转矩阵C对于时间的积分，
    //作者非常喜欢使用两个时刻的均值作为当前时刻的值。0.5 *(.)，
    //这里可能是考虑到delta_C非常小，如果差值求解，效率慢精度差。
    //这里计算旋转矩阵C对于时间的积分，主要是dv_db_a和dp_db_a计算中需要用到旋转矩阵对于时间的积分，及二次积分。
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;

    //加速度积分-速度
    const Eigen::Vector3d acc_integral_1 =
        acc_integral_ + 0.5 * (C + C_1) * acc_S_true * dt;
    // rotation matrix double integral:
    C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
    //加速度的二倍积分-距离
    acc_doubleintegral_ +=
        acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt;

    // Jacobian parts
    dalpha_db_g_ +=
        C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    // delta_R_db_g，可以在IMU Preintegration的supplement
    // B中找到计算过程。其中注意R_ii为单位阵
    //根据论文计算dv_db_g时，需要时使用delta_R_db_g。
    const Eigen::Matrix3d cross_1 =
        dq.inverse().toRotationMatrix() * cross_ +
        okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 =
        dv_db_g_ + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    dp_db_g_ +=
        dt * dv_db_g_ +
        0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    // covariance propagation
    Eigen::Matrix<double, 15, 15> F_delta =
        Eigen::Matrix<double, 15, 15>::Identity();
    // transform
    // 雅克比矩阵的计算同时参考okvis和IMU Preintegration,
    // okvis中计算雅克比矩阵的形式有误，参考word文件和Preintegration。
    // IMU Preintegration中使用旋转矩阵R构建残差，
    // 而okvis中则使用四元数计算残差，因此对于旋转的求导过程不能参考IMU
    // Preintegration.其余以IMU Preintegration为准。

    //参考okvis, 这里的值本质上是“第一帧坐标系下速度V”的反对称矩阵。又由于Fd = I +
    // Fc*dt，这里又对速度进行了积分。
    // Preintegration中认为这里是“前一帧坐标系下的p”
    F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
        acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
    // Preintegration 如果速度的坐标系为前一帧坐标系，则这里为I。
    F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;

    //参考Preintegration,
    //这里为dp_db_g,是dv_db_g对时间的积分。dv_db_g参考下面(6,9)
    F_delta.block<3, 3>(0, 9) =
        dt * dv_db_g_ +
        0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    //参考Preintegration,
    //这里为dp_db_a,是dv_db_a对时间的积分。dv_db_g参考下面(6,12)
    F_delta.block<3, 3>(0, 12) = -C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;

    //参考okvis，由于okvis使用四元数，其角度的求导和Preintegration不同，幸亏这里和论文没有出入。
    F_delta.block<3, 3>(3, 9) = -dt * C_1;

    //和原文基本相同，只是将dt乘在了Cross里面
    // Preintegration 使用速度okvis则使用加速度的积分。
    F_delta.block<3, 3>(6, 3) =
        -okvis::kinematics::crossMx(0.5 * (C + C_1) * acc_S_true * dt);

    //参考Preintegration, 这里是和论文的主要出入。
    // cross表示角度相对陀螺仪偏差的导数delta_R_db_g。作者非常喜欢用两个时刻的均值表示整个过程。
    F_delta.block<3, 3>(6, 9) =
        0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;
    P_delta_ = F_delta * P_delta_ * F_delta.transpose();
    // add noise. Note that transformations with rotation matrices can be
    // ignored, since the noise is isotropic.
    // F_tot = F_delta*F_tot;
    const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
    P_delta_(3, 3) += sigma2_dalpha;
    P_delta_(4, 4) += sigma2_dalpha;
    P_delta_(5, 5) += sigma2_dalpha;
    const double sigma2_v = dt * sigma_a_c * sigma_a_c;
    P_delta_(6, 6) += sigma2_v;
    P_delta_(7, 7) += sigma2_v;
    P_delta_(8, 8) += sigma2_v;
    const double sigma2_p = 0.5 * dt * dt * sigma2_v;
    P_delta_(0, 0) += sigma2_p;
    P_delta_(1, 1) += sigma2_p;
    P_delta_(2, 2) += sigma2_p;
    const double sigma2_b_g =
        dt * imuParameters_.sigma_gw_c * imuParameters_.sigma_gw_c;
    P_delta_(9, 9) += sigma2_b_g;
    P_delta_(10, 10) += sigma2_b_g;
    P_delta_(11, 11) += sigma2_b_g;
    const double sigma2_b_a =
        dt * imuParameters_.sigma_aw_c * imuParameters_.sigma_aw_c;
    P_delta_(12, 12) += sigma2_b_a;
    P_delta_(13, 13) += sigma2_b_a;
    P_delta_(14, 14) += sigma2_b_a;

    // memory shift
    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == t1_) break;
  }

  // store the reference (linearisation) point
  speedAndBiases_ref_ = speedAndBiases;

  // get the weighting:
  // enforce symmetric
  P_delta_ = 0.5 * P_delta_ + 0.5 * P_delta_.transpose().eval();

  // calculate inverse
  information_ = P_delta_.inverse();
  information_ = 0.5 * information_ + 0.5 * information_.transpose().eval();

  // square root
  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();

  // std::cout << F_tot;

  return i;
}

// Propagates pose, speeds and biases with given IMU measurements.
int ImuError::propagation(const okvis::ImuMeasurementDeque &imuMeasurements,
                          const okvis::ImuParameters &imuParams,
                          okvis::kinematics::Transformation &T_WS,
                          okvis::SpeedAndBias &speedAndBiases,
                          const okvis::Time &t_start, const okvis::Time &t_end,
                          covariance_t *covariance, jacobian_t *jacobian) {
  // now the propagation
  okvis::Time time = t_start;
  okvis::Time end = t_end;

  // sanity check:
  assert(imuMeasurements.front().timeStamp <= time);
  if (!(imuMeasurements.back().timeStamp >= end))
    return -1;  // nothing to do...

  // initial condition
  Eigen::Vector3d r_0 = T_WS.r();
  Eigen::Quaterniond q_WS_0 = T_WS.q();
  Eigen::Matrix3d C_WS_0 = T_WS.C();

  // increments (initialise with identity)
  Eigen::Quaterniond Delta_q(1, 0, 0, 0);
  Eigen::Matrix3d C_integral = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d C_doubleintegral = Eigen::Matrix3d::Zero();
  Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_doubleintegral = Eigen::Vector3d::Zero();

  // cross matrix accumulatrion
  Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_db_g = Eigen::Matrix3d::Zero();

  // the Jacobian of the increment (w/o biases)
  Eigen::Matrix<double, 15, 15> P_delta = Eigen::Matrix<double, 15, 15>::Zero();

  double Delta_t = 0;
  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it != imuMeasurements.end(); ++it) {
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements.end()) {
      nexttime = t_end;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t_end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParams.sigma_g_c;
    double sigma_a_c = imuParams.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParams.g_max ||
        fabs(omega_S_0[1]) > imuParams.g_max ||
        fabs(omega_S_0[2]) > imuParams.g_max ||
        fabs(omega_S_1[0]) > imuParams.g_max ||
        fabs(omega_S_1[1]) > imuParams.g_max ||
        fabs(omega_S_1[2]) > imuParams.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING) << "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParams.a_max ||
        fabs(acc_S_0[1]) > imuParams.a_max ||
        fabs(acc_S_0[2]) > imuParams.a_max ||
        fabs(acc_S_1[0]) > imuParams.a_max ||
        fabs(acc_S_1[1]) > imuParams.a_max ||
        fabs(acc_S_1[2]) > imuParams.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING) << "acc saturation";
    }

    // actual propagation
    // orientation:
    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true =
        (0.5 * (omega_S_0 + omega_S_1) - speedAndBiases.segment<3>(3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = ode::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q * dq;
    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true =
        (0.5 * (acc_S_0 + acc_S_1) - speedAndBiases.segment<3>(6));
    const Eigen::Matrix3d C_integral_1 = C_integral + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 =
        acc_integral + 0.5 * (C + C_1) * acc_S_true * dt;
    // rotation matrix double integral:
    C_doubleintegral += C_integral * dt + 0.25 * (C + C_1) * dt * dt;
    acc_doubleintegral +=
        acc_integral * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt;

    // Jacobian parts
    dalpha_db_g += dt * C_1;
    const Eigen::Matrix3d cross_1 =
        dq.inverse().toRotationMatrix() * cross +
        okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 =
        dv_db_g + 0.5 * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
    dp_db_g += dt * dv_db_g +
               0.25 * dt * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);

    // covariance propagation
    if (covariance) {
      Eigen::Matrix<double, 15, 15> F_delta =
          Eigen::Matrix<double, 15, 15>::Identity();
      // transform
      F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
          acc_integral * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
      F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
      F_delta.block<3, 3>(0, 9) =
          dt * dv_db_g +
          0.25 * dt * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
      F_delta.block<3, 3>(0, 12) =
          -C_integral * dt + 0.25 * (C + C_1) * dt * dt;
      F_delta.block<3, 3>(3, 9) = -dt * C_1;
      F_delta.block<3, 3>(6, 3) =
          -okvis::kinematics::crossMx(0.5 * (C + C_1) * acc_S_true * dt);
      F_delta.block<3, 3>(6, 9) =
          0.5 * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
      F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;
      P_delta = F_delta * P_delta * F_delta.transpose();
      // add noise. Note that transformations with rotation matrices can be
      // ignored, since the noise is isotropic.
      // F_tot = F_delta*F_tot;
      const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
      P_delta(3, 3) += sigma2_dalpha;
      P_delta(4, 4) += sigma2_dalpha;
      P_delta(5, 5) += sigma2_dalpha;
      const double sigma2_v = dt * sigma_a_c * imuParams.sigma_a_c;
      P_delta(6, 6) += sigma2_v;
      P_delta(7, 7) += sigma2_v;
      P_delta(8, 8) += sigma2_v;
      const double sigma2_p = 0.5 * dt * dt * sigma2_v;
      P_delta(0, 0) += sigma2_p;
      P_delta(1, 1) += sigma2_p;
      P_delta(2, 2) += sigma2_p;
      const double sigma2_b_g =
          dt * imuParams.sigma_gw_c * imuParams.sigma_gw_c;
      P_delta(9, 9) += sigma2_b_g;
      P_delta(10, 10) += sigma2_b_g;
      P_delta(11, 11) += sigma2_b_g;
      const double sigma2_b_a =
          dt * imuParams.sigma_aw_c * imuParams.sigma_aw_c;
      P_delta(12, 12) += sigma2_b_a;
      P_delta(13, 13) += sigma2_b_a;
      P_delta(14, 14) += sigma2_b_a;
    }

    // memory shift
    Delta_q = Delta_q_1;
    C_integral = C_integral_1;
    acc_integral = acc_integral_1;
    cross = cross_1;
    dv_db_g = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == t_end) break;
  }

  // actual propagation output:
  const Eigen::Vector3d g_W =
      imuParams.g * Eigen::Vector3d(0, 0, 6371009).normalized();
  T_WS.set(
      r_0 + speedAndBiases.head<3>() * Delta_t +
          C_WS_0 *
              (acc_doubleintegral /*-C_doubleintegral*speedAndBiases.segment<3>(6)*/) -
          0.5 * g_W * Delta_t * Delta_t,
      q_WS_0 * Delta_q);
  speedAndBiases.head<3>() +=
      C_WS_0 * (acc_integral /*-C_integral*speedAndBiases.segment<3>(6)*/) -
      g_W * Delta_t;

  // assign Jacobian, if requested
  if (jacobian) {
    Eigen::Matrix<double, 15, 15> &F = *jacobian;
    F.setIdentity();  // holds for all states, including d/dalpha, d/db_g,
                      // d/db_a
    F.block<3, 3>(0, 3) =
        -okvis::kinematics::crossMx(C_WS_0 * acc_doubleintegral);
    F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * Delta_t;
    F.block<3, 3>(0, 9) = C_WS_0 * dp_db_g;
    F.block<3, 3>(0, 12) = -C_WS_0 * C_doubleintegral;
    F.block<3, 3>(3, 9) = -C_WS_0 * dalpha_db_g;
    F.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(C_WS_0 * acc_integral);
    F.block<3, 3>(6, 9) = C_WS_0 * dv_db_g;
    F.block<3, 3>(6, 12) = -C_WS_0 * C_integral;
  }

  // overall covariance, if requested
  if (covariance) {
    Eigen::Matrix<double, 15, 15> &P = *covariance;
    // transform from local increments to actual states
    Eigen::Matrix<double, 15, 15> T = Eigen::Matrix<double, 15, 15>::Identity();
    T.topLeftCorner<3, 3>() = C_WS_0;
    T.block<3, 3>(3, 3) = C_WS_0;
    T.block<3, 3>(6, 6) = C_WS_0;
    P = T * P_delta * T.transpose();
  }
  return i;
}

// This evaluates the error term and additionally computes the Jacobians.
bool ImuError::Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool ImuError::EvaluateWithMinimalJacobians(double const *const *parameters,
                                            double *residuals,
                                            double **jacobians,
                                            double **jacobiansMinimal) const {
  // get poses
  // paramters[0] is robot pose, the position and quaterniond. by Kang
  // parameters[1] is SpeedAndBias,[v, b_g, b_a]. by Kang
  const okvis::kinematics::Transformation T_WS_0(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]));

  const okvis::kinematics::Transformation T_WS_1(
      Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
      Eigen::Quaterniond(parameters[2][6], parameters[2][3], parameters[2][4],
                         parameters[2][5]));

  // get speed and bias
  okvis::SpeedAndBias speedAndBiases_0;
  okvis::SpeedAndBias speedAndBiases_1;
  for (size_t i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
    speedAndBiases_1[i] = parameters[3][i];
  }

  // this will NOT be changed:
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  // call the propagation
  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b;
  // ensure unique access
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    Delta_b = speedAndBiases_0.tail<6>() - speedAndBiases_ref_.tail<6>();
  }

  // Kang
  // speedAndBiases_ref_ is the point of taylor expansion. And it is related to
  // many modules in Jacobian matrix.
  // So when Delta_Bias_g * Delta_t is too big, it will recompute.
  // redoPreintegration will recompute the tayor expansion in the new point of
  // Biases and their related modules.
  redo_ = redo_ || (Delta_b.head<3>().norm() * Delta_t > 0.0001);
  if (redo_) {
    redoPreintegration(T_WS_0, speedAndBiases_0);
    redoCounter_++;
    Delta_b.setZero();
    redo_ = false;
    /*if (redoCounter_ > 1) {
  std::cout << "pre-integration no. " << redoCounter_ << std::endl;
}*/
  }

  // actual propagation output:
  // the point of this function is to compute the residuals and jacobian. by
  // Kang
  // The computation will go through both T0 and T1.
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);  // this is a bit
                                                             // stupid, but
                                                             // shared
                                                             // read-locks only
                                                             // come in C++14
    const Eigen::Vector3d g_W =
        imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    // assign Jacobian w.r.t. x0
    Eigen::Matrix<double, 15, 15> F0 =
        Eigen::Matrix<double, 15, 15>::Identity();  // holds for d/db_g, d/db_a

    // all in the world coordinate. by Kang
    const Eigen::Vector3d delta_p_est_W = T_WS_0.r() - T_WS_1.r() +
                                          speedAndBiases_0.head<3>() * Delta_t -
                                          0.5 * g_W * Delta_t * Delta_t;
    const Eigen::Vector3d delta_v_est_W =
        speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>() - g_W * Delta_t;
    const Eigen::Quaterniond Dq =
        okvis::kinematics::deltaQ(-dalpha_db_g_ * Delta_b.head<3>()) * Delta_q_;

    F0.block<3, 3>(0, 0) =
        C_S0_W;  //                                                                                                                                                                                                                                                                                                                                                                                                                                      ;
    F0.block<3, 3>(0, 3) = C_S0_W * okvis::kinematics::crossMx(delta_p_est_W);
    F0.block<3, 3>(0, 6) = C_S0_W * Eigen::Matrix3d::Identity() * Delta_t;
    F0.block<3, 3>(0, 9) = dp_db_g_;
    F0.block<3, 3>(0, 12) = -C_doubleintegral_;
    F0.block<3, 3>(3, 3) = (okvis::kinematics::plus(Dq * T_WS_1.q().inverse()) *
                            okvis::kinematics::oplus(T_WS_0.q()))
                               .topLeftCorner<3, 3>();
    F0.block<3, 3>(3, 9) =
        (okvis::kinematics::oplus(T_WS_1.q().inverse() * T_WS_0.q()) *
         okvis::kinematics::oplus(Dq))
            .topLeftCorner<3, 3>() *
        (-dalpha_db_g_);
    F0.block<3, 3>(6, 3) = C_S0_W * okvis::kinematics::crossMx(delta_v_est_W);
    F0.block<3, 3>(6, 6) = C_S0_W;
    F0.block<3, 3>(6, 9) = dv_db_g_;
    F0.block<3, 3>(6, 12) = -C_integral_;

    // assign Jacobian w.r.t. x1
    Eigen::Matrix<double, 15, 15> F1 =
        -Eigen::Matrix<double, 15, 15>::Identity();  // holds for the biases
    F1.block<3, 3>(0, 0) = -C_S0_W;
    F1.block<3, 3>(3, 3) =
        -(okvis::kinematics::plus(Dq) * okvis::kinematics::oplus(T_WS_0.q()) *
          okvis::kinematics::plus(T_WS_1.q().inverse()))
             .topLeftCorner<3, 3>();
    F1.block<3, 3>(6, 6) = -C_S0_W;

    // the overall error vector
    /*Kang
     the error in okvis is under the IMU sensor coordinate.
     delta_p_est_W is compute by pose and speed without IMU data.
     Here error is delta_p_est_W adding IMU inform, including acc and bias.
     Because Delta_b is not compute every time, so it use first taylor expansion
     here. The detail is in Preintegration VI-C.
    */
    Eigen::Matrix<double, 15, 1> error;
    error.segment<3>(0) = C_S0_W * delta_p_est_W + acc_doubleintegral_ +
                          F0.block<3, 6>(0, 9) * Delta_b;
    error.segment<3>(3) =
        2 * (Dq * (T_WS_1.q().inverse() * T_WS_0.q()))
                .vec();  // 2*T_WS_0.q()*Dq*T_WS_1.q().inverse();//
    error.segment<3>(6) =
        C_S0_W * delta_v_est_W + acc_integral_ + F0.block<3, 6>(6, 9) * Delta_b;
    error.tail<6>() = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();

    // error weighting
    Eigen::Map<Eigen::Matrix<double, 15, 1>> weighted_error(residuals);
    weighted_error = squareRootInformation_ * error;

    // get the Jacobians
    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {  // jacobian[0] is for pose.
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, 15, 6> J0_minimal =
            squareRootInformation_ * F0.block<15, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseLocalParameterization::liftJacobian(parameters[0], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        // the delta in PoseLocalParameterization is pose + quaterniond, Here to
        // lift. by Kang
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J0(
            jacobians[0]);
        J0 = J0_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>>
                J0_minimal_mapped(jacobiansMinimal[0]);
            J0_minimal_mapped = J0_minimal;
          }
        }
      }
      if (jacobians[1] != NULL) {  // jacobian[1] is for speedAndBias.
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J1(
            jacobians[1]);
        J1 = squareRootInformation_ * F0.block<15, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[1] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>>
                J1_minimal_mapped(jacobiansMinimal[1]);
            J1_minimal_mapped = J1;
          }
        }
      }
      if (jacobians[2] != NULL) {
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, 15, 6> J2_minimal =
            squareRootInformation_ * F1.block<15, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseLocalParameterization::liftJacobian(parameters[2], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J2(
            jacobians[2]);
        J2 = J2_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[2] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>>
                J2_minimal_mapped(jacobiansMinimal[2]);
            J2_minimal_mapped = J2_minimal;
          }
        }
      }
      if (jacobians[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J3(
            jacobians[3]);
        J3 = squareRootInformation_ * F1.block<15, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[3] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>>
                J3_minimal_mapped(jacobiansMinimal[3]);
            J3_minimal_mapped = J3;
          }
        }
      }
    }
  }
  return true;
}

}  // namespace ceres
}  // namespace okvis
