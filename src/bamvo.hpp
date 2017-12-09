/*****************************************************************************
*
* Copyright (c) 2016, Deok-Hwa Kim
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. Neither the name of the KAIST nor the
*    names of its contributors may be used to endorse or promote products
*    derived from this software without specific prior written permission.
* 4. It does not guarantee any patent licenses.
*
* THIS SOFTWARE IS PROVIDED BY DEOK-HWA KIM ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL DEOK-HWA KIM BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
********************************************************************************/

#ifndef __BAMVO_HPP__
#define __BAMVO_HPP__

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>

#include <xmmintrin.h>

#include <tbb/tbb.h>

#include <iostream>
#include <vector>
#include <list>
#include <deque>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cfenv>


#include "parameter.hpp"
#include "rgbd_image.hpp"

namespace goodguy {
class bamvo {
public:
    bamvo() : m_global_pose(Eigen::Matrix4f::Identity()), m_velocity(Eigen::Matrix4f::Identity()) { }

    Eigen::Matrix4f add(const cv::Mat& color, const cv::Mat& depth) {
        Eigen::Matrix4f odometry = Eigen::Matrix4f::Identity();
        if(depth.type() != CV_32FC1 || color.type() != CV_8UC3) {
            std::cerr << "Not supported data type!" << std::endl;
            return odometry;
        }

        if(m_curr_rgbd_pyramid.size() != 0) {
            m_hist_poses.emplace_back(std::make_shared<Eigen::Matrix4f>(Eigen::Matrix4f(Eigen::Matrix4f::Identity())));
            m_hist_depth.emplace_back(m_curr_rgbd_pyramid[m_param.bgm_level]->get_depth());
            m_hist_intensity.emplace_back(m_curr_rgbd_pyramid[m_param.bgm_level]->get_intensity());

            if(m_hist_depth.size() > m_param.hist_size) {
                m_hist_intensity.pop_front();
                m_hist_depth.pop_front();
                m_hist_poses.pop_front();
            }
        }

        std::swap(m_curr_rgbd_pyramid, m_prev_rgbd_pyramid);
        m_curr_rgbd_pyramid.clear();

        cv::Mat curr_gray;
        cv::cvtColor(color, curr_gray, CV_BGR2GRAY);
        cv::Mat curr_intensity;
        curr_gray.convertTo(curr_intensity, CV_32FC1, 1.0/255.0);

        std::shared_ptr<Eigen::MatrixXf> curr_intensity_eig = std::make_shared<Eigen::MatrixXf>(cv2eigen(curr_intensity));
        std::shared_ptr<Eigen::MatrixXf> curr_depth_eig = std::make_shared<Eigen::MatrixXf>(cv2eigen(depth));
        std::shared_ptr<Eigen::MatrixXf> curr_derivative_x = calculate_derivative_x(curr_intensity_eig);
        std::shared_ptr<Eigen::MatrixXf> curr_derivative_y = calculate_derivative_y(curr_intensity_eig);


        m_curr_rgbd_pyramid.emplace_back(std::make_shared<goodguy::rgbd_image>(
                                             goodguy::rgbd_image(curr_intensity_eig, curr_depth_eig, curr_derivative_x, curr_derivative_y, m_param.camera_params)));


        for(std::size_t i = 1; i < m_param.iter_count.size(); ++i) {
            goodguy::camera_parameter half_param = m_curr_rgbd_pyramid[i-1]->get_param();
            half_param.fx /= 2.0;
            half_param.fy /= 2.0;
            half_param.cx /= 2.0;
            half_param.cy /= 2.0;

            std::shared_ptr<Eigen::MatrixXf> half_intensity = get_half_image(m_curr_rgbd_pyramid[i-1]->get_intensity());
            std::shared_ptr<Eigen::MatrixXf> half_depth = get_half_image(m_curr_rgbd_pyramid[i-1]->get_depth());
            std::shared_ptr<Eigen::MatrixXf> half_derivative_x = calculate_derivative_x(half_intensity);
            std::shared_ptr<Eigen::MatrixXf> half_derivative_y = calculate_derivative_y(half_intensity);
            m_curr_rgbd_pyramid.emplace_back(std::make_shared<goodguy::rgbd_image>(
                                                 goodguy::rgbd_image(half_intensity, half_depth, half_derivative_x, half_derivative_y, half_param)));

        }



        // Compute background model image
        std::pair<std::shared_ptr<Eigen::MatrixXf>, std::shared_ptr<Eigen::MatrixXf>> bgm_set = compute_bgm(m_hist_intensity, m_hist_depth, m_hist_poses, m_curr_rgbd_pyramid[m_param.bgm_level]->get_param());

        std::vector<std::shared_ptr<Eigen::MatrixXf>> bgm, labeled_bgm;


        if(std::get<0>(bgm_set) == NULL) {
            const std::shared_ptr<Eigen::MatrixXf>& curr_depth = m_curr_rgbd_pyramid[0]->get_depth();
            bgm.push_back(std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(Eigen::MatrixXf::Ones(curr_depth->rows(), curr_depth->cols()))));
            labeled_bgm.push_back(std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(Eigen::MatrixXf::Ones(curr_depth->rows(), curr_depth->cols()))));
        }
        else {
            std::shared_ptr<Eigen::MatrixXf> bgm_level_0 = std::get<0>(bgm_set);
            std::shared_ptr<Eigen::MatrixXf> labeled_bgm_level_0 = std::get<1>(bgm_set);
            for(std::size_t j = 0; j < m_param.bgm_level; ++j) {
                bgm_level_0 = get_double_image(bgm_level_0);
                labeled_bgm_level_0 = get_double_image(labeled_bgm_level_0);
            }
            bgm.push_back(bgm_level_0);
            labeled_bgm.push_back(labeled_bgm_level_0);
        }


        for(std::size_t i = 1; i < m_param.iter_count.size(); ++i) {
            bgm.emplace_back(get_half_image(bgm[i-1]));
            labeled_bgm.emplace_back(get_half_image(labeled_bgm[i-1]));
        }


        // Show BGM
        if(bgm.size() > 0) {
            cv::Mat bgm_cv = eigen2cv(*bgm[0]);
            cv::Mat labeled_bgm_cv = eigen2cv(*labeled_bgm[0]);
            cv::imshow(std::string("BGM ")+boost::lexical_cast<std::string>(0), bgm_cv/20.0);
            cv::imshow(std::string("LABELED BGM ")+boost::lexical_cast<std::string>(0), labeled_bgm_cv);
            m_bgm = bgm[0];
        }
        else{
            m_bgm = NULL;
        }


        if(m_prev_rgbd_pyramid.size() != 0) {
            // Compute Odometry
            odometry = compute_odometry(m_prev_rgbd_pyramid, m_curr_rgbd_pyramid, bgm, labeled_bgm, m_param.iter_count);

            Eigen::Matrix4f log_curr_pose = odometry.log();
            Eigen::VectorXf vel_curr_pose(6);
            vel_curr_pose(0) = -log_curr_pose(0,1);
            vel_curr_pose(1) = log_curr_pose(0,2);
            vel_curr_pose(2) = -log_curr_pose(1,2);
            vel_curr_pose(3) = log_curr_pose(0,3);
            vel_curr_pose(4) = log_curr_pose(1,3);
            vel_curr_pose(5) = log_curr_pose(2,3);
            if(vel_curr_pose.norm() > 0.4) {
                std::cout << "============================= Odometry calculation is failed!! =========" << std::endl;
                odometry = Eigen::Matrix4f::Identity();
                m_hist_poses.clear();
                m_hist_depth.clear();
                m_hist_intensity.clear();
                m_velocity = Eigen::Matrix4f::Identity();
            }
            else {
                // Current pose between current image and previous in the previous view point(t->t-1)
                *(m_hist_poses.back()) = odometry;
                m_velocity = odometry;
            }




            m_global_pose =  odometry * m_global_pose;

        }

        return odometry;

    }

    Eigen::Matrix4f get_current_pose() const {
        return m_global_pose;
    }

    void set_param(const bamvo_parameter& param) {
        m_param = param;
    }
    bamvo_parameter& get_param() {
        return m_param;
    }

private:


    Eigen::Matrix4f compute_odometry(
        const std::vector<std::shared_ptr<goodguy::rgbd_image>>& prev_rgbd_pyramid,
        const std::vector<std::shared_ptr<goodguy::rgbd_image>>& curr_rgbd_pyramid,
        const std::vector<std::shared_ptr<Eigen::MatrixXf>>& bgm_pyramid,
        const std::vector<std::shared_ptr<Eigen::MatrixXf>>& labeled_bgm_pyramid,
        const std::vector<int>& iter_count)

    {
        Eigen::Matrix4f odometry = Eigen::Matrix4f::Identity(); 
        odometry = m_velocity;

        if(prev_rgbd_pyramid.size() != curr_rgbd_pyramid.size()
                && prev_rgbd_pyramid.size() != bgm_pyramid.size()
                && bgm_pyramid.size() != labeled_bgm_pyramid.size()
                && bgm_pyramid.size() != iter_count.size())
        {
            std::cerr << "Not equal level size" << std::endl;
            return odometry;
        }

        int max_level = prev_rgbd_pyramid.size()-1;
        float sigma = std::numeric_limits<float>::max();
        float mu = 0.0;

        for(int level = max_level; level >= 0; --level) {
            const goodguy::camera_parameter& param = curr_rgbd_pyramid[level]->get_param();

            std::shared_ptr<Eigen::MatrixXf> residuals;
            std::shared_ptr<Eigen::MatrixXf> corresps;
            std::shared_ptr<Eigen::MatrixXf> A;

            const std::shared_ptr<goodguy::rgbd_image>& prev = prev_rgbd_pyramid[level];
            const std::shared_ptr<goodguy::rgbd_image>& curr = curr_rgbd_pyramid[level];
            const std::shared_ptr<Eigen::MatrixXf>& bgm = bgm_pyramid[level];

            std::shared_ptr<Eigen::MatrixXf> weight;


            for(int iter = 0; iter < iter_count[level]; ++iter) {

                compute_residuals_with_A(prev, curr, odometry, param, residuals, corresps, A);
                compute_weight(prev, residuals, corresps, bgm, mu, sigma, weight);


                Eigen::VectorXf ksi = compute_ksi(weight, residuals, corresps, A);

                Eigen::Matrix4f twist;
                twist <<
                      0.,      -ksi(2), ksi(1),  ksi(3),
                      ksi(2),  0.,      -ksi(0), ksi(4),
                      -ksi(1), ksi(0),  0,       ksi(5),
                      0.,      0.,      0.,      0.;
                Eigen::Matrix4f odom_slice = twist.exp();

                odometry = odom_slice * odometry;
            }
        }

        return odometry;
    }

    void compute_weight(
        const std::shared_ptr<goodguy::rgbd_image>& prev,
        const std::shared_ptr<Eigen::MatrixXf>& residuals,
        const std::shared_ptr<Eigen::MatrixXf>& corresps,
        const std::shared_ptr<Eigen::MatrixXf>& bgm,
        float& mu,
        float& sigma,
        std::shared_ptr<Eigen::MatrixXf>& weight)
    {
        const int rows = prev->get_intensity()->rows();
        const int cols = prev->get_intensity()->cols();

        if(weight == NULL) {
            weight = std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(rows, cols));
        }

        const std::shared_ptr<Eigen::MatrixXf>& prev_depth = prev->get_depth();

        __m128* prev_depth_sse = (__m128*)prev_depth->data();
        __m128* residuals_sse = (__m128*)residuals->data();
        __m128* corresps_sse = (__m128*)corresps->data();
        __m128* bgm_sse = (__m128*)bgm->data();
        __m128* weight_sse = (__m128*)weight->data();

        __m128 ones = _mm_set_ps1(1);
        __m128 zeros = _mm_set_ps1(0);

        float corresps_num = 0;
        float sigma_temp = 0;
        float mu_temp = 0;
        float nu = 5.0;

        __m128 sigma_sse = _mm_set_ps1(sigma);
        __m128 mu_sse = _mm_set_ps1(mu);
        __m128 sigma_rcp_sse = _mm_rcp_ps(sigma_sse);
        __m128 sigma_rcp_sqaure_sse = _mm_mul_ps(sigma_rcp_sse,sigma_rcp_sse);
        __m128 nu_sse = _mm_set_ps1(nu);
        __m128 nu_plus_one_sse = _mm_set_ps1(nu+1);

        __m128 depth_min = _mm_set_ps1(m_param.range_odo.min);
        __m128 depth_max = _mm_set_ps1(m_param.range_odo.max);

        auto lambda_for_weight = [&](const tbb::blocked_range<int>& r) {
            for(int x0 = r.begin(); x0 < r.end(); ++x0) {
                for(int y0 = 0; y0 < rows/4; ++y0) {
                    std::size_t index = x0*prev_depth->rows()/4+y0;
                    __m128 d0 = prev_depth_sse[index];
                    __m128 corresps_val = corresps_sse[index];
                    __m128 residuals_ori_val = residuals_sse[index];
                    __m128 bgm_val = bgm_sse[index];
                    __m128 residuals_val = _mm_sub_ps(residuals_ori_val, mu_sse);

                    __m128 available = _mm_and_ps(_mm_cmpeq_ps(ones, corresps_val), _mm_cmpneq_ps(bgm_val, zeros));
                    available = _mm_and_ps(available, _mm_and_ps(_mm_cmpge_ps(d0, depth_min), _mm_cmpge_ps(depth_max, d0)));
                    __m128 available_one = _mm_and_ps(available, ones);

                    __m128 residuals_sqaure = _mm_mul_ps(residuals_val, residuals_val);

                    // Compute weight for sensor model
                    __m128 w1 = _mm_add_ps(nu_sse, _mm_mul_ps(residuals_sqaure, sigma_rcp_sqaure_sse));
                    __m128 w2 = _mm_mul_ps(nu_plus_one_sse, _mm_rcp_ps(w1));
                    __m128 w3 = _mm_and_ps(available, w2);


                    // Compute weight for background and sensor models
                    weight_sse[index] = _mm_and_ps(_mm_mul_ps(bgm_val, w3), available);   // Consider both sensor and background models
                    //weight_sse[index] = _mm_and_ps(ones, available);                    // Not consider any model
                    //weight_sse[index] = _mm_and_ps(w3, available);                      // Consider only sensor model
                    //weight_sse[index] = _mm_and_ps(bgm_val, available);                 // Consider only background model

                    // Estimate sigma and mu for sensor model
                    __m128 v0 = _mm_mul_ps(residuals_sqaure, nu_plus_one_sse);
                    __m128 v1 = _mm_add_ps(nu_sse, _mm_mul_ps(residuals_sqaure, sigma_rcp_sqaure_sse));
                    __m128 v2 = _mm_mul_ps(v0, _mm_rcp_ps(v1));
                    __m128 v3 = _mm_and_ps(available, v2);
                    residuals_ori_val = _mm_and_ps(available, residuals_ori_val);
                    for(int k = 0; k < 4; ++k) {
                        mu_temp += ((float*)&residuals_ori_val)[k];
                        sigma_temp += ((float*)&v3)[k];
                        corresps_num += ((float*)&available_one)[k];
                    }

                }
            }
        };

        lambda_for_weight(tbb::blocked_range<int>(0,cols));
        // Update sigma
        mu = mu_temp/corresps_num;
        sigma = std::sqrt(sigma_temp/corresps_num);
    }




    Eigen::VectorXf compute_ksi(
        const std::shared_ptr<Eigen::MatrixXf>& weight,
        const std::shared_ptr<Eigen::MatrixXf>& residuals,
        const std::shared_ptr<Eigen::MatrixXf>& corresps,
        const std::shared_ptr<Eigen::MatrixXf>& A)
    {
        Eigen::VectorXf ksi = Eigen::VectorXf::Zero(6,1);

        int corresps_num = 0;
        for(int j = 0; j < corresps->cols(); ++j) {
            for(int i = 0; i < corresps->rows(); ++i) {
                if((*weight)(i,j) != 0.0) {
                    corresps_num++;
                }
            }
        }

        Eigen::MatrixXf b_solve(corresps_num, 1);
        Eigen::MatrixXf A_solve(corresps_num, 6);
        Eigen::MatrixXf W_solve(corresps_num, 1);

        int count = 0;
        for(int j = 0; j < corresps->cols(); ++j) {
            for(int i = 0; i < corresps->rows(); ++i) {
                if((*weight)(i,j) != 0.0) {
                    b_solve(count,0) = (*residuals)(i,j);
                    W_solve(count,0) = (*weight)(i,j);
                    A_solve.row(count) = A->row(j*corresps->rows()+i);
                    count++;
                }
            }
        }

        Eigen::MatrixXf AtWA = A_solve.transpose() *W_solve.asDiagonal()* A_solve;
        Eigen::MatrixXf AtWb = A_solve.transpose() *W_solve.asDiagonal()* b_solve;

        ksi = AtWA.ldlt().solve(AtWb);

        return ksi;
    }

    inline __m128 _mm_floor_ps2(const __m128& x) {
        __m128i v0 = _mm_setzero_si128();
        __m128i v1 = _mm_cmpeq_epi32(v0,v0);
        __m128i ji = _mm_srli_epi32( v1, 25);
        __m128 j = (__m128)_mm_slli_epi32( ji, 23);
        __m128i i = _mm_cvttps_epi32(x);
        __m128 fi = _mm_cvtepi32_ps(i);
        __m128 igx = _mm_cmpgt_ps(fi, x);
        j = _mm_and_ps(igx, j);
        return _mm_sub_ps(fi, j);
    }

    void compute_residuals_with_A(
        const std::shared_ptr<goodguy::rgbd_image>& prev,
        const std::shared_ptr<goodguy::rgbd_image>& curr,
        const Eigen::Matrix4f& transform,
        const goodguy::camera_parameter& cparam,
        std::shared_ptr<Eigen::MatrixXf>& residuals,
        std::shared_ptr<Eigen::MatrixXf>& corresps,
        std::shared_ptr<Eigen::MatrixXf>& A)
    {
        const int rows = prev->get_intensity()->rows();
        const int cols = prev->get_intensity()->cols();

        if(residuals == NULL) {
            residuals = std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(rows, cols));
        }
        if(corresps == NULL) {
            corresps = std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(rows, cols));
        }
        if(A == NULL) {
            A = std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(rows*cols, 6));
        }

        const float& fx = cparam.fx;
        const float& fy = cparam.fy;
        const float& cx = cparam.cx;
        const float& cy = cparam.cy;

        const std::shared_ptr<Eigen::MatrixXf>& prev_intensity = prev->get_intensity();
        const std::shared_ptr<Eigen::MatrixXf>& curr_intensity = curr->get_intensity();

        const std::shared_ptr<Eigen::MatrixXf>& prev_x_derivative = prev->get_x_derivative();
        const std::shared_ptr<Eigen::MatrixXf>& prev_y_derivative = prev->get_y_derivative();

        const std::shared_ptr<Eigen::MatrixXf>& prev_depth = prev->get_depth();
        const std::shared_ptr<Eigen::MatrixXf>& curr_depth = curr->get_depth();

        Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
        K(0,0) = fx;
        K(1,1) = fy;
        K(0,2) = cx;
        K(1,2) = cy;

        Eigen::Matrix3f R = transform.block<3,3>(0,0);
        Eigen::Matrix3f KRK_inv = K*R*K.inverse();
        Eigen::Vector3f Kt = K*transform.block<3,1>(0,3);

        __m128 fx_sse = _mm_set_ps1(fx);
        __m128 fy_sse = _mm_set_ps1(fy);
        __m128 fx_rcp_sse = _mm_rcp_ps(fx_sse);
        __m128 fy_rcp_sse = _mm_rcp_ps(fy_sse);
        __m128 cx_sse = _mm_set_ps1(cx);
        __m128 cy_sse = _mm_set_ps1(cy);

        __m128* prev_depth_sse = (__m128*)prev_depth->data();
        __m128* prev_intensity_sse = (__m128*)prev_intensity->data();
        __m128* residuals_sse = (__m128*)residuals->data();
        __m128* corresps_sse = (__m128*)corresps->data();
        __m128* A_sse = (__m128*)A->data();
        __m128* prev_x_derivative_sse = (__m128*)prev_x_derivative->data();
        __m128* prev_y_derivative_sse = (__m128*)prev_y_derivative->data();

        __m128 KRK_inv_0_0 = _mm_set_ps1(KRK_inv(0,0));
        __m128 KRK_inv_0_1 = _mm_set_ps1(KRK_inv(0,1));
        __m128 KRK_inv_0_2 = _mm_set_ps1(KRK_inv(0,2));
        __m128 KRK_inv_1_0 = _mm_set_ps1(KRK_inv(1,0));
        __m128 KRK_inv_1_1 = _mm_set_ps1(KRK_inv(1,1));
        __m128 KRK_inv_1_2 = _mm_set_ps1(KRK_inv(1,2));
        __m128 KRK_inv_2_0 = _mm_set_ps1(KRK_inv(2,0));
        __m128 KRK_inv_2_1 = _mm_set_ps1(KRK_inv(2,1));
        __m128 KRK_inv_2_2 = _mm_set_ps1(KRK_inv(2,2));
        __m128 Kt_0 = _mm_set_ps1(Kt(0));
        __m128 Kt_1 = _mm_set_ps1(Kt(1));
        __m128 Kt_2 = _mm_set_ps1(Kt(2));

        __m128 inc = _mm_set_ps(3,2,1,0);

        __m128 x_min = _mm_set_ps1(0);
        __m128 x_max = _mm_set_ps1(prev_depth->cols()-2);
        __m128 y_min = _mm_set_ps1(0);
        __m128 y_max = _mm_set_ps1(prev_depth->rows()-2);

        __m128 ones = _mm_set_ps1(1);
        __m128 zeros = _mm_set_ps1(0);
        __m128 minus_ones = _mm_set_ps1(-1);

        __m128 depth_min = _mm_set_ps1(m_param.range_odo.min);
        __m128 depth_max = _mm_set_ps1(m_param.range_odo.max);

        __m128 max_depth_diff = _mm_set_ps1(m_param.depth_diff_max);


        auto lambda_for_residuals = [&](const tbb::blocked_range<int>& r) {
            for(int x0 = r.begin(); x0 < r.end(); ++x0) {
                __m128 x0_sse = _mm_set_ps1(x0);

                for(int y0 = 0; y0 < rows/4; ++y0) {

                    __m128 y0_sse = _mm_set_ps1(y0*4);
                    y0_sse = _mm_add_ps(y0_sse, inc);

                    __m128 d0 = prev_depth_sse[x0*prev_depth->rows()/4+y0];

                    __m128 d0_available = _mm_and_ps(_mm_cmpge_ps(d0, depth_min), _mm_cmpge_ps(depth_max, d0));


                    __m128 d1_warp = _mm_add_ps(_mm_mul_ps(KRK_inv_2_0, x0_sse),  _mm_mul_ps(KRK_inv_2_1, y0_sse));
                    d1_warp = _mm_add_ps(_mm_mul_ps(_mm_add_ps(d1_warp, KRK_inv_2_2), d0), Kt_2);

                    __m128 d1_warp_rcp = _mm_rcp_ps(d1_warp);

                    __m128 x1_warp = _mm_add_ps(_mm_mul_ps(KRK_inv_0_0, x0_sse),  _mm_mul_ps(KRK_inv_0_1, y0_sse));
                    x1_warp = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(x1_warp, KRK_inv_0_2), d0), Kt_0), d1_warp_rcp);

                    __m128 y1_warp = _mm_add_ps(_mm_mul_ps(KRK_inv_1_0, x0_sse),  _mm_mul_ps(KRK_inv_1_1, y0_sse));
                    y1_warp = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(y1_warp, KRK_inv_1_2), d0), Kt_1), d1_warp_rcp);

                    __m128 x1_warp_int = _mm_floor_ps2(x1_warp);
                    __m128 y1_warp_int = _mm_floor_ps2(y1_warp);

                    d0_available = _mm_and_ps(d0_available, _mm_cmpgt_ps(x1_warp_int, x_min));
                    d0_available = _mm_and_ps(d0_available, _mm_cmpgt_ps(x_max, x1_warp_int));
                    d0_available = _mm_and_ps(d0_available, _mm_cmpgt_ps(y1_warp_int, y_min));
                    d0_available = _mm_and_ps(d0_available, _mm_cmpgt_ps(y_max, y1_warp_int));
                    d0_available = _mm_and_ps(_mm_cmpge_ps(d1_warp, depth_min), _mm_cmpge_ps(depth_max, d1_warp));

                    x1_warp_int = _mm_min_ps(x1_warp_int, x_max);
                    x1_warp_int = _mm_max_ps(x1_warp_int, x_min);

                    y1_warp_int = _mm_min_ps(y1_warp_int, y_max);
                    y1_warp_int = _mm_max_ps(y1_warp_int, y_min);

                    __m128 x1w = _mm_sub_ps(x1_warp, x1_warp_int);
                    __m128 x0w = _mm_sub_ps(ones, x1w);
                    __m128 y1w = _mm_sub_ps(y1_warp, y1_warp_int);
                    __m128 y0w = _mm_sub_ps(ones, y1w);


                    __m128 d1 = _mm_set_ps((*curr_depth)(((float*)&y1_warp_int)[3]+0, ((float*)&x1_warp_int)[3]+0),
                                           (*curr_depth)(((float*)&y1_warp_int)[2]+0, ((float*)&x1_warp_int)[2]+0),
                                           (*curr_depth)(((float*)&y1_warp_int)[1]+0, ((float*)&x1_warp_int)[1]+0),
                                           (*curr_depth)(((float*)&y1_warp_int)[0]+0, ((float*)&x1_warp_int)[0]+0));

                    __m128 d1_available = _mm_and_ps(_mm_cmpge_ps(d1, depth_min), _mm_cmpge_ps(depth_max, d1));
                    d0_available = _mm_and_ps(d0_available, d1_available);

                    __m128 depth_diff = _mm_sub_ps(d1_warp, d1);
                    depth_diff = _mm_or_ps(_mm_and_ps(_mm_cmpge_ps(depth_diff, zeros), depth_diff), _mm_and_ps(_mm_cmplt_ps(depth_diff, zeros), _mm_mul_ps(depth_diff, minus_ones)));
                    __m128 depth_diff_available = _mm_cmpge_ps(max_depth_diff, depth_diff);
                    d0_available = _mm_and_ps(depth_diff_available, d0_available);
                    d0_available = _mm_and_ps(d0_available, _mm_cmpord_ps(d0, d1));

                    __m128 x0y0 = _mm_set_ps((*curr_intensity)(((float*)&y1_warp_int)[3]+0, ((float*)&x1_warp_int)[3]+0),
                                             (*curr_intensity)(((float*)&y1_warp_int)[2]+0, ((float*)&x1_warp_int)[2]+0),
                                             (*curr_intensity)(((float*)&y1_warp_int)[1]+0, ((float*)&x1_warp_int)[1]+0),
                                             (*curr_intensity)(((float*)&y1_warp_int)[0]+0, ((float*)&x1_warp_int)[0]+0));
                    __m128 x0y1 = _mm_set_ps((*curr_intensity)(((float*)&y1_warp_int)[3]+1, ((float*)&x1_warp_int)[3]+0),
                                             (*curr_intensity)(((float*)&y1_warp_int)[2]+1, ((float*)&x1_warp_int)[2]+0),
                                             (*curr_intensity)(((float*)&y1_warp_int)[1]+1, ((float*)&x1_warp_int)[1]+0),
                                             (*curr_intensity)(((float*)&y1_warp_int)[0]+1, ((float*)&x1_warp_int)[0]+0));
                    __m128 x1y0 = _mm_set_ps((*curr_intensity)(((float*)&y1_warp_int)[3]+0, ((float*)&x1_warp_int)[3]+1),
                                             (*curr_intensity)(((float*)&y1_warp_int)[2]+0, ((float*)&x1_warp_int)[2]+1),
                                             (*curr_intensity)(((float*)&y1_warp_int)[1]+0, ((float*)&x1_warp_int)[1]+1),
                                             (*curr_intensity)(((float*)&y1_warp_int)[0]+0, ((float*)&x1_warp_int)[0]+1));
                    __m128 x1y1 = _mm_set_ps((*curr_intensity)(((float*)&y1_warp_int)[3]+1, ((float*)&x1_warp_int)[3]+1),
                                             (*curr_intensity)(((float*)&y1_warp_int)[2]+1, ((float*)&x1_warp_int)[2]+1),
                                             (*curr_intensity)(((float*)&y1_warp_int)[1]+1, ((float*)&x1_warp_int)[1]+1),
                                             (*curr_intensity)(((float*)&y1_warp_int)[0]+1, ((float*)&x1_warp_int)[0]+1));

                    __m128 prev_warped_intensity_val_0 = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(x0y0, x0w), _mm_mul_ps(x1y0, x1w)), y0w);
                    __m128 prev_warped_intensity_val_1 = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(x0y1, x0w), _mm_mul_ps(x1y1, x1w)), y1w);
                    __m128 prev_warped_intensity_val = _mm_add_ps(prev_warped_intensity_val_0, prev_warped_intensity_val_1);

                    __m128 prev_intensity_val = prev_intensity_sse[x0*prev_depth->rows()/4+y0];
                    __m128 residuals_val = _mm_mul_ps(_mm_sub_ps(prev_intensity_val, prev_warped_intensity_val), ones);

                    __m128 dIdx = prev_x_derivative_sse[x0*prev_depth->rows()/4+y0];
                    __m128 dIdy = prev_y_derivative_sse[x0*prev_depth->rows()/4+y0];

                    residuals_sse[x0*prev_depth->rows()/4+y0] = _mm_and_ps(d0_available, residuals_val);
                    corresps_sse[x0*prev_depth->rows()/4+y0] = _mm_and_ps(d0_available, ones);

                    __m128 Z1 = d1;
                    __m128 Z1_rcp = _mm_rcp_ps(Z1);

                    __m128 X1 = _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(x1_warp_int, cx_sse), fx_rcp_sse), Z1);
                    __m128 Y1 = _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(y1_warp_int, cy_sse), fy_rcp_sse), Z1);

                    __m128 v0 = _mm_mul_ps(_mm_mul_ps(dIdx, fx_sse), Z1_rcp);
                    __m128 v1 = _mm_mul_ps(_mm_mul_ps(dIdy, fy_sse), Z1_rcp);
                    __m128 v2 = _mm_mul_ps(minus_ones, _mm_mul_ps(Z1_rcp, _mm_add_ps(_mm_mul_ps(v0, X1), _mm_mul_ps(v1, Y1))));

                    __m128 A0 = _mm_sub_ps(_mm_mul_ps(Y1, v2), _mm_mul_ps(Z1, v1));
                    __m128 A1 = _mm_sub_ps(_mm_mul_ps(Z1, v0), _mm_mul_ps(X1, v2));
                    __m128 A2 = _mm_sub_ps(_mm_mul_ps(X1, v1), _mm_mul_ps(Y1, v0));
                    __m128 A3 = v0;
                    __m128 A4 = v1;
                    __m128 A5 = v2;

                    A_sse[x0*rows/4+y0 + rows/4*cols*0] = _mm_and_ps(d0_available, A0);
                    A_sse[x0*rows/4+y0 + rows/4*cols*1] = _mm_and_ps(d0_available, A1);
                    A_sse[x0*rows/4+y0 + rows/4*cols*2] = _mm_and_ps(d0_available, A2);
                    A_sse[x0*rows/4+y0 + rows/4*cols*3] = _mm_and_ps(d0_available, A3);
                    A_sse[x0*rows/4+y0 + rows/4*cols*4] = _mm_and_ps(d0_available, A4);
                    A_sse[x0*rows/4+y0 + rows/4*cols*5] = _mm_and_ps(d0_available, A5);

                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<int>(0,cols), lambda_for_residuals);
    }


    std::shared_ptr<Eigen::MatrixXf> get_double_image(const std::shared_ptr<Eigen::MatrixXf>& image) {

        std::shared_ptr<Eigen::MatrixXf> double_image(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(image->rows()*2, image->cols()*2)));

        auto lambda_for_double = [&](const tbb::blocked_range<int>& r) {
            for(int j = r.begin(); j < r.end(); j += 1) {
                for(int i = 0; i < image->rows(); i += 1) {
                    int i_plus = std::min(i+1, (int)image->rows()-1);
                    int j_plus = std::min(j+1, (int)image->cols()-1);
                    (*double_image)(2*i+0,2*j+0) = (*image)(i+0,j+0);
                    (*double_image)(2*i+0,2*j+1) = (*image)(i+0,j+0)*0.5 + (*image)(i+0,j_plus)*0.5;
                    (*double_image)(2*i+1,2*j+0) = (*image)(i+0,j+0)*0.5 + (*image)(i_plus,j+0)*0.5;
                    (*double_image)(2*i+1,2*j+1) = (*image)(i+0,j+0)*0.5 + (*image)(i_plus,j_plus)*0.5;
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<int>(0,image->cols()), lambda_for_double);

        return double_image;
    }

    std::shared_ptr<Eigen::MatrixXf> get_half_image(const std::shared_ptr<Eigen::MatrixXf>& image) {

        std::shared_ptr<Eigen::MatrixXf> half_image(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(image->rows()/2, image->cols()/2)));

        auto lambda_for_half = [&](const tbb::blocked_range<int>& r) {
            for(int j = r.begin(); j < r.end(); j += 2) {
                for(int i = 0; i < image->rows(); i += 2) {
                    (*half_image)(i/2,j/2) = (*image)(i,j);
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<int>(0,image->cols()), lambda_for_half);

        return half_image;
    }

    std::shared_ptr<Eigen::MatrixXf> calculate_derivative_x(const std::shared_ptr<Eigen::MatrixXf>& intensity) {
        std::shared_ptr<Eigen::MatrixXf> derivative_x(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(intensity->rows(), intensity->cols())));

        auto lambda_for_derivative = [&](const tbb::blocked_range<int>& r) {
            for(int j = r.begin(); j < r.end(); ++j) {
                for(int i = 0; i < intensity->rows()/4; ++i) {

                    int left = std::max(j-1, 0);
                    int right = std::min(j+1, (int)intensity->cols()-1);

                    int up = std::max(i*4-1, 0);
                    int down = std::min(i*4+4, (int)intensity->rows()-1);

                    __m128 ul, ur;
                    if((i*4-1) < 0) {
                        ul = _mm_set_ps((*intensity)(up+2, left), (*intensity)(up+2, left), (*intensity)(up+1, left), (*intensity)(up+0, left));
                        ur = _mm_set_ps((*intensity)(up+2, right), (*intensity)(up+2, right), (*intensity)(up+1, right), (*intensity)(up+0, right));
                    }
                    else {
                        ul = _mm_set_ps((*intensity)(up+3, left), (*intensity)(up+2, left), (*intensity)(up+1, left), (*intensity)(up+0, left));
                        ur = _mm_set_ps((*intensity)(up+3, right), (*intensity)(up+2, right), (*intensity)(up+1, right), (*intensity)(up+0, right));
                    }


                    __m128 cl = _mm_load_ps(intensity->data()+left*intensity->rows()+i*4);
                    __m128 cr = _mm_load_ps(intensity->data()+right*intensity->rows()+i*4);


                    __m128 dl, dr;
                    if(i*4+4 > ((int)intensity->rows()-1)) {
                        dl = _mm_set_ps((*intensity)(down-0, left), (*intensity)(down-0, left), (*intensity)(down-1, left), (*intensity)(down-2, left));
                        dr = _mm_set_ps((*intensity)(down-0, right), (*intensity)(down-0, right), (*intensity)(down-1, right), (*intensity)(down-2, right));
                    }
                    else {
                        dl = _mm_set_ps((*intensity)(down-0, left), (*intensity)(down-1, left), (*intensity)(down-2, left), (*intensity)(down-3, left));
                        dr = _mm_set_ps((*intensity)(down-0, right), (*intensity)(down-1, right), (*intensity)(down-2, right), (*intensity)(down-3, right));
                    }

                    __m128 val =  _mm_add_ps(_mm_add_ps(_mm_add_ps(ur, dr), cr), cr);
                    __m128 val2 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_sub_ps(val, ul), dl), cl), cl);
                    __m128 val3 = _mm_mul_ps(val2, _mm_set1_ps(0.125));

                    _mm_store_ps(derivative_x->data()+j*(int)intensity->rows()+4*i, val3);
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<int>(0,intensity->cols()), lambda_for_derivative);

        return derivative_x;
    }

    std::shared_ptr<Eigen::MatrixXf> calculate_derivative_y(const std::shared_ptr<Eigen::MatrixXf>& intensity) {
        std::shared_ptr<Eigen::MatrixXf> derivative_y(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(intensity->rows(), intensity->cols())));

        auto lambda_for_derivative = [&](const tbb::blocked_range<int>& r) {
            for(int j = r.begin(); j < r.end(); ++j) {
                for(int i = 0; i < intensity->rows()/4; ++i) {

                    int left = std::max(j-1, 0);
                    int right = std::min(j+1, (int)intensity->cols()-1);

                    int up = std::max(i*4-1, 0);
                    int down = std::min(i*4+4, (int)intensity->rows()-1);

                    __m128 ul, uc, ur;
                    if((i*4-1) < 0) {
                        ul = _mm_set_ps((*intensity)(up+2, left), (*intensity)(up+1, left), (*intensity)(up+0, left), (*intensity)(up+0, left));
                        uc = _mm_set_ps((*intensity)(up+2, j), (*intensity)(up+1, j), (*intensity)(up+0, j), (*intensity)(up+0, j));
                        ur = _mm_set_ps((*intensity)(up+2, right), (*intensity)(up+1, right), (*intensity)(up+0, right), (*intensity)(up+0, right));
                    }
                    else {
                        ul = _mm_set_ps((*intensity)(up+3, left), (*intensity)(up+2, left), (*intensity)(up+1, left), (*intensity)(up+0, left));
                        uc = _mm_set_ps((*intensity)(up+3, j), (*intensity)(up+2, j), (*intensity)(up+1, j), (*intensity)(up+0, j));
                        ur = _mm_set_ps((*intensity)(up+3, right), (*intensity)(up+2, right), (*intensity)(up+1, right), (*intensity)(up+0, right));
                    }

                    __m128 dl, dc, dr;
                    if(i*4+4 > ((int)intensity->rows()-1)) {
                        dl = _mm_set_ps((*intensity)(down-0, left), (*intensity)(down-0, left), (*intensity)(down-1, left), (*intensity)(down-2, left));
                        dc = _mm_set_ps((*intensity)(down-0, j), (*intensity)(down-0, j), (*intensity)(down-1, j), (*intensity)(down-2, j));
                        dr = _mm_set_ps((*intensity)(down-0, right), (*intensity)(down-0, right), (*intensity)(down-1, right), (*intensity)(down-2, right));
                    }
                    else {
                        dl = _mm_set_ps((*intensity)(down-0, left), (*intensity)(down-1, left), (*intensity)(down-2, left), (*intensity)(down-3, left));
                        dc = _mm_set_ps((*intensity)(down-0, j), (*intensity)(down-1, j), (*intensity)(down-2, j), (*intensity)(down-3, j));
                        dr = _mm_set_ps((*intensity)(down-0, right), (*intensity)(down-1, right), (*intensity)(down-2, right), (*intensity)(down-3, right));
                    }

                    __m128 val =  _mm_add_ps(_mm_add_ps(_mm_add_ps(dr, dl), dc), dc);
                    __m128 val2 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_sub_ps(val, ul), ur), uc), uc);
                    __m128 val3 = _mm_mul_ps(val2, _mm_set1_ps(0.125));

                    _mm_store_ps(derivative_y->data()+j*(int)intensity->rows()+4*i, val3);

                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<int>(0,intensity->cols()), lambda_for_derivative);

        return derivative_y;
    }



    bool support_image_size(const int rows, const int cols) {
        if(rows == 480 && cols == 640) {
            return true;
        }
        else if(rows == 240 && cols == 320) {
            return true;
        }
        else if(rows == 120 && cols == 160) {
            return true;
        }
        else {
            return false;
        }
    }

    std::pair<std::shared_ptr<Eigen::MatrixXf>,std::shared_ptr<Eigen::MatrixXf>> warp_depth(const Eigen::MatrixXf& depth, const Eigen::MatrixXf& intensity, const Eigen::Matrix4f& pose, const goodguy::camera_parameter& cparam) {

        const int rows = depth.rows();
        const int cols = depth.cols();

        std::shared_ptr<Eigen::MatrixXf> warped_depth(new Eigen::MatrixXf(rows, cols));
        std::shared_ptr<Eigen::MatrixXf> warped_intensity(new Eigen::MatrixXf(rows, cols));

        const float& fx = cparam.fx;
        const float& fy = cparam.fy;
        const float& cx = cparam.cx;
        const float& cy = cparam.cy;

        Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
        K(0,0) = fx;
        K(1,1) = fy;
        K(0,2) = cx;
        K(1,2) = cy;

        Eigen::Matrix3f R = pose.block<3,3>(0,0);
        Eigen::Matrix3f KRK_inv = K*R*K.inverse();
        Eigen::Vector3f Kt = K*pose.block<3,1>(0,3);

        __m128* depth_sse = (__m128*)depth.data();
        __m128* intensity_sse = (__m128*)intensity.data();

        __m128 KRK_inv_0_0 = _mm_set_ps1(KRK_inv(0,0));
        __m128 KRK_inv_0_1 = _mm_set_ps1(KRK_inv(0,1));
        __m128 KRK_inv_0_2 = _mm_set_ps1(KRK_inv(0,2));
        __m128 KRK_inv_1_0 = _mm_set_ps1(KRK_inv(1,0));
        __m128 KRK_inv_1_1 = _mm_set_ps1(KRK_inv(1,1));
        __m128 KRK_inv_1_2 = _mm_set_ps1(KRK_inv(1,2));
        __m128 KRK_inv_2_0 = _mm_set_ps1(KRK_inv(2,0));
        __m128 KRK_inv_2_1 = _mm_set_ps1(KRK_inv(2,1));
        __m128 KRK_inv_2_2 = _mm_set_ps1(KRK_inv(2,2));
        __m128 Kt_0 = _mm_set_ps1(Kt(0));
        __m128 Kt_1 = _mm_set_ps1(Kt(1));
        __m128 Kt_2 = _mm_set_ps1(Kt(2));

        __m128 inc = _mm_set_ps(3,2,1,0);

        __m128 x_min = _mm_set_ps1(0);
        __m128 x_max = _mm_set_ps1(cols-1);
        __m128 y_min = _mm_set_ps1(0);
        __m128 y_max = _mm_set_ps1(rows-1);

        __m128 ones = _mm_set_ps1(1);

        __m128 depth_min = _mm_set_ps1(m_param.range_odo.min);
        __m128 depth_max = _mm_set_ps1(m_param.range_odo.max);


        auto lambda_for_warp = [&](const tbb::blocked_range<int>& r) {
            for(int x0 = r.begin(); x0 < r.end(); ++x0) {
                __m128 x0_sse = _mm_set_ps1(x0);

                for(int y0 = 0; y0 < rows/4; ++y0) {

                    __m128 y0_sse = _mm_set_ps1(y0*4);
                    y0_sse = _mm_add_ps(y0_sse, inc);

                    __m128 d0 = depth_sse[x0*rows/4+y0];
                    __m128 i0 = intensity_sse[x0*rows/4+y0];

                    __m128 d0_available = _mm_and_ps(_mm_cmpge_ps(d0, depth_min), _mm_cmpge_ps(depth_max, d0));


                    __m128 d1_warp = _mm_add_ps(_mm_mul_ps(KRK_inv_2_0, x0_sse),  _mm_mul_ps(KRK_inv_2_1, y0_sse));
                    d1_warp = _mm_add_ps(_mm_mul_ps(_mm_add_ps(d1_warp, KRK_inv_2_2), d0), Kt_2);

                    __m128 d1_warp_rcp = _mm_rcp_ps(d1_warp);

                    __m128 x1_warp = _mm_add_ps(_mm_mul_ps(KRK_inv_0_0, x0_sse),  _mm_mul_ps(KRK_inv_0_1, y0_sse));
                    x1_warp = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(x1_warp, KRK_inv_0_2), d0), Kt_0), d1_warp_rcp);

                    __m128 y1_warp = _mm_add_ps(_mm_mul_ps(KRK_inv_1_0, x0_sse),  _mm_mul_ps(KRK_inv_1_1, y0_sse));
                    y1_warp = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(y1_warp, KRK_inv_1_2), d0), Kt_1), d1_warp_rcp);

                    __m128 x1_warp_int = _mm_floor_ps2(x1_warp);
                    __m128 y1_warp_int = _mm_floor_ps2(y1_warp);

                    d0_available = _mm_and_ps(d0_available, _mm_cmpgt_ps(x1_warp_int, x_min));
                    d0_available = _mm_and_ps(d0_available, _mm_cmpgt_ps(x_max, x1_warp_int));
                    d0_available = _mm_and_ps(d0_available, _mm_cmpgt_ps(y1_warp_int, y_min));
                    d0_available = _mm_and_ps(d0_available, _mm_cmpgt_ps(y_max, y1_warp_int));
                    d0_available = _mm_and_ps(d0_available, ones);

                    x1_warp_int = _mm_min_ps(x1_warp_int, x_max);
                    x1_warp_int = _mm_max_ps(x1_warp_int, x_min);

                    y1_warp_int = _mm_min_ps(y1_warp_int, y_max);
                    y1_warp_int = _mm_max_ps(y1_warp_int, y_min);

                    for(int i = 0; i < 4; ++i) {
                        if(((float*)&d0_available)[i]) {
                            (*warped_depth)(((float*)&y1_warp_int)[i], ((float*)&x1_warp_int)[i]) = ((float*)&d1_warp)[i];
                            (*warped_intensity)(((float*)&y1_warp_int)[i], ((float*)&x1_warp_int)[i]) = ((float*)&i0)[i];
                        }
                    }
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<int>(0,cols), lambda_for_warp);


        return std::make_pair(warped_depth, warped_intensity);
    }

    std::pair<std::shared_ptr<Eigen::MatrixXf>, std::shared_ptr<Eigen::MatrixXf>> compute_bgm(
                const std::deque<std::shared_ptr<Eigen::MatrixXf>>& intensity,
                const std::deque<std::shared_ptr<Eigen::MatrixXf>>& depth,
                const std::deque<std::shared_ptr<Eigen::Matrix4f>>& poses,
                const goodguy::camera_parameter& param)
    {

        if(depth.size() == 0) {
            return std::make_pair(std::shared_ptr<Eigen::MatrixXf>(), std::shared_ptr<Eigen::MatrixXf>());
        }
        int cols = depth.front()->cols();;
        int rows = depth.front()->rows();;
        if(!support_image_size(rows, cols)) {
            std::cerr << "Unsupported Image size" << std::endl;
            return std::make_pair(std::shared_ptr<Eigen::MatrixXf>(), std::shared_ptr<Eigen::MatrixXf>());
        }

        Eigen::Matrix4f warp_pose = Eigen::Matrix4f::Identity();
        std::vector<Eigen::Matrix4f> poses_accumulate(depth.size(), Eigen::Matrix4f::Identity());
        auto it_for_poses = poses.rbegin();
        auto it_for_poses_acc = poses_accumulate.rbegin();
        for(std::size_t i = 0; i < depth.size(); ++i) {
            warp_pose = warp_pose*(**it_for_poses++);
            *it_for_poses_acc++ = warp_pose;
        }

        std::vector<std::shared_ptr<Eigen::MatrixXf>> depth_differences(depth.size()-1);
        std::vector<std::shared_ptr<Eigen::MatrixXf>> intensity_differences(depth.size()-1);

        auto last_hist_images = warp_depth(*depth.back(), *intensity.back(), poses_accumulate.back(), param);

        const std::shared_ptr<Eigen::MatrixXf> last_hist_depth = last_hist_images.first;
        const std::shared_ptr<Eigen::MatrixXf> last_hist_intensity = last_hist_images.second;
        auto lambda_for_calculate_depth_difference = [&](const tbb::blocked_range<std::size_t>& r) {
            for(std::size_t k = r.begin(); k < r.end(); ++k) {
                std::pair<std::shared_ptr<Eigen::MatrixXf>, std::shared_ptr<Eigen::MatrixXf>> warped_images = warp_depth(*depth[k], *intensity[k], poses_accumulate[k], param);
                std::shared_ptr<Eigen::MatrixXf> warped_depth = warped_images.first;
                std::shared_ptr<Eigen::MatrixXf> warped_intensity = warped_images.second;
                depth_differences[k] = std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(rows, cols));
                intensity_differences[k] = std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(rows, cols));
                *(depth_differences[k]) = (*last_hist_depth - *warped_depth).cwiseAbs();
                *(intensity_differences[k]) = (*last_hist_intensity - *warped_intensity).cwiseAbs();
            }
        };
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0,depth.size()-1), lambda_for_calculate_depth_difference);


        auto sigmas = calculate_sigma(depth_differences, intensity_differences);
        std::shared_ptr<Eigen::MatrixXf> sigma_depth = sigmas.first; 
        std::shared_ptr<Eigen::MatrixXf> sigma_intensity = sigmas.second; 
        std::shared_ptr<Eigen::MatrixXf> bgm = calculate_bgm_impl(depth_differences, intensity_differences, last_hist_depth, sigma_depth, sigma_intensity);

        if(bgm != NULL) {
        cv::Mat bgm_cv = eigen2cv(*bgm);
        cv::Mat bgm_filtered_cv;
        cv::bilateralFilter(bgm_cv, bgm_filtered_cv, 5, 0.02, 0.02);
        cv::imshow("Filter", bgm_filtered_cv/50.0);
        (*bgm) = cv2eigen(bgm_filtered_cv);
        }


        std::shared_ptr<Eigen::MatrixXf> labeled_bgm = calculate_labeled_bgm_impl_sse(bgm);


        return std::make_pair(bgm, labeled_bgm);
    }


    std::shared_ptr<Eigen::MatrixXf> calculate_labeled_bgm_impl_sse(const std::shared_ptr<Eigen::MatrixXf>& bgm) {
        if(bgm == NULL) {
            return std::shared_ptr<Eigen::MatrixXf>();
        }

        const int& rows = bgm->rows();
        const int& cols = bgm->cols();

        const float v_d = 1.96 * 16.23 / 1000.0;
        const float epsilon_B = 0.6744 * std::exp(-0.4548)/(std::sqrt(M_PI)*v_d);

        std::shared_ptr<Eigen::MatrixXf> labeled_bgm(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));

        __m128 epsilon_B_sse = _mm_set1_ps(epsilon_B);
        __m128 one_sse = _mm_set1_ps(1);
        const __m128* bgm_sse = (__m128*)bgm->data();
        __m128* labeled_bgm_sse = (__m128*)labeled_bgm->data();

        auto lambda_for_labeled_bgm = [&](const tbb::blocked_range<int>& r) {
            for(int j = r.begin(); j < r.end(); ++j) {
                for(int i = 0; i < rows/4; ++i) {
                    std::size_t index = j*(rows/4) + i;
                    __m128 bgm_val = bgm_sse[index];
                    labeled_bgm_sse[index] = _mm_min_ps(_mm_cmpgt_ps(bgm_val, epsilon_B_sse), one_sse);
                }
            }
        };

        tbb::parallel_for(tbb::blocked_range<int>(0,cols), lambda_for_labeled_bgm);
        return labeled_bgm;
    }


    std::shared_ptr<Eigen::MatrixXf> calculate_labeled_bgm_impl(const std::shared_ptr<Eigen::MatrixXf>& bgm) {
        if(bgm == NULL) {
            return std::shared_ptr<Eigen::MatrixXf>();
        }

        const int& rows = bgm->rows();
        const int& cols = bgm->cols();

        const float v_d = 1.96 * 16.23 / 1000.0;
        const float epsilon_B = 0.6744 * std::exp(-0.4548)/(std::sqrt(M_PI)*v_d);

        std::shared_ptr<Eigen::MatrixXf> labeled_bgm(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));

        auto lambda_for_labeled_bgm = [&](const tbb::blocked_range<int>& r) {
            for(int i = r.begin(); i < r.end(); ++i) {
                for(int j = 0; j < cols; ++j) {
                    float bgm_val = (*bgm)(i,j);
                    if(bgm_val < epsilon_B) {
                        (*labeled_bgm)(i,j) = 0.0;
                    }
                    else {
                        (*labeled_bgm)(i,j) = 1.0;
                    }
                }
            }
        };

        tbb::parallel_for(tbb::blocked_range<int>(0,rows), lambda_for_labeled_bgm);

        return labeled_bgm;
    }

    std::shared_ptr<Eigen::MatrixXf> calculate_bgm_impl(
            const std::vector<std::shared_ptr<Eigen::MatrixXf>>& depth_differences,
            const std::vector<std::shared_ptr<Eigen::MatrixXf>>& intensity_differences,
            const std::shared_ptr<Eigen::MatrixXf>& last_depth,
            const std::shared_ptr<Eigen::MatrixXf>& sigma_depth, 
            const std::shared_ptr<Eigen::MatrixXf>& sigma_intensity) 
    {
        if(depth_differences.size() == 0) {
            return std::shared_ptr<Eigen::MatrixXf>();
        }

        const int& rows = depth_differences[0]->rows();
        const int& cols = depth_differences[0]->cols();

        std::shared_ptr<Eigen::MatrixXf> bgm(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));


        float pre = 1/std::sqrt(2*M_PI);

        auto lambda_for_bgm = [&](const tbb::blocked_range<int>& r) {
            for(int i = r.begin(); i < r.end(); ++i) {
                for(int j = 0; j < cols; ++j) {
                    float sigma_depth_val_inv = 1/(*sigma_depth)(i,j);
                    float sigma_intensity_val_inv = 1/(*sigma_intensity)(i,j);
                    float bgm_depth_val = 0;
                    float bgm_intensity_val = 0;

                    for(std::size_t k = 0; k < depth_differences.size(); ++k) {
                        float exp_in = (*depth_differences[k])(i,j)*sigma_depth_val_inv;
                        bgm_depth_val += std::exp(-0.5*exp_in*exp_in);
                    }
                    for(std::size_t k = 0; k < intensity_differences.size(); ++k) {
                        float exp_in = (*intensity_differences[k])(i,j)*sigma_intensity_val_inv;
                        bgm_intensity_val += std::exp(-0.5*exp_in*exp_in);
                    }

                    float last_depth_val = (*last_depth)(i,j);
                    //bgm_depth_val = bgm_depth_val*pre*sigma_depth_val_inv*(1.0/(float)depth_differences.size());
                    bgm_depth_val = bgm_depth_val*pre*sigma_depth_val_inv*(1.0/(float)depth_differences.size())*last_depth_val;
                    bgm_intensity_val = bgm_intensity_val*pre*sigma_intensity_val_inv*(1.0/(float)intensity_differences.size());

                    if(last_depth_val < m_param.range_bgm.max && last_depth_val > m_param.range_bgm.min) {
                        if(bgm_intensity_val > 10.0 ) bgm_intensity_val = 10.0;
                        //(*bgm)(i,j) = bgm_depth_val*bgm_intensity_val;
                        (*bgm)(i,j) = bgm_depth_val;
                    }
                    else {
                        (*bgm)(i,j) = 0.0;
                    }
                }
            }
        };

        tbb::parallel_for(tbb::blocked_range<int>(0,rows), lambda_for_bgm);

        return bgm;
    }


    std::pair<std::shared_ptr<Eigen::MatrixXf>, std::shared_ptr<Eigen::MatrixXf>> calculate_sigma(
            const std::vector<std::shared_ptr<Eigen::MatrixXf>>& depth_differences,
            const std::vector<std::shared_ptr<Eigen::MatrixXf>>& intensity_differences) 
    {
        if(depth_differences.size() == 0) {
            return std::make_pair(std::shared_ptr<Eigen::MatrixXf>(), std::shared_ptr<Eigen::MatrixXf>());
        }

        const int& rows = depth_differences[0]->rows();
        const int& cols = depth_differences[0]->cols();

        std::shared_ptr<Eigen::MatrixXf> sigma(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));
        std::shared_ptr<Eigen::MatrixXf> sigma_intensity(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));

        auto lambda_for_sigma = [&](const tbb::blocked_range<int>& r) {
            for(int i = r.begin(); i < r.end(); ++i) {
                for(int j = 0; j < cols; ++j) {
                    std::vector<float> diff;
                    for(std::size_t k = 0; k < depth_differences.size(); ++k) {
                        diff.emplace_back((*depth_differences[k])(i,j));
                    }
                    std::vector<float> diff_intensity;
                    for(std::size_t k = 0; k < intensity_differences.size(); ++k) {
                        diff_intensity.emplace_back((*intensity_differences[k])(i,j));
                    }
                    std::sort(diff.begin(), diff.end());
                    std::sort(diff_intensity.begin(), diff_intensity.end());
                    float median = diff[diff.size()/2];
                    float median_intensity = diff_intensity[diff_intensity.size()/2];
                    float sigma_val = median/0.6744/std::sqrt(2);
                    float sigma_val_intensity = median_intensity/0.6744/std::sqrt(2);

                    if(sigma_val_intensity < std::numeric_limits<float>::epsilon()) {
                        (*sigma_intensity)(i,j) = std::numeric_limits<float>::epsilon();
                    }
                    else {
                        (*sigma_intensity)(i,j) = median_intensity/0.6744/std::sqrt(2);
                    }
                    if(sigma_val < std::numeric_limits<float>::epsilon()) {
                        (*sigma)(i,j) = std::numeric_limits<float>::epsilon();
                    }
                    else {
                        (*sigma)(i,j) = median/0.6744/std::sqrt(2);
                    }
                }
            }
        };

        tbb::parallel_for(tbb::blocked_range<int>(0,rows), lambda_for_sigma);


        return std::make_pair(sigma, sigma_intensity);
    }

public:


    static Eigen::MatrixXf cv2eigen(const cv::Mat& depth) {
        Eigen::MatrixXf depth_eigen;
        cv::cv2eigen(depth, depth_eigen);
        return depth_eigen;
    }

    static cv::Mat eigen2cv(const Eigen::MatrixXf& depth) {
        cv::Mat depth_cv;
        cv::eigen2cv(depth,depth_cv);
        return depth_cv;
    }

    std::shared_ptr<Eigen::MatrixXf> get_bgm(){
        return m_bgm;
    }



private:
    std::vector<std::shared_ptr<goodguy::rgbd_image>> m_curr_rgbd_pyramid;
    std::vector<std::shared_ptr<goodguy::rgbd_image>> m_prev_rgbd_pyramid;

    std::shared_ptr<Eigen::MatrixXf> m_bgm;

    std::deque<std::shared_ptr<Eigen::MatrixXf>> m_hist_intensity;
    std::deque<std::shared_ptr<Eigen::MatrixXf>> m_hist_depth;
    std::deque<std::shared_ptr<Eigen::Matrix4f>> m_hist_poses;

    bamvo_parameter m_param;

    Eigen::Matrix4f m_global_pose;
    Eigen::Matrix4f m_velocity;


};
}

#endif
