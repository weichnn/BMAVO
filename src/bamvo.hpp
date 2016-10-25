#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Eigen>
#include <xmmintrin.h>

#include <vector>
#include <list>
#include <deque>
#include <memory>
#include <cmath>
#include <algorithm>
//#include <math.h>
#include <cfenv>


#include <tbb/tbb.h>
//#include <tbb/task_scheduler_init.h>
//#include <tbb/parallel_for.h>


#include "computation_time.hpp"
#include "parameter.hpp"
#include "rgbd_image.hpp"

namespace goodguy{
    class bamvo{
        public:
            bamvo() : m_time("BAMVO") { }

            void add(const cv::Mat& color, const cv::Mat& depth){
                m_time.global_tic();
                if(depth.type() != CV_32FC1 || color.type() != CV_8UC3){ 
                    std::cerr << "Not supported data type!" << std::endl; 
                    return; 
                }

                if(m_curr_rgbd_pyramid.size() != 0){
                    m_hist_poses.emplace_back(std::make_shared<Eigen::Matrix4f>(Eigen::Matrix4f(Eigen::Matrix4f::Identity())));
                    m_hist_depth.emplace_back(m_curr_rgbd_pyramid[m_param.bgm_level]->get_depth());
                    m_hist_point_cloud.emplace_back(m_curr_rgbd_pyramid[m_param.bgm_level]->get_point_cloud());

                    if(m_hist_depth.size() > m_param.hist_size){
                        m_hist_depth.pop_front();
                        m_hist_poses.pop_front();
                        m_hist_point_cloud.pop_front();
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
                std::shared_ptr<Eigen::MatrixXf> curr_point_cloud_eig = generate_point_cloud_sse(curr_depth_eig, m_param.camera_params);
                std::shared_ptr<Eigen::MatrixXf> curr_derivative_x = calculate_derivative_x(curr_intensity_eig);
                std::shared_ptr<Eigen::MatrixXf> curr_derivative_y = calculate_derivative_y(curr_intensity_eig);

                m_time.tic();

                //cv::Mat curr_derivative_x_cv = eigen2cv(*curr_derivative_x);
                //cv::Mat curr_derivative_y_cv = eigen2cv(*curr_derivative_y);
                //cv::imshow("Derivative_x", curr_derivative_x_cv);
                //cv::imshow("Derivative_y", curr_derivative_y_cv);

                m_curr_rgbd_pyramid.emplace_back(std::make_shared<goodguy::rgbd_image>(
                            goodguy::rgbd_image(curr_intensity_eig, curr_depth_eig, curr_point_cloud_eig, curr_derivative_x, curr_derivative_y, m_param.camera_params)));



                
                for(std::size_t i = 1; i < m_param.iter_count.size(); ++i){
                    goodguy::camera_parameter half_param = m_curr_rgbd_pyramid[i-1]->get_param();
                    half_param.fx /= 2.0;
                    half_param.fy /= 2.0;
                    half_param.cx /= 2.0;
                    half_param.cy /= 2.0;

                    std::shared_ptr<Eigen::MatrixXf> half_intensity = get_half_image(m_curr_rgbd_pyramid[i-1]->get_intensity());
                    std::shared_ptr<Eigen::MatrixXf> half_depth = get_half_image(m_curr_rgbd_pyramid[i-1]->get_depth());
                    std::shared_ptr<Eigen::MatrixXf> half_point_cloud = generate_point_cloud_sse(half_depth, half_param);
                    std::shared_ptr<Eigen::MatrixXf> half_derivative_x = get_half_image(m_curr_rgbd_pyramid[i-1]->get_x_derivative());
                    std::shared_ptr<Eigen::MatrixXf> half_derivative_y = get_half_image(m_curr_rgbd_pyramid[i-1]->get_y_derivative());
                    m_curr_rgbd_pyramid.emplace_back(std::make_shared<goodguy::rgbd_image>(
                                goodguy::rgbd_image(half_intensity, half_depth, half_point_cloud, half_derivative_x, half_derivative_y, half_param)));

                }
                m_time.toc();
                m_time.tic();



                // Compute background model image
                std::pair<std::shared_ptr<Eigen::MatrixXf>, std::shared_ptr<Eigen::MatrixXf>> bgm_set = compute_bgm(m_hist_point_cloud, m_hist_poses, m_curr_rgbd_pyramid[m_param.bgm_level]->get_param());

                m_time.toc();
                m_time.tic();

                std::vector<std::shared_ptr<Eigen::MatrixXf>> bgm, labeled_bgm;


                if(std::get<0>(bgm_set) == NULL){
                    const std::shared_ptr<Eigen::MatrixXf>& curr_depth = m_curr_rgbd_pyramid[0]->get_depth();
                    bgm.push_back(std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(Eigen::MatrixXf::Ones(curr_depth->rows(), curr_depth->cols()))));
                    labeled_bgm.push_back(std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(Eigen::MatrixXf::Ones(curr_depth->rows(), curr_depth->cols()))));
                }
                else{
                    bgm.emplace_back(std::get<0>(bgm_set));
                    labeled_bgm.emplace_back(std::get<1>(bgm_set));
                }


                for(std::size_t i = 1; i < m_param.iter_count.size(); ++i){
                    bgm.emplace_back(get_half_image(bgm[i-1]));
                    labeled_bgm.emplace_back(get_half_image(labeled_bgm[i-1]));
                }


                // Show pyramid for RGB-D
                if(false)
                    for(std::size_t i = 0; i < m_param.iter_count.size(); ++i){
                        cv::Mat rgb = eigen2cv(*m_curr_rgbd_pyramid[i]->get_intensity());
                        cv::Mat depth = eigen2cv(*m_curr_rgbd_pyramid[i]->get_depth());
                        cv::Mat deri_x = eigen2cv(*m_curr_rgbd_pyramid[i]->get_x_derivative());
                        cv::Mat deri_y = eigen2cv(*m_curr_rgbd_pyramid[i]->get_y_derivative());
                        cv::Mat bgm_cv = eigen2cv(*bgm[i]);
                        cv::imshow(std::string("Intensity ")+boost::lexical_cast<std::string>(i), rgb);
                        cv::imshow(std::string("Depth ")+boost::lexical_cast<std::string>(i), depth);
                        cv::imshow(std::string("Derivative X ")+boost::lexical_cast<std::string>(i), deri_x);
                        cv::imshow(std::string("Derivative Y ")+boost::lexical_cast<std::string>(i), deri_y);
                        cv::imshow(std::string("BGM ")+boost::lexical_cast<std::string>(i), bgm_cv/20.0);
                    }


                m_time.toc();

                m_time.tic();


                if(m_prev_rgbd_pyramid.size() != 0){
                    // Compute Odometry
                    Eigen::Matrix4f curr_pose = compute_odometry(m_prev_rgbd_pyramid, m_curr_rgbd_pyramid, bgm, labeled_bgm, m_param.iter_count);

                    // Current pose between current image and previous in the previous view point(t->t-1)
                    //*(m_hist_poses.back()) = curr_pose;

                }

                m_time.toc();

                m_time.global_toc();
            }

            void set_param(const bamvo_parameter& param){ m_param = param; }
            bamvo_parameter& get_param(){ return m_param; }

        private:

            Eigen::Matrix4f compute_odometry(
                    const std::vector<std::shared_ptr<goodguy::rgbd_image>>& prev_rgbd_pyramid, 
                    const std::vector<std::shared_ptr<goodguy::rgbd_image>>& curr_rgbd_pyramid, 
                    const std::vector<std::shared_ptr<Eigen::MatrixXf>>& bgm, 
                    const std::vector<std::shared_ptr<Eigen::MatrixXf>>& labeled_bgm, 
                    const std::vector<int> iter_count)

            {
                ComputationTime odom_time("odometry");
                odom_time.global_tic();
                Eigen::Matrix4f odometry = Eigen::Matrix4f::Identity();

                if(prev_rgbd_pyramid.size() != curr_rgbd_pyramid.size() 
                        && prev_rgbd_pyramid.size() != bgm.size() 
                        && bgm.size() != labeled_bgm.size()
                        && bgm.size() != iter_count.size())
                {
                    std::cerr << "Not equal level size" << std::endl;
                    return odometry;
                }

                int max_level = prev_rgbd_pyramid.size()-1;


                for(int level = max_level; level >= 0; --level){
                    const goodguy::camera_parameter& param = curr_rgbd_pyramid[level]->get_param();

                    const int rows = curr_rgbd_pyramid[level]->get_intensity()->rows();
                    const int cols = curr_rgbd_pyramid[level]->get_intensity()->cols();

                    std::shared_ptr<Eigen::MatrixXf> residuals;
                    std::shared_ptr<Eigen::MatrixXf> corresps;

                    const std::shared_ptr<goodguy::rgbd_image>& prev = prev_rgbd_pyramid[level];
                    const std::shared_ptr<goodguy::rgbd_image>& curr = curr_rgbd_pyramid[level];
                    
                    for(int iter = 0; iter < iter_count[level]; ++iter){

                        residuals = compute_residuals(prev, curr, odometry, param, residuals, corresps);


                        std::cout <<"LV: " << level << " " << iter <<std::endl;


                        if(residuals != NULL){
                            //cv::Mat residuals_cv = eigen2cv(*residuals);
                            //cv::imshow(std::string("Residuals ")+boost::lexical_cast<std::string>(level), residuals_cv*10);
                        }
                    }

                }


                odom_time.global_toc();
                return odometry;
            }

            std::shared_ptr<Eigen::MatrixXf> compute_residuals(
                    const std::shared_ptr<goodguy::rgbd_image>& prev, 
                    const std::shared_ptr<goodguy::rgbd_image>& curr, 
                    const Eigen::Matrix4f& transform,
                    const goodguy::camera_parameter& cparam, 
                    std::shared_ptr<Eigen::MatrixXf>& residuals,
                    std::shared_ptr<Eigen::MatrixXf>& corresps)
            {
                ComputationTime time("residuals");

                time.tic();


                const int rows = prev->get_intensity()->rows();
                const int cols = prev->get_intensity()->cols();

                if(residuals == NULL){
                    residuals = std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(rows, cols));
                }
                if(corresps == NULL){
                    corresps = std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(rows, cols));
                }

                corresps->setZero();

                const float& fx = cparam.fx;
                const float& fy = cparam.fy;
                const float& cx = cparam.cx;
                const float& cy = cparam.cy;

                Eigen::MatrixXf transformed_curr_point_cloud = transform_point_cloud_sse(*prev->get_point_cloud(), transform);
                time.toc();
                time.tic();

                const std::shared_ptr<Eigen::MatrixXf>& prev_intensity = prev->get_intensity();
                const std::shared_ptr<Eigen::MatrixXf>& curr_intensity = curr->get_intensity();

                const std::shared_ptr<Eigen::MatrixXf>& prev_depth = prev->get_depth();
                const std::shared_ptr<Eigen::MatrixXf>& curr_depth = curr->get_depth();

                Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
                K(0,0) = fx; K(1,1) = fy; K(0,2) = cx; K(1,2) = cy;

                Eigen::Matrix3f R = transform.block<3,3>(0,0);
                Eigen::Matrix3f KRK_inv = K*R*K.inverse();
                Eigen::Vector3f Kt = K*transform.block<3,1>(0,3);

                for(int x0 = 0; x0 < cols; ++x0){
                    for(int y0 = 0; y0 < rows; ++y0){
                        float d0 = (*prev_depth)(y0, x0);

                        float d1_warp = d0 *(KRK_inv(2,0)*x0+KRK_inv(2,1)*y0+KRK_inv(2,2)) + Kt(2);
                        float x1_warp = (d0 *(KRK_inv(0,0)*x0+KRK_inv(0,1)*y0+KRK_inv(0,2)) + Kt(0))/d1_warp;
                        float y1_warp = (d0 *(KRK_inv(1,0)*x0+KRK_inv(1,1)*y0+KRK_inv(1,2)) + Kt(1))/d1_warp;

                        int x1_warp_int = std::floor(x1_warp);
                        int y1_warp_int = std::floor(y1_warp);

                        if(x1_warp_int >= 0 && x1_warp_int < cols-1 && y1_warp_int >= 0 && y1_warp_int < rows-1){
                            (*corresps)(y0, x0) = 1.0;
                            float x1w = x1_warp - x1_warp_int;
                            float x0w = 1.0 - x1w;
                            float y1w = y1_warp - y1_warp_int;
                            float y0w = 1.0 - y1w;

                            float x0y0 = (*curr_intensity)(y1_warp+0, x1_warp+0);
                            float x0y1 = (*curr_intensity)(y1_warp+1, x1_warp+0);
                            float x1y0 = (*curr_intensity)(y1_warp+0, x1_warp+1);
                            float x1y1 = (*curr_intensity)(y1_warp+1, x1_warp+1);


                            float prev_intensity_val = (*prev_intensity)(y0, x0);
                            float prev_warped_intensity_val = (x0y0 * x0w + x1y0 * x1w) * y0w + (x0y1 * x0w + x1y1 * x1w) * y1w;
                            //float prev_warped_intensity = x0y0;

                            (*residuals)(y0, x0) = -prev_warped_intensity_val + prev_intensity_val;
                        }


                    }
                }

                /*

                for(int i = 0; i < transformed_curr_point_cloud.cols(); ++i){
                    float d0_w = transformed_curr_point_cloud(2,i);
                    float x0_w = fx*transformed_curr_point_cloud(0,i)*(1/d0_w) + cx;
                    float y0_w = fy*transformed_curr_point_cloud(1,i)*(1/d0_w) + cy;

                    int x0_wi = std::floor(x0_w);
                    int y0_wi = std::floor(y0_w);

                    int x1_i = i / rows;
                    int y1_i = i % rows;

                    if(x0_wi >= 0 && x0_wi < cols-1 && y0_wi >= 0 && y0_wi < rows-1){
                        (*corresps)(y0_wi, x0_wi) = 1.0;
                        float x1w = x0_w - x0_wi;
                        float x0w = 1.0 - x1w;
                        float y1w = y0_w - y0_wi;
                        float y0w = 1.0 - y1w;

                        float x0y0 = (*prev_intensity)(y0_wi+0, x0_wi+0);
                        float x0y1 = (*prev_intensity)(y0_wi+1, x0_wi+0);
                        float x1y0 = (*prev_intensity)(y0_wi+0, x0_wi+1);
                        float x1y1 = (*prev_intensity)(y0_wi+1, x0_wi+1);
                        

                        float curr_intensity_val = (*curr_intensity)(y1_i, x1_i);
                        float prev_warped_intensity_val = (x0y0 * x0w + x1y0 * x1w) * y0w + (x0y1 * x0w + x1y1 * x1w) * y1w;
                        //float prev_warped_intensity = x0y0;

                        (*residuals)(y0_wi, x0_wi) = prev_warped_intensity_val - curr_intensity_val;
                    }
                }
                */
                time.toc();

                return residuals;
            }

            std::shared_ptr<Eigen::MatrixXf> get_half_image(const std::shared_ptr<Eigen::MatrixXf>& image){

                std::shared_ptr<Eigen::MatrixXf> half_image(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(image->rows()/2, image->cols()/2)));

                auto lambda_for_half = [&](const tbb::blocked_range<int>& r){
                    for(int j = r.begin(); j < r.end(); j += 2){
                        for(int i = 0; i < image->rows(); i += 2){
                            (*half_image)(i/2,j/2) = (*image)(i,j);
                        }
                    }
                };
                tbb::parallel_for(tbb::blocked_range<int>(0,image->cols()), lambda_for_half);

                return half_image;
            }

            std::shared_ptr<Eigen::MatrixXf> calculate_derivative_x(const std::shared_ptr<Eigen::MatrixXf>& intensity){
                std::shared_ptr<Eigen::MatrixXf> derivative_x(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(intensity->rows(), intensity->cols())));

                auto lambda_for_derivative = [&](const tbb::blocked_range<int>& r){
                    for(int i = r.begin(); i < r.end(); ++i){
                        for(int j = 0; j < intensity->cols(); ++j){
                            int prev = std::max(j-1, 0);
                            int next = std::min(j+1, (int)intensity->cols()-1);
                            (*derivative_x)(i,j) = ((*intensity)(i, next) - (*intensity)(i, prev))*0.5f;
                        }
                    }
                };
                auto lambda_for_derivative2 = [&](const tbb::blocked_range<int>& r){
                    for(int i = r.begin(); i < r.end(); ++i){
                        for(int j = 0; j < intensity->cols(); ++j){
                            int left = std::max(j-1, 0);
                            int right = std::min(j+1, (int)intensity->cols()-1);
                            int up = std::max(i-1, 0);
                            int down = std::min(i+1, (int)intensity->rows()-1);
                            (*derivative_x)(i,j)  = (*intensity)(up, right) + 2.0*(*intensity)(i, right) + (*intensity)(down, right);
                            (*derivative_x)(i,j)  += -(*intensity)(up, left) - 2.0*(*intensity)(i, left) - (*intensity)(down, left);
                            (*derivative_x)(i,j) *= 1/8.0;
                        }
                    }
                };
                //tbb::parallel_for(tbb::blocked_range<int>(0,intensity->rows()), lambda_for_derivative);

                auto lambda_for_derivative3 = [&](const tbb::blocked_range<int>& r){
                    for(int j = r.begin(); j < r.end(); ++j){
                        for(int i = 0; i < intensity->rows()/4; ++i){

                            int left = std::max(j-1, 0);
                            int right = std::min(j+1, (int)intensity->cols()-1);

                            int up = std::max(i*4-1, 0);
                            int down = std::min(i*4+4, (int)intensity->rows()-1);

                            __m128 ul, ur;
                            if((i*4-1) < 0){
                                ul = _mm_set_ps((*intensity)(up+2, left), (*intensity)(up+2, left), (*intensity)(up+1, left), (*intensity)(up+0, left));
                                ur = _mm_set_ps((*intensity)(up+2, right), (*intensity)(up+2, right), (*intensity)(up+1, right), (*intensity)(up+0, right));
                            }
                            else{
                                ul = _mm_set_ps((*intensity)(up+3, left), (*intensity)(up+2, left), (*intensity)(up+1, left), (*intensity)(up+0, left));
                                ur = _mm_set_ps((*intensity)(up+3, right), (*intensity)(up+2, right), (*intensity)(up+1, right), (*intensity)(up+0, right));
                            }


                            __m128 cl = _mm_load_ps(intensity->data()+left*intensity->rows()+i*4);
                            __m128 cr = _mm_load_ps(intensity->data()+right*intensity->rows()+i*4);


                            __m128 dl, dr;
                            if(i*4+4 > ((int)intensity->rows()-1)){
                                dl = _mm_set_ps((*intensity)(down-0, left), (*intensity)(down-0, left), (*intensity)(down-1, left), (*intensity)(down-2, left));
                                dr = _mm_set_ps((*intensity)(down-0, right), (*intensity)(down-0, right), (*intensity)(down-1, right), (*intensity)(down-2, right));
                            }
                            else{
                                dl = _mm_set_ps((*intensity)(down-0, left), (*intensity)(down-1, left), (*intensity)(down-2, left), (*intensity)(down-3, left));
                                dr = _mm_set_ps((*intensity)(down-0, right), (*intensity)(down-1, right), (*intensity)(down-2, right), (*intensity)(down-3, right));
                            }

                            __m128 val =  _mm_add_ps(_mm_add_ps(_mm_add_ps(ur, dr), cr), cr);
                            __m128 val2 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_sub_ps(val, ul), dl), cl), cl);
                            __m128 val3 = _mm_mul_ps(val2, _mm_set1_ps(0.125));
                            //__m128 val = _mm_sub_ps(cr, cl);


                            _mm_store_ps(derivative_x->data()+j*(int)intensity->rows()+4*i, val3);


                            //derivative_x_sse[index] = _mm_add_ps(_mm_add_ps(_mm_add_ps(ur, dr), cr), cr);
                            //derivative_x_sse[index] = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_sub_ps(derivative_x_sse[index], ul), dl), cl), cl);



                            /*
                            (*derivative_x)(i,j)  = (*intensity)(up, right) + 2.0*(*intensity)(i, right) + (*intensity)(down, right);
                            (*derivative_x)(i,j)  += -(*intensity)(up, left) - 2.0*(*intensity)(i, left) - (*intensity)(down, left);
                            (*derivative_x)(i,j) *= 1/8.0;
                            */
                        }
                    }
                };
                tbb::parallel_for(tbb::blocked_range<int>(0,intensity->cols()), lambda_for_derivative3);

                return derivative_x;
            }

            std::shared_ptr<Eigen::MatrixXf> calculate_derivative_y(const std::shared_ptr<Eigen::MatrixXf>& intensity){
                std::shared_ptr<Eigen::MatrixXf> derivative_y(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(intensity->rows(), intensity->cols())));

                auto lambda_for_derivative = [&](const tbb::blocked_range<int>& r){
                    for(int j = r.begin(); j < r.end(); ++j){
                        for(int i = 0; i < intensity->rows(); ++i){
                            int prev = std::max(0, i-1);
                            int next = std::min((int)intensity->rows()-1, i+1);
                            (*derivative_y)(i,j) = ((*intensity)(next, j) - (*intensity)(prev, j))*0.5f;
                        }
                    }
                };

                __m128* derivative_y_sse = (__m128*)derivative_y->data();
                auto lambda_for_derivative3 = [&](const tbb::blocked_range<int>& r){
                    for(int j = r.begin(); j < r.end(); ++j){
                        for(int i = 0; i < intensity->rows()/4; ++i){

                            std::size_t index = j*(intensity->rows()/4)+i;

                            int left = std::max(j-1, 0);
                            int right = std::min(j+1, (int)intensity->cols()-1);

                            int up = std::max(i*4-1, 0);
                            int down = std::min(i*4+4, (int)intensity->rows()-1);

                            __m128 ul, uc, ur;
                            if((i*4-1) < 0){
                                ul = _mm_set_ps((*intensity)(up+2, left), (*intensity)(up+1, left), (*intensity)(up+0, left), (*intensity)(up+0, left));
                                uc = _mm_set_ps((*intensity)(up+2, j), (*intensity)(up+1, j), (*intensity)(up+0, j), (*intensity)(up+0, j));
                                ur = _mm_set_ps((*intensity)(up+2, right), (*intensity)(up+1, right), (*intensity)(up+0, right), (*intensity)(up+0, right));
                            }
                            else{
                                ul = _mm_set_ps((*intensity)(up+3, left), (*intensity)(up+2, left), (*intensity)(up+1, left), (*intensity)(up+0, left));
                                uc = _mm_set_ps((*intensity)(up+3, j), (*intensity)(up+2, j), (*intensity)(up+1, j), (*intensity)(up+0, j));
                                ur = _mm_set_ps((*intensity)(up+3, right), (*intensity)(up+2, right), (*intensity)(up+1, right), (*intensity)(up+0, right));
                            }

                            __m128 dl, dc, dr;
                            if(i*4+4 > ((int)intensity->rows()-1)){
                                dl = _mm_set_ps((*intensity)(down-0, left), (*intensity)(down-0, left), (*intensity)(down-1, left), (*intensity)(down-2, left));
                                dc = _mm_set_ps((*intensity)(down-0, j), (*intensity)(down-0, j), (*intensity)(down-1, j), (*intensity)(down-2, j));
                                dr = _mm_set_ps((*intensity)(down-0, right), (*intensity)(down-0, right), (*intensity)(down-1, right), (*intensity)(down-2, right));
                            }
                            else{
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
                tbb::parallel_for(tbb::blocked_range<int>(0,intensity->cols()), lambda_for_derivative3);
                //lambda_for_derivative3(tbb::blocked_range<int>(0,intensity->cols()));

                return derivative_y;
            }



            bool check_size_for_converting_depth(const std::size_t size, int& depth_rows, int& depth_cols){
                if(size == 640*480){
                    depth_cols = 640;
                    depth_rows = 480;
                }
                else if(size == 320*240){
                    depth_cols = 320;
                    depth_rows = 240;
                }
                else if(size == 160*120){
                    depth_cols = 160;
                    depth_rows = 120;
                }
                else{
                    return false;
                }
                return true;
            }

            inline Eigen::MatrixXf transform_point_cloud(const Eigen::MatrixXf& point_cloud, const Eigen::Matrix4f& pose){
                return pose*point_cloud;
            }

            inline Eigen::MatrixXf transform_point_cloud_sse(const Eigen::MatrixXf& point_cloud, const Eigen::Matrix4f& pose){
                Eigen::Matrix4f pose_t = pose.transpose();
                Eigen::MatrixXf transformed_point_cloud = Eigen::MatrixXf::Zero(4,point_cloud.cols());

                __m128 *transformed_point_cloud_sse = (__m128*)transformed_point_cloud.data();
                __m128 *point_cloud_sse = (__m128*)point_cloud.data();
                __m128 *pose_t_sse = (__m128*)pose_t.data();

                for(int i = 0; i < point_cloud.cols(); ++i){
                    __m128 t0 = _mm_mul_ps(pose_t_sse[0],point_cloud_sse[i]);
                    __m128 t1 = _mm_mul_ps(pose_t_sse[1],point_cloud_sse[i]);
                    __m128 t2 = _mm_mul_ps(pose_t_sse[2],point_cloud_sse[i]);
                    __m128 t3 = _mm_mul_ps(pose_t_sse[3],point_cloud_sse[i]);
                    transformed_point_cloud_sse[i] = _mm_add_ps(_mm_add_ps(_mm_add_ps(t0,t1),t2),t3);
                }
                return transformed_point_cloud;
            }

            std::pair<std::shared_ptr<Eigen::MatrixXf>, std::shared_ptr<Eigen::MatrixXf>> compute_bgm(
                    const std::deque<std::shared_ptr<Eigen::MatrixXf>>& point_clouds,
                    const std::deque<std::shared_ptr<Eigen::Matrix4f>>& poses, 
                    const goodguy::camera_parameter& param)
            {

                if(point_clouds.size() == 0){
                    return std::make_pair(std::shared_ptr<Eigen::MatrixXf>(), std::shared_ptr<Eigen::MatrixXf>());
                }

                std::size_t size = point_clouds.front()->cols();
                int cols = 0;
                int rows = 0;
                if(!check_size_for_converting_depth(size, rows, cols)){
                    std::cerr << "Unsupported Image size" << std::endl;
                    return std::make_pair(std::shared_ptr<Eigen::MatrixXf>(), std::shared_ptr<Eigen::MatrixXf>());
                }

                std::vector<Eigen::MatrixXf> warped_hist_point_clouds(point_clouds.size());

                Eigen::Matrix4f warp_pose = Eigen::Matrix4f::Identity();
                std::vector<Eigen::Matrix4f> poses_accumulate(point_clouds.size(), Eigen::Matrix4f::Identity());
                auto it_for_poses = poses.rbegin();
                auto it_for_poses_acc = poses_accumulate.rbegin();
                for(std::size_t i = 0; i < point_clouds.size(); ++i){
                    warp_pose = (**it_for_poses++)*warp_pose;
                    *it_for_poses_acc++ = warp_pose;
                }

                auto lambda_for_transform = [&](const tbb::blocked_range<std::size_t>& r){
                    for(std::size_t i = r.begin(); i < r.end(); ++i){
                        warped_hist_point_clouds[i] = transform_point_cloud_sse(*point_clouds[i], poses_accumulate[i]);
                    }
                };
                tbb::parallel_for(tbb::blocked_range<std::size_t>(0,point_clouds.size()), lambda_for_transform);


                std::vector<std::shared_ptr<Eigen::MatrixXf>> depth_differences(point_clouds.size()-1);
                const std::shared_ptr<Eigen::MatrixXf> last_hist_depth = generate_depth_from_point_cloud_sse(warped_hist_point_clouds.back(), param);
                auto lambda_for_calculate_depth_difference = [&](const tbb::blocked_range<std::size_t>& r){
                    for(std::size_t k = r.begin(); k < r.end(); ++k){
                        std::shared_ptr<Eigen::MatrixXf> depth = generate_depth_from_point_cloud_sse(warped_hist_point_clouds[k], param);
                        depth_differences[k] = std::make_shared<Eigen::MatrixXf>(Eigen::MatrixXf(depth->rows(), depth->cols()));
                        *(depth_differences[k]) = (*last_hist_depth - *depth).cwiseAbs();
                    }
                };
                tbb::parallel_for(tbb::blocked_range<std::size_t>(0,point_clouds.size()-1), lambda_for_calculate_depth_difference);


                std::shared_ptr<Eigen::MatrixXf> sigma = calculate_sigma(depth_differences);
                std::shared_ptr<Eigen::MatrixXf> bgm = calculate_bgm_impl(depth_differences, last_hist_depth, sigma);
                
                std::shared_ptr<Eigen::MatrixXf> labeled_bgm = calculate_labeled_bgm_impl_sse(bgm);


                return std::make_pair(bgm, labeled_bgm);
            }


            std::shared_ptr<Eigen::MatrixXf> calculate_labeled_bgm_impl_sse(const std::shared_ptr<Eigen::MatrixXf>& bgm){
                if(bgm == NULL){
                    std::cerr << "BGM is empty!" << std::endl;
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

                auto lambda_for_labeled_bgm = [&](const tbb::blocked_range<int>& r){
                    for(int j = r.begin(); j < r.end(); ++j){
                        for(int i = 0; i < rows/4; ++i){
                            std::size_t index = j*(rows/4) + i;
                            __m128 bgm_val = bgm_sse[index];
                            labeled_bgm_sse[index] = _mm_min_ps(_mm_cmpgt_ps(bgm_val, epsilon_B_sse), one_sse);
                        }
                    }
                };

                tbb::parallel_for(tbb::blocked_range<int>(0,cols), lambda_for_labeled_bgm);
                return labeled_bgm;
            }


            std::shared_ptr<Eigen::MatrixXf> calculate_labeled_bgm_impl(const std::shared_ptr<Eigen::MatrixXf>& bgm){
                if(bgm == NULL){
                    std::cerr << "BGM is empty!" << std::endl;
                    return std::shared_ptr<Eigen::MatrixXf>();
                }

                const int& rows = bgm->rows();
                const int& cols = bgm->cols();

                const float v_d = 1.96 * 16.23 / 1000.0;
                const float epsilon_B = 0.6744 * std::exp(-0.4548)/(std::sqrt(M_PI)*v_d);

                std::shared_ptr<Eigen::MatrixXf> labeled_bgm(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));

                auto lambda_for_labeled_bgm = [&](const tbb::blocked_range<int>& r){
                    for(int i = r.begin(); i < r.end(); ++i){
                        for(int j = 0; j < cols; ++j){
                            float bgm_val = (*bgm)(i,j);
                            if(bgm_val < epsilon_B){
                                (*labeled_bgm)(i,j) = 0.0;
                            }
                            else{
                                (*labeled_bgm)(i,j) = 1.0;
                            }
                        }
                    }
                };

                tbb::parallel_for(tbb::blocked_range<int>(0,rows), lambda_for_labeled_bgm);

                return labeled_bgm;
            }

            std::shared_ptr<Eigen::MatrixXf> calculate_bgm_impl(const std::vector<std::shared_ptr<Eigen::MatrixXf>>& depth_differences, const std::shared_ptr<Eigen::MatrixXf>& last_depth, const std::shared_ptr<Eigen::MatrixXf>& sigma){
                if(depth_differences.size() == 0){
                    std::cerr << "Data is empty!" << std::endl;
                    return std::shared_ptr<Eigen::MatrixXf>();
                }

                const int& rows = depth_differences[0]->rows();
                const int& cols = depth_differences[0]->cols();

                std::shared_ptr<Eigen::MatrixXf> bgm(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));


                float pre = 1/std::sqrt(2*M_PI);

                auto lambda_for_bgm = [&](const tbb::blocked_range<int>& r){
                    for(int i = r.begin(); i < r.end(); ++i){
                        for(int j = 0; j < cols; ++j){
                            float sigma_val_inv = 1/(*sigma)(i,j);
                            float bgm_val = 0; 

                            for(std::size_t k = 0; k < depth_differences.size(); ++k){
                                float exp_in = (*depth_differences[k])(i,j)*sigma_val_inv;
                                bgm_val += std::exp(-0.5*exp_in*exp_in);
                            }

                            bgm_val = bgm_val*pre*sigma_val_inv*(1.0/(float)depth_differences.size());

                            float last_depth_val = (*last_depth)(i,j);
                            if(last_depth_val < m_param.range_bgm.max && last_depth_val > m_param.range_bgm.min){
                                (*bgm)(i,j) = bgm_val;
                            }
                            else{
                                (*bgm)(i,j) = 0.0;
                            }
                        }
                    }
                };

                tbb::parallel_for(tbb::blocked_range<int>(0,rows), lambda_for_bgm);

                return bgm;
            }


            std::shared_ptr<Eigen::MatrixXf> calculate_sigma(const std::vector<std::shared_ptr<Eigen::MatrixXf>>& depth_differences){
                if(depth_differences.size() == 0){
                    std::cerr << "Data is empty!" << std::endl;
                    return std::shared_ptr<Eigen::MatrixXf>();
                }

                const int& rows = depth_differences[0]->rows();
                const int& cols = depth_differences[0]->cols();

                std::shared_ptr<Eigen::MatrixXf> sigma(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));

                auto lambda_for_sigma = [&](const tbb::blocked_range<int>& r){
                    for(int i = r.begin(); i < r.end(); ++i){
                        for(int j = 0; j < cols; ++j){
                            std::vector<float> diff;
                            for(std::size_t k = 0; k < depth_differences.size(); ++k){
                                diff.emplace_back((*depth_differences[k])(i,j));
                            }
                            std::sort(diff.begin(), diff.end());
                            float median = diff[diff.size()/2];
                            float sigma_val = median/0.6744/std::sqrt(2);

                            if(sigma_val < m_param.sig_min){
                                (*sigma)(i,j) = m_param.sig_min;
                            }
                            else{
                                (*sigma)(i,j) = median/0.6744/std::sqrt(2);
                            }
                        }
                    }
                };

                tbb::parallel_for(tbb::blocked_range<int>(0,rows), lambda_for_sigma);

                
                return sigma;
            }


            std::shared_ptr<Eigen::MatrixXf> generate_depth_from_point_cloud(const Eigen::MatrixXf& point_cloud, const goodguy::camera_parameter& param){
                std::size_t size = point_cloud.cols();

                int cols = 0;
                int rows = 0;

                if(!check_size_for_converting_depth(size, rows, cols)){
                    std::cerr << "Unsupported Image size" << std::endl;
                    return std::shared_ptr<Eigen::MatrixXf>();
                }

                std::cout << cols << "/" << rows << std::endl;

                const float& fx = param.fx;
                const float& fy = param.fy;
                const float& cx = param.cx;
                const float& cy = param.cy;

                std::shared_ptr<Eigen::MatrixXf> depth(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));

                for(std::size_t i = 0; i < size; ++i){
                    float d = point_cloud(2,i);
                    int x = std::round(fx*point_cloud(0,i)*(1/d) + cx);
                    int y = std::round(fy*point_cloud(1,i)*(1/d) + cy);

                    if(x >= 0 && x < cols && y >= 0 && y < rows){
                        if((*depth)(y,x) < d && d > 0.3 && d < 3.0){
                            (*depth)(y,x) = d;
                        }
                    }
                }
                return depth;
            }

            std::shared_ptr<Eigen::MatrixXf> generate_depth_from_point_cloud_sse(const Eigen::MatrixXf& point_cloud, const goodguy::camera_parameter& param){
                std::size_t size = point_cloud.cols();

                int cols = 0;
                int rows = 0;

                if(!check_size_for_converting_depth(size, rows, cols)){
                    std::cerr << "Unsupported Image size" << std::endl;
                    return std::shared_ptr<Eigen::MatrixXf>();
                }

                const float& fx = param.fx;
                const float& fy = param.fy;
                const float& cx = param.cx;
                const float& cy = param.cy;


                __m128 fx_sse = _mm_set1_ps(fx);
                __m128 fy_sse = _mm_set1_ps(fy);
                __m128 cx_sse = _mm_set1_ps(cx);
                __m128 cy_sse = _mm_set1_ps(cy);

                std::shared_ptr<Eigen::MatrixXf> depth(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(rows,cols)));

                for(std::size_t i = 0; i < size/4; ++i){
                    __m128 d = _mm_set_ps(point_cloud(2,i*4+3),point_cloud(2,i*4+2),point_cloud(2,i*4+1),point_cloud(2,i*4+0));
                    __m128 X = _mm_set_ps(point_cloud(0,i*4+3),point_cloud(0,i*4+2),point_cloud(0,i*4+1),point_cloud(0,i*4+0));
                    __m128 Y = _mm_set_ps(point_cloud(1,i*4+3),point_cloud(1,i*4+2),point_cloud(1,i*4+1),point_cloud(1,i*4+0));
                    __m128 d_inv = _mm_rcp_ps(d);

                    __m128 x = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(fx_sse, X), d_inv), cx_sse);
                    __m128 y = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(fy_sse, Y), d_inv), cy_sse);

                    for(std::size_t k = 0; k < 4; ++k){
                        int x_sub = std::round(((float*)&x)[k]);
                        int y_sub = std::round(((float*)&y)[k]);
                        float d_sub = ((float*)&d)[k];

                        if(x_sub >= 0 && x_sub < cols && y_sub >= 0 && y_sub < rows){
                            if((*depth)(y_sub,x_sub) < d_sub && d_sub > 0.3 && d_sub < 3.0){
                                (*depth)(y_sub,x_sub) = d_sub;
                            }
                        }
                    }

                }
                return depth;
            }

            inline std::shared_ptr<Eigen::MatrixXf> generate_point_cloud(const std::shared_ptr<Eigen::MatrixXf>& depth, const goodguy::camera_parameter& param){
            //[[gnu::target("default")]] Eigen::MatrixXf generate_point_cloud(const Eigen::MatrixXf& depth){
                std::size_t size = depth->cols()*depth->rows();

                std::shared_ptr<Eigen::MatrixXf> point_cloud(new Eigen::MatrixXf(4,size));

                const float& fx = param.fx;
                const float& fy = param.fy;
                const float& cx = param.cx;
                const float& cy = param.cy;

                auto lambda_for_pcl = [&](const tbb::blocked_range<int>& r){
                    for(int j = r.begin(); j < r.end(); ++j){
                        for(int i = 0; i < depth->rows(); ++i){
                            std::size_t index = i*depth->cols() + j;
                            float d = (*depth)(i,j);
                            (*point_cloud)(0,index) = (j-cx)*d*(1/fx);
                            (*point_cloud)(1,index) = (i-cy)*d*(1/fy);
                            (*point_cloud)(2,index) = d;
                            (*point_cloud)(3,index) = 1;
                        }
                    }
                };
                tbb::parallel_for(tbb::blocked_range<int>(0,depth->cols()), lambda_for_pcl);

                return point_cloud;
            }

            inline std::shared_ptr<Eigen::MatrixXf> generate_point_cloud_sse(const std::shared_ptr<Eigen::MatrixXf>& depth, const goodguy::camera_parameter& param){
            //[[gnu::target("sse3")]] Eigen::MatrixXf generate_point_cloud(const Eigen::MatrixXf& depth){
                std::size_t size = depth->cols()*depth->rows();

                std::shared_ptr<Eigen::MatrixXf> point_cloud(new Eigen::MatrixXf(4,size));

                const float& fx = param.fx;
                const float& fy = param.fy;
                const float& cx = param.cx;
                const float& cy = param.cy;
                __m128 *point_cloud_sse = (__m128*)point_cloud->data();


                __m128 f_sse = _mm_set_ps(1, 1, 1/fy, 1/fx);
                __m128 c_sse = _mm_set_ps(0, 0, cy, cx);


                auto lambda_for_pcl = [&](const tbb::blocked_range<int>& r){
                    for(int j = r.begin(); j < r.end(); ++j){
                        for(int i = 0; i < depth->rows(); ++i){
                            float d = (*depth)(i,j);
                            __m128 z = _mm_set1_ps(d);
                            __m128 p = _mm_set_ps(1/d, 1, i, j);
                            __m128 p1 = _mm_sub_ps(p, c_sse);
                            __m128 p2 = _mm_mul_ps(p1, f_sse);
                            __m128 p3 = _mm_mul_ps(p2, z);
                            point_cloud_sse[j*depth->rows()+i] = p3;
                            
                        }
                    }
                };
                tbb::parallel_for(tbb::blocked_range<int>(0,depth->cols()), lambda_for_pcl);

                return point_cloud;
            }




            inline Eigen::MatrixXf cv2eigen(const cv::Mat& depth){
                Eigen::MatrixXf depth_eigen;
                cv::cv2eigen(depth, depth_eigen);
                return depth_eigen;
            }

            inline cv::Mat eigen2cv(const Eigen::MatrixXf& depth){
                cv::Mat depth_cv;
                cv::eigen2cv(depth,depth_cv);
                return depth_cv;
            }



        private:
            std::vector<std::shared_ptr<goodguy::rgbd_image>> m_curr_rgbd_pyramid;
            std::vector<std::shared_ptr<goodguy::rgbd_image>> m_prev_rgbd_pyramid;

            Eigen::MatrixXf m_bgm;

            std::deque<std::shared_ptr<Eigen::MatrixXf>> m_hist_depth;
            std::deque<std::shared_ptr<Eigen::MatrixXf>> m_hist_point_cloud;
            std::deque<std::shared_ptr<Eigen::Matrix4f>> m_hist_poses;

            bamvo_parameter m_param;

            ComputationTime m_time;


    };
}
