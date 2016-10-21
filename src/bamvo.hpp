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
            bamvo(){ }

            void add(const cv::Mat& color, const cv::Mat& depth){
                if(depth.type() != CV_32FC1 || color.type() != CV_8UC3){ 
                    std::cerr << "Not supported data type!" << std::endl; 
                    return; 
                }

                if(!m_curr_depth.empty()){
                    m_hist_poses.emplace_back(std::make_shared<Eigen::Matrix4f>(Eigen::Matrix4f(Eigen::Matrix4f::Identity())));
                    m_hist_depth.emplace_back(std::make_shared<Eigen::MatrixXf>(cv2eigen(m_curr_depth)));
                    m_hist_point_cloud.emplace_back(generate_point_cloud(*(m_hist_depth.back())));


                    if(m_hist_depth.size() > m_param.hist_size){
                        m_hist_depth.pop_front();
                        m_hist_poses.pop_front();
                        m_hist_point_cloud.pop_front();

                        //std::shared_ptr<Eigen::MatrixXf> temp_depth = generate_depth_from_point_cloud(*m_hist_point_cloud.back());
                        //cv::Mat temp_cv = eigen2cv(*temp_depth);
                        //cv::Mat temp_cv2 = eigen2cv(*m_hist_depth.back());
                        //cv::imshow(std::string("TEMP2"), temp_cv);
                        //cv::imshow(std::string("TEMP3"), temp_cv2);
                    }
                }

                // Copy images into current and previous
                m_curr_color.copyTo(m_prev_color);
                m_curr_depth.copyTo(m_prev_depth);
                color.copyTo(m_curr_color);
                depth.copyTo(m_curr_depth);


                
                m_time.global_tic();
                // Compute background model image
                std::pair<std::shared_ptr<Eigen::MatrixXf>, std::shared_ptr<Eigen::MatrixXf>> bgm_set = compute_bgm(m_hist_point_cloud, m_hist_poses);

                std::shared_ptr<Eigen::MatrixXf> bgm = std::get<0>(bgm_set);
                std::shared_ptr<Eigen::MatrixXf> labeled_bgm = std::get<1>(bgm_set);

                if(bgm != NULL){
                    cv::Mat bgm_cv = eigen2cv(*bgm);
                    cv::imshow("BGM", bgm_cv/100.0);
                }
                if(labeled_bgm != NULL){
                    cv::Mat bgm_cv = eigen2cv(*labeled_bgm);
                    cv::imshow("LABELED BGM", bgm_cv);
                }

                if(!m_curr_depth.empty() && !m_prev_depth.empty()){
                    std::shared_ptr<Eigen::MatrixXf> prev_depth_eig = std::make_shared<Eigen::MatrixXf>(cv2eigen(m_prev_depth));
                    std::shared_ptr<Eigen::MatrixXf> curr_depth_eig = std::make_shared<Eigen::MatrixXf>(cv2eigen(m_curr_depth));

                    cv::Mat prev_gray, curr_gray;
                    cv::cvtColor(m_prev_color, prev_gray, CV_BGR2GRAY);
                    cv::cvtColor(m_curr_color, curr_gray, CV_BGR2GRAY);
                    cv::Mat prev_intensity, curr_intensity;
                    prev_gray.convertTo(prev_intensity, CV_32FC1, 1.0/255.0);
                    curr_gray.convertTo(curr_intensity, CV_32FC1, 1.0/255.0);

                    std::shared_ptr<Eigen::MatrixXf> prev_intensity_eig = std::make_shared<Eigen::MatrixXf>(cv2eigen(prev_intensity));
                    std::shared_ptr<Eigen::MatrixXf> curr_intensity_eig = std::make_shared<Eigen::MatrixXf>(cv2eigen(curr_intensity));

                    // Compute Odometry
                    Eigen::Matrix4f curr_pose = compute_odometry(bgm, prev_intensity_eig, prev_depth_eig, curr_intensity_eig, curr_depth_eig);

                    // Current pose between current image and previous in the previous view point(t->t-1)
                    if(m_hist_poses.size() != 0){
                        //// Compute poses

                        // Fix saved pose
                        *(m_hist_poses.back()) = curr_pose;
                    }

                }
                m_time.global_toc();
            }

            void set_param(const bamvo_parameter& param){ m_param = param; }
            bamvo_parameter& get_param(){ return m_param; }

        private:

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


            Eigen::Matrix4f compute_odometry(
                    const std::shared_ptr<Eigen::MatrixXf>& bgm, 
                    const std::shared_ptr<Eigen::MatrixXf>& prev_intensity, 
                    const std::shared_ptr<Eigen::MatrixXf>& prev_depth, 
                    const std::shared_ptr<Eigen::MatrixXf>& curr_intensity, 
                    const std::shared_ptr<Eigen::MatrixXf>& curr_depth)
            {
                ComputationTime odom_time("odometry");
                Eigen::Matrix4f odometry = Eigen::Matrix4f::Identity();


                odom_time.tic();
                std::shared_ptr<Eigen::MatrixXf> derivative_x = calculate_derivative_x(curr_intensity);
                odom_time.toc();
                odom_time.tic();
                std::shared_ptr<Eigen::MatrixXf> derivative_y = calculate_derivative_y(curr_intensity);
                odom_time.toc();
                cv::Mat derivative_x_cv = eigen2cv(*derivative_x);
                cv::Mat derivative_y_cv = eigen2cv(*derivative_y);
                cv::imshow("Derivative_x", derivative_x_cv);
                cv::imshow("Derivative_y", derivative_y_cv);


                return odometry;
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
                    const std::deque<std::shared_ptr<Eigen::Matrix4f>>& poses)
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
                const std::shared_ptr<Eigen::MatrixXf> last_hist_depth = generate_depth_from_point_cloud(warped_hist_point_clouds.back());
                auto lambda_for_calculate_depth_difference = [&](const tbb::blocked_range<std::size_t>& r){
                    for(std::size_t k = r.begin(); k < r.end(); ++k){
                        std::shared_ptr<Eigen::MatrixXf> depth = generate_depth_from_point_cloud_sse(warped_hist_point_clouds[k]);
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
                std::cout << "Epsilon_B: " << epsilon_B << std::endl;

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
                std::cout << "Epsilon_B: " << epsilon_B << std::endl;

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


            std::shared_ptr<Eigen::MatrixXf> generate_depth_from_point_cloud(const Eigen::MatrixXf& point_cloud){
                std::size_t size = point_cloud.cols();

                int cols = 0;
                int rows = 0;

                if(!check_size_for_converting_depth(size, rows, cols)){
                    std::cerr << "Unsupported Image size" << std::endl;
                    return std::shared_ptr<Eigen::MatrixXf>();
                }

                const float& fx = m_param.camera_params.fx;
                const float& fy = m_param.camera_params.fy;
                const float& cx = m_param.camera_params.cx;
                const float& cy = m_param.camera_params.cy;
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

            std::shared_ptr<Eigen::MatrixXf> generate_depth_from_point_cloud_sse(const Eigen::MatrixXf& point_cloud){
                std::size_t size = point_cloud.cols();

                int cols = 0;
                int rows = 0;

                if(!check_size_for_converting_depth(size, rows, cols)){
                    std::cerr << "Unsupported Image size" << std::endl;
                    return std::shared_ptr<Eigen::MatrixXf>();
                }

                const float& fx = m_param.camera_params.fx;
                const float& fy = m_param.camera_params.fy;
                const float& cx = m_param.camera_params.cx;
                const float& cy = m_param.camera_params.cy;


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

            inline std::shared_ptr<Eigen::MatrixXf> generate_point_cloud(const Eigen::MatrixXf& depth){
            //[[gnu::target("default")]] Eigen::MatrixXf generate_point_cloud(const Eigen::MatrixXf& depth){
                std::size_t size = depth.cols()*depth.rows();

                std::shared_ptr<Eigen::MatrixXf> point_cloud(new Eigen::MatrixXf(4,size));

                const float& fx = m_param.camera_params.fx;
                const float& fy = m_param.camera_params.fy;
                const float& cx = m_param.camera_params.cx;
                const float& cy = m_param.camera_params.cy;
                
                for(int i = 0; i < depth.rows(); ++i){
                    for(int j = 0; j < depth.cols(); ++j){
                        std::size_t index = i*depth.cols() + j;
                        float d = depth(i,j);
                        (*point_cloud)(0,index) = (j-cx)*d*(1/fx);
                        (*point_cloud)(1,index) = (i-cy)*d*(1/fy);
                        (*point_cloud)(2,index) = d;
                        (*point_cloud)(3,index) = 1;
                    }
                }
                return point_cloud;
            }

            inline std::shared_ptr<Eigen::MatrixXf> generate_point_cloud_sse(const Eigen::MatrixXf& depth){
            //[[gnu::target("sse3")]] Eigen::MatrixXf generate_point_cloud(const Eigen::MatrixXf& depth){
                std::size_t size = depth.cols()*depth.rows();

                Eigen::MatrixXf point_cloud(size, 4);

                const float& fx = m_param.camera_params.fx;
                const float& fy = m_param.camera_params.fy;
                const float& cx = m_param.camera_params.cx;
                const float& cy = m_param.camera_params.cy;

                __m128 *point_cloud_sse = (__m128*)point_cloud.data();

                __m128 fx_sse = _mm_set1_ps(fx);
                __m128 fy_sse = _mm_set1_ps(fy);
                __m128 fx_inv_sse = _mm_rcp_ps(fx_sse);
                __m128 fy_inv_sse = _mm_rcp_ps(fy_sse);
                __m128 cx_sse = _mm_set1_ps(cx);
                __m128 cy_sse = _mm_set1_ps(cy);

                __m128 one_sse = _mm_set1_ps(1);

                __m128 delta_sse = _mm_set_ps(3,2,1,0);


                for(int i = 0; i < depth.rows()/4; ++i){
                    for(int j = 0; j < depth.cols(); ++j){

                        std::size_t index = j*(depth.rows()/4) + i;

                        __m128 d = _mm_set_ps(depth(i*4+3, j), depth(i*4+2, j),depth(i*4+1, j),depth(i*4+0, j));
                        __m128 x = _mm_set1_ps(j);
                        __m128 x1 = _mm_sub_ps(x, cx_sse);
                        __m128 x2 = _mm_mul_ps(x1, fx_inv_sse);
                        __m128 x3 = _mm_mul_ps(x2, d);

                        __m128 y = _mm_add_ps(_mm_set1_ps(i*4), delta_sse);
                        __m128 y1 = _mm_sub_ps(y, cy_sse);
                        __m128 y2 = _mm_mul_ps(y1, fy_inv_sse);
                        __m128 y3 = _mm_mul_ps(y2, d);

                        point_cloud_sse[index+0*size/4] = x3;
                        point_cloud_sse[index+1*size/4] = y3;
                        point_cloud_sse[index+2*size/4] = d;
                        point_cloud_sse[index+3*size/4] = one_sse;
                    }
                }
                return std::shared_ptr<Eigen::MatrixXf>(new Eigen::MatrixXf(point_cloud.transpose()));
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
            cv::Mat m_curr_color;
            cv::Mat m_curr_depth;

            cv::Mat m_prev_color;
            cv::Mat m_prev_depth;

            Eigen::MatrixXf m_bgm;

            std::deque<std::shared_ptr<Eigen::MatrixXf>> m_hist_depth;
            std::deque<std::shared_ptr<Eigen::MatrixXf>> m_hist_point_cloud;
            std::deque<std::shared_ptr<Eigen::Matrix4f>> m_hist_poses;

            bamvo_parameter m_param;

            ComputationTime m_time;


    };
}
