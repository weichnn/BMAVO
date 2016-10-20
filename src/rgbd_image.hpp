#include <Eigen/Eigen>
#include <iostream>
#include <memory>
#include <vector>

#include "parameter.hpp"

namespace goodguy{
    class rgbd_image{
        public:
            rgbd_image(
                    std::shared_ptr<Eigen::MatrixXf>& intensity, 
                    std::shared_ptr<Eigen::MatrixXf>& depth, 
                    std::shared_ptr<Eigen::MatrixXf>& point_cloud, 
                    std::shared_ptr<Eigen::MatrixXf>& x_derivative, 
                    std::shared_ptr<Eigen::MatrixXf>& y_derivative,
                    camera_parameter& param)
                :m_intensity(intensity), m_depth(depth), m_point_cloud(point_cloud), m_x_derivative(x_derivative), m_y_derivative(y_derivative), m_param(param)
            {


            }

            rgbd_image(){ }


            bool empty(){
                return m_depth == NULL;
            }

            void set(std::shared_ptr<Eigen::MatrixXf>& intensity, 
                    std::shared_ptr<Eigen::MatrixXf>& depth, 
                    std::shared_ptr<Eigen::MatrixXf>& point_cloud,
                    std::shared_ptr<Eigen::MatrixXf>& x_derivative, 
                    std::shared_ptr<Eigen::MatrixXf>& y_derivative,
                    camera_parameter& param)

            {
                m_intensity = intensity;
                m_depth = depth;
                m_point_cloud = point_cloud;
                m_x_derivative = x_derivative;
                m_y_derivative = y_derivative;
                m_param = param;
            }

            void set_intensity(std::shared_ptr<Eigen::MatrixXf>& intensity){
                m_intensity = intensity;
            }
            void set_depth(std::shared_ptr<Eigen::MatrixXf>& depth){
                m_depth = depth;
            }
            void set_point_cloud(std::shared_ptr<Eigen::MatrixXf>& point_cloud){
                m_point_cloud = point_cloud;
            }
            void set_x_derivative(std::shared_ptr<Eigen::MatrixXf>& x_derivative){
                m_x_derivative = x_derivative;
            }
            void set_y_derivative(std::shared_ptr<Eigen::MatrixXf>& y_derivative){
                m_y_derivative = y_derivative;
            }
            void set_param(camera_parameter& param){
                m_param = param;
            }


            std::shared_ptr<Eigen::MatrixXf> get_intensity() const { return m_intensity; }
            std::shared_ptr<Eigen::MatrixXf> get_depth() const { return m_depth; }
            std::shared_ptr<Eigen::MatrixXf> get_point_cloud() const { return m_point_cloud; }
            std::shared_ptr<Eigen::MatrixXf> get_x_derivative() const { return m_x_derivative; }
            std::shared_ptr<Eigen::MatrixXf> get_y_derivative() const { return m_y_derivative; }
            camera_parameter get_param() const { return m_param; }

        private:

            std::shared_ptr<Eigen::MatrixXf> m_intensity;
            std::shared_ptr<Eigen::MatrixXf> m_depth;
            std::shared_ptr<Eigen::MatrixXf> m_point_cloud;
            std::shared_ptr<Eigen::MatrixXf> m_x_derivative;
            std::shared_ptr<Eigen::MatrixXf> m_y_derivative;
            camera_parameter m_param;
    };
}
