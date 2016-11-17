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

#ifndef __RGBD_IMAGE_HPP__
#define __RGBD_IMAGE_HPP__

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
                    std::shared_ptr<Eigen::MatrixXf>& x_derivative, 
                    std::shared_ptr<Eigen::MatrixXf>& y_derivative,
                    camera_parameter& param)
                :m_intensity(intensity), m_depth(depth), m_x_derivative(x_derivative), m_y_derivative(y_derivative), m_param(param)
            {


            }

            rgbd_image(){ }


            bool empty(){
                return m_depth == NULL;
            }

            void set(std::shared_ptr<Eigen::MatrixXf>& intensity, 
                    std::shared_ptr<Eigen::MatrixXf>& depth, 
                    std::shared_ptr<Eigen::MatrixXf>& x_derivative, 
                    std::shared_ptr<Eigen::MatrixXf>& y_derivative,
                    camera_parameter& param)

            {
                m_intensity = intensity;
                m_depth = depth;
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
            void set_x_derivative(std::shared_ptr<Eigen::MatrixXf>& x_derivative){
                m_x_derivative = x_derivative;
            }
            void set_y_derivative(std::shared_ptr<Eigen::MatrixXf>& y_derivative){
                m_y_derivative = y_derivative;
            }
            void set_param(camera_parameter& param){
                m_param = param;
            }


            std::shared_ptr<Eigen::MatrixXf>& get_intensity() { return m_intensity; }
            std::shared_ptr<Eigen::MatrixXf>& get_depth() { return m_depth; }
            std::shared_ptr<Eigen::MatrixXf>& get_x_derivative() { return m_x_derivative; }
            std::shared_ptr<Eigen::MatrixXf>& get_y_derivative() { return m_y_derivative; }
            camera_parameter& get_param() { return m_param; }

        private:

            std::shared_ptr<Eigen::MatrixXf> m_intensity;
            std::shared_ptr<Eigen::MatrixXf> m_depth;
            std::shared_ptr<Eigen::MatrixXf> m_x_derivative;
            std::shared_ptr<Eigen::MatrixXf> m_y_derivative;
            camera_parameter m_param;
    };
}

#endif
