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

#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

#include <nav_msgs/Odometry.h>

#include <std_msgs/Empty.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>


#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "bamvo.hpp"


float g_scale;

bool g_enable_odom_tf;
bool g_enable_odom;

std::string g_camera_name;
std::string g_rgb_frame_name;
std::string g_odom_frame_name;
std::string g_base_frame_name;

void callback(
    const sensor_msgs::ImageConstPtr& image,
    const sensor_msgs::ImageConstPtr& depth,
    const sensor_msgs::CameraInfoConstPtr& camera_info,
    ros::Publisher pub_odom,
    tf::TransformBroadcaster* tf_broadcaster_ptr,
    tf::TransformListener* tf_listener_ptr,
    goodguy::bamvo* vo)
{

    ros::Time received_time = ros::Time::now();

    cv::Mat rgb_in = cv_bridge::toCvShare(image,"bgr8")->image;
    cv::Mat depth_in = cv_bridge::toCvShare(depth)->image;

    cv::Mat depth_meter;
    if(depth_in.type() == CV_16UC1) {
        depth_in.convertTo( depth_meter, CV_32FC1, 1.0/1000.0 );
    }
    else if(depth_in.type() == CV_32FC1) {
        depth_in.copyTo(depth_meter);
    }
    else {
        std::cerr << "Wrong Depth! " << std::endl;
        return;
    }

    int cols = camera_info->width*g_scale;
    int rows = camera_info->height*g_scale;

    vo->get_param().camera_params.fx = camera_info->K[0]*g_scale;
    vo->get_param().camera_params.cx = camera_info->K[2]*g_scale;
    vo->get_param().camera_params.fy = camera_info->K[4]*g_scale;
    vo->get_param().camera_params.cy = camera_info->K[5]*g_scale;


    cv::Size compute_size(cols, rows);
    cv::Mat rgb_resize, depth_resize;
    cv::resize(rgb_in, rgb_resize, compute_size);
    cv::resize(depth_meter, depth_resize, compute_size);

    vo->add(rgb_resize, depth_resize);

    Eigen::Matrix4f curr_pose = vo->get_current_pose().inverse();

    tf::StampedTransform base2rgb_tf;
    try {
        tf_listener_ptr->lookupTransform(g_base_frame_name, g_rgb_frame_name, ros::Time(0), base2rgb_tf);
    } catch (tf::TransformException& ex) {
        ROS_ERROR("TRANSFORM EXCEPTION: %s", ex.what());
        return;
    }

    Eigen::Affine3d base2rgb_eigen;
    tf::transformTFToEigen(base2rgb_tf, base2rgb_eigen);

    Eigen::Affine3d global_pose_eigen(curr_pose.cast<double>());
    global_pose_eigen = base2rgb_eigen * global_pose_eigen * base2rgb_eigen.inverse();

    geometry_msgs::PoseStamped local_camera_pose;
    tf::poseEigenToMsg(global_pose_eigen, local_camera_pose.pose);
    local_camera_pose.header.stamp = received_time;
    local_camera_pose.header.frame_id = g_rgb_frame_name;

    geometry_msgs::PoseStamped global_pose;
    tf::poseEigenToMsg(global_pose_eigen, global_pose.pose);

    if(g_enable_odom_tf) {
        tf::Transform odom_tf;
        tf::transformEigenToTF(global_pose_eigen, odom_tf);
        tf_broadcaster_ptr->sendTransform(tf::StampedTransform(odom_tf, received_time, g_odom_frame_name,  g_base_frame_name));
    }

    if(g_enable_odom) {
        nav_msgs::Odometry odom_nav;
        odom_nav.header.stamp = received_time;

        odom_nav.child_frame_id = g_base_frame_name;
        odom_nav.header.frame_id = g_odom_frame_name;
        odom_nav.pose.pose = global_pose.pose;

        pub_odom.publish(odom_nav);

    }

    cv::imshow("Received RGB image", rgb_resize);
    cv::imshow("Received Depth image", depth_resize);

    cv::waitKey(1);
}



int main(int argc, char** argv) {

    ros::init(argc, argv, "bamvo");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::NodeHandle nh;
    ros::NodeHandle local_nh("~");

    std::string depth_image_name, rgb_image_name, camera_info_name;

    local_nh.getParam("scale", g_scale);
    local_nh.getParam("camera", g_camera_name);
    local_nh.getParam("rgb_frame_id", g_rgb_frame_name);
    local_nh.getParam("odom_frame_id", g_odom_frame_name);
    local_nh.getParam("base_frame_id", g_base_frame_name);
    local_nh.getParam("enable_odom_tf", g_enable_odom_tf);
    local_nh.getParam("enable_odom", g_enable_odom);
    local_nh.getParam("depth_image", depth_image_name);
    local_nh.getParam("rgb_image", rgb_image_name);
    local_nh.getParam("camera_info", camera_info_name);





    goodguy::bamvo vo;

    ros::Publisher pub_odom = local_nh.advertise<nav_msgs::Odometry>(g_odom_frame_name, 50);

    tf::TransformBroadcaster tf_broadcaster;
    tf::TransformListener tf_listener;

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, g_camera_name + std::string("/") + rgb_image_name, 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, g_camera_name + std::string("/") + depth_image_name, 10);
    message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub(nh, g_camera_name + std::string("/") + camera_info_name, 10);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> RGBDSyncPolicy;
    message_filters::Synchronizer<RGBDSyncPolicy> sync(RGBDSyncPolicy(10), image_sub, depth_sub, camera_info_sub);

    sync.registerCallback(boost::bind(&callback, _1, _2, _3, pub_odom, &tf_broadcaster, &tf_listener, &vo));


    ros::waitForShutdown();

    return 0;
}

