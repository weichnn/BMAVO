#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

#include <nav_msgs/Odometry.h>


#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>


#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


#include "bamvo.hpp"

void callback(
        const sensor_msgs::ImageConstPtr& image, 
        const sensor_msgs::ImageConstPtr& depth, 
        const sensor_msgs::CameraInfoConstPtr& camera_info, 
        goodguy::bamvo* vo)
{
    cv::Mat rgb_in = cv_bridge::toCvShare(image,"bgr8")->image;
    cv::Mat depth_in = cv_bridge::toCvShare(depth)->image;

    cv::Mat depth_meter;
    if(depth_in.type() == CV_16UC1){
        depth_in.convertTo( depth_meter, CV_32FC1, 1.0/1000.0 );
    }
    else if(depth_in.type() == CV_32FC1){
        depth_in.copyTo(depth_meter);
    }
    else{
        std::cerr << "Wrong Depth! " << std::endl;
        return;
    }

    cv::Mat rgb_resize, depth_resize;

    cv::Size compute_size(640,480);
    //cv::Size compute_size(320,240);

    cv::resize(rgb_in, rgb_resize, compute_size);
    cv::resize(depth_meter, depth_resize, compute_size);

    vo->add(rgb_resize, depth_resize);

    cv::imshow("Received RGB image", rgb_in);
    cv::imshow("Received Depth image", depth_meter);

    cv::waitKey(1);
}



int main(int argc, char** argv){

    ros::init(argc, argv, "bamvo");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::NodeHandle nh;

    goodguy::bamvo vo;

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth_registered/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub(nh, "/camera/rgb/camera_info", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> RGBDSyncPolicy;
    message_filters::Synchronizer<RGBDSyncPolicy> sync(RGBDSyncPolicy(10), image_sub, depth_sub, camera_info_sub);

    sync.registerCallback(boost::bind(&callback, _1, _2, _3, &vo));


    ros::waitForShutdown();

    return 0;
}

