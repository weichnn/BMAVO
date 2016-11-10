#ifndef __PARAMETER_HPP__
#define __PARAMETER_HPP__
#include <vector>

namespace goodguy{
    struct camera_parameter{
        float fx;
        float fy;
        float cx;
        float cy;
    };

    struct depth_range{
        float min;
        float max;
    };

    struct bamvo_parameter{
        bamvo_parameter(){
            set_default();
        }
        void set_default(){
            camera_params.fx = 549.094829;
            camera_params.fy = 545.329258;
            camera_params.cx = 292.782603;
            camera_params.cy = 226.667207;

            hist_size = 4;
            sig_min = 0.000001;

            bgm_level = 0.0; // Same as level for image pyramid

            iter_count.clear();
            iter_count.push_back(10);
            iter_count.push_back(10);
            iter_count.push_back(10);

            grad_mag_min.clear();
            grad_mag_min.push_back(12);
            grad_mag_min.push_back(5);
            grad_mag_min.push_back(3);

            range_bgm.min = 0.5;
            range_bgm.max = 4.0;

            range_odo.min = 0.5;
            range_odo.max = 4.0;
        }


        float hist_size;
        float sig_min;

        std::size_t bgm_level;

        depth_range range_bgm;
        depth_range range_odo;

        camera_parameter camera_params;

        std::vector<int> iter_count;
        std::vector<float> grad_mag_min;

    };

    bamvo_parameter get_default_param(){
        return bamvo_parameter();
    }
}

#endif
