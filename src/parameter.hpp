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

            hist_size = 10;
            depth_diff_max = 0.2; // metric unit

            bgm_level = 1.0; // Same as level for image pyramid

            iter_count.clear();
            iter_count.push_back(5);
            iter_count.push_back(20);
            iter_count.push_back(20);
            iter_count.push_back(20);

            range_bgm.min = 0.5;
            range_bgm.max = 5.0;

            range_odo.min = 0.5;
            range_odo.max = 5.0;
        }


        float hist_size;
        float sig_min;
        float depth_diff_max;

        std::size_t bgm_level;

        depth_range range_bgm;
        depth_range range_odo;

        camera_parameter camera_params;

        std::vector<int> iter_count;

    };

    bamvo_parameter get_default_param(){
        return bamvo_parameter();
    }
}

#endif
