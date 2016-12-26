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

#ifndef __PARAMETER_HPP__
#define __PARAMETER_HPP__
#include <vector>

namespace goodguy {
struct camera_parameter {
    float fx;
    float fy;
    float cx;
    float cy;
};

struct depth_range {
    float min;
    float max;
};

struct bamvo_parameter {
    bamvo_parameter() {
        set_default();
    }
    void set_default() {
        camera_params.fx = 549.094829;
        camera_params.fy = 545.329258;
        camera_params.cx = 292.782603;
        camera_params.cy = 226.667207;

        hist_size = 4;
        depth_diff_max = 0.2; // metric unit

        bgm_level = 1.0; // Same as level for image pyramid

        iter_count.clear();
        iter_count.push_back(10);
        iter_count.push_back(10);

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

bamvo_parameter get_default_param() {
    return bamvo_parameter();
}
}

#endif
