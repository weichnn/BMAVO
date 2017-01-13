# This program pre-requires the installation of ROS
# It was tested on ROS indigo and kinetic versions

## Build program
1. Create catkin_ws for this program
  cd ~/
  mkdir -p ~/research_ws/src/
  cd ~/research_ws/src/ && catkin_init_workspace

2. Clone git repository into your workspace 
  git clone http://bitbucket.org/goodguy/bamvo.git
  cd ~/research_ws/
  catkin_make -j4

3. Export ROS environment variables
  echo "source ~/research_ws/devel/setup.bash" >> ~/.bashrc
  source ~/.bashrc


## Run
1. Launch openni2 node
  roslaunch openni2_launch openni2.launch depth_registration:=true auto_exposure:=false auto_white_balance:=false

2. Launch bamvo node
  roslaunch bamvo bamvo.launch

3. Launch RViz
  rviz
  (rviz) Change "Fixed Frame" variable to "odom" in Global Options
  (rviz) Add "TF" visulization


## Cite us
if you use BaMVO for a publication, please cite it as:
@article{kim2016effective,
  title={Effective Background Model-Based RGB-D Dense Visual Odometry in a Dynamic Environment},
  author={Kim, Deok-Hwa and Kim, Jong-Hwan},
  journal={IEEE Transactions on Robotics},
  volume={32},
  number={6},
  pages={1565--1573},
  year={2016},
  publisher={IEEE}
}