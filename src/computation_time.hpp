#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>


namespace goodguy{
    class ComputationTime{
        public:
            ComputationTime(){

            }
            ComputationTime(const std::string& prefix_str): m_prefix_str(prefix_str), m_step(0), m_prior_count(m_timer.elapsed().wall){

            }


            void global_tic(){
                m_global_prior_count = m_timer.elapsed().wall;
            }

            void global_toc(){
                boost::timer::nanosecond_type diff = m_timer.elapsed().wall - m_global_prior_count;
                std::cout << m_prefix_str << " [TOTAL COMPUTATION TIME] " << diff/1000.0/1000.0 << " ms" << std::endl;
            }



            void reset(){
                m_step = 0;

            }

            void tic(){
                m_prior_count = m_timer.elapsed().wall;
                m_step++;
            }

            double toc(){
                boost::timer::nanosecond_type diff = m_timer.elapsed().wall - m_prior_count;
                std::cout << m_prefix_str << " [Step " << m_step << " ] " << diff/1000.0/1000.0 << " ms" << std::endl;

                return diff/1000.0/1000.0;
            }


        private:

            std::string m_prefix_str;

            int m_step;
            boost::timer::auto_cpu_timer m_timer;
            boost::timer::nanosecond_type m_prior_count;
            boost::timer::nanosecond_type m_global_prior_count;

    };

}
