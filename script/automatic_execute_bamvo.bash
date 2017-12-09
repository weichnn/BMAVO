#!/bin/bash

database=~/database/density/
for tum_dir in ${database}/* ;
do
    echo "==========================================="
    echo "Start Dataset [$(basename $tum_dir)] "

    echo "  Check Doubleness"
    last_seq="EMPTY"
    is_already_done=false
    mkdir -p `pwd`/experiments/
    for dataset_dir in `pwd`/experiments/* ;
    do
        echo "$(basename $dataset_dir)"
        if [[ $(basename $dataset_dir) == $(basename $tum_dir) ]]
        then
            echo "$(basename $dataset_dir)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            is_already_done=true
        fi
        last_seq=$(basename $dataset_dir)
    done

    echo "  -----------------------------------------"
    echo "  Start BAMVO"

    save_loc=`pwd`/experiments/$(basename $tum_dir)
    mkdir -p ${save_loc}
    sleep 2



    if [[ $is_already_done == false ]]
    then
        roslaunch rgbd_saver reader.launch dataset:=$tum_dir camera:=camera color_png:=false > ${save_loc}/saver.log&
        reader_pid=$!
        sleep 2
        roslaunch bamvo  bamvo.launch save_loc:=${save_loc}  > ${save_loc}/demo.log& 
        algorithm_pid=$!
        sleep 5  
        rostopic pub /camera/next_frame std_msgs/Empty "{}" -r 15 > ${save_loc}/topic.log&
        topic_pid=$!
        echo "  Wait [$reader_pid]"
        wait $reader_pid
        kill $algorithm_pid
        kill $topic_pid
        sleep 10  
    else
        echo "  SKIP!!!"
    fi


    echo "Finish Dataset [$(basename $tum_dir)]"
done
