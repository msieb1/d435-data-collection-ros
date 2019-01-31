# kill -9 $(lsof -t /dev/video0) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video1) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video2) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video3) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video4) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video5) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video6) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video7) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video8) # Make sure port is freed up, somehow D435 port stays busy after terminating run


##### Sorted by experiments and depth, audio, video are subfolders #####
dataset=$1  # Name of the dataset.
seqname=$2
mode=demo  # E.g. 'train', 'validation', 'test', 'demo'.
num_views=1 # Number of webcams.
expdir=/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/cvpr2019/video_data
viddir=videos # Output directory for the videos.
depthdir=depth
auddir=audio # Output directory for the videos.
tmp_imagedir=/home/zhouxian/projects/experiments/tmp
# Temp directory to hold images.
debug_vids=1 # Whether or not to generate side-by-side debug videos.
# seqname=pushing

export DISPLAY=:0.0  # This allows real time matplotlib display.



python src/d435_ros_video.py \
--dataset $dataset \
--mode $mode \
--num_views $num_views \
--tmp_imagedir $tmp_imagedir \
--viddir $viddir \
--depthdir $depthdir \
--auddir $auddir \
--debug_vids 1 \
--expdir $expdir \
--seqname $seqname
# --seqname $seqname 
# python test.py


