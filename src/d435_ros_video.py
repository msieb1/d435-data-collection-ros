# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Collect images from multiple simultaneous webcams.

Usage:

1. Define some environment variables that describe what you're collecting.
dataset=tutorial
mode=train
num_views=2
viddir=/tmp/tcn/videos
tmp_imagedir=/tmp/tcn/tmp_images
debug_vids=1

2. Run the script.
export DISPLAY=:0.0 && \
root=~/max/models/research/tcn && \
bazel build -c opt --copt=-mavx webcam && \
bazel-bin/webcam \
--dataset $dataset \
--mode $mode \
--num_views $num_views \
--tmp_imagedir $tmp_imagedir \
--viddir $viddir \
--debug_vids 1 \
--logtostderr

3. Hit Ctrl-C when done collecting, upon which the script will compile videos
for each view and optionally a debug video concatenating multiple
simultaneous views.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import multiprocessing
from multiprocessing import Process
import imageio
import os
import subprocess
import sys
import time
import cv2
import string
import matplotlib
from os.path import join
import tf
matplotlib.use('TkAgg')

from matplotlib import animation  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
from six.moves import input
import wave
from scipy import misc
import pyrealsense2 as rs
import io
import rospy
from pdb import set_trace as st
import json
import baxter_interface

sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-2]))
from utils.subscribers import img_subscriber, depth_subscriber

GLOBAL_IMAGE_BUFFER = []
GLOBAL_DEPTH_BUFFER = []
GLOBAL_DEPTH_RAW_BUFFER = []

# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--index', type=str, default='2,3')
# args = parser.parse_args()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='training_2', help='Name of the dataset we`re collecting.')
parser.add_argument('--mode', default='',help='What type of data we`re collecting. E.g.:'
                       '`train`,`valid`,`test`, or `demo`')
parser.add_argument('--seqname', default='',help='Name of this sequence. If empty, the script will use'
                       'the name seq_N+1 where seq_N is the latest'
                       'integer-named sequence in the videos directory.')
parser.add_argument('--num_views', type=int, default=2,help='Number of webcams.')
parser.add_argument('--expdir', default='/home/zhouxian/projects/experiments',
                       help='dir to write experimental data to.')
parser.add_argument('--tmp_imagedir', default='/tmp/tcn/data',
                       help='Temporary outdir to write images.')
parser.add_argument('--viddir', default='/tmp/tcn/videos',
                       help='Base directory to write videos.')
parser.add_argument('--depthdir', default='/tmp/tcn/depth',
                       help='Base directory to write depth.')
parser.add_argument('--debug_vids', default=False,
                        help='Whether to generate debug vids with multiple concatenated views.')
parser.add_argument('--debug_lhs_view', default='1',
                       help='Which viewpoint to use for the lhs video.')
parser.add_argument('--debug_rhs_view', default='2',
                       help='Which viewpoint to use for the rhs video.')
parser.add_argument('--height', default=1080, help='Raw input height.')
parser.add_argument('--width', default=1920, help='Raw input width.')
# parser.add_argument('webcam_ports', '{}'.format(args.index),
#                        'Comma-separated list of each webcam usb port.')
parser.add_argument('--webcam_ports', default='2,5,8',help='Comma-separated list of each webcam usb port.')
args = parser.parse_args()
args.mode=''
FPS = 25.0

class ImageQueue(object):
  """An image queue holding each stream's most recent image.

  Basically implements a process-safe collections.deque(maxlen=1).
  """

  def __init__(self):
    self.lock = multiprocessing.Lock()
    self._queue = multiprocessing.Queue(maxsize=1)

  def append(self, data):
    with self.lock:
      if self._queue.full():
        # Pop the first element.
        _ = self._queue.get()
      self._queue.put(data)

  def get(self):
    with self.lock:
      return self._queue.get()

  def empty(self):
    return self._queue.empty()

  def close(self):
    return self._queue.close()


def timer(start, end):
  """Returns a formatted time elapsed."""
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)


def setup_paths():
  """Sets up the necessary paths to collect videos."""
  assert args.dataset
  assert args.num_views
  assert args.expdir

  # Setup directory for final images used to create videos for this sequence.
  tmp_imagedir = os.path.join(args.tmp_imagedir, args.dataset, args.mode)
  if not os.path.exists(tmp_imagedir):
    os.makedirs(tmp_imagedir)

  # Create a base directory to hold all sequence videos if it doesn't exist.
  vidbase = os.path.join(args.expdir, args.dataset, args.viddir, args.mode)
  if not os.path.exists(vidbase):
    os.makedirs(vidbase)

  # Get one directory per concurrent view and a sequence name.
  view_dirs, seqname = get_view_dirs(vidbase, tmp_imagedir)

  # Get an output path to each view's video.
  vid_paths = []
  for idx, _ in enumerate(view_dirs):
    vid_path = os.path.join(vidbase, '%s_view%d.mp4' % (seqname, idx))
    vid_paths.append(vid_path)

  # Optionally build paths to debug_videos.
  debug_path = None
  if args.debug_vids:
    debug_base = os.path.join('%s_debug' % args.viddir, args.dataset,
                              args.mode)
    if not os.path.exists(debug_base):
      os.makedirs(debug_base)
    debug_path = '%s/%s.mp4' % (debug_base, seqname)

  return view_dirs, vid_paths, debug_path, seqname

def setup_paths_w_depth():
  """Sets up the necessary paths to collect videos."""
  assert args.dataset
  # assert args.mode
  assert args.num_views
  assert args.expdir

  # Setup directory for final images used to create videos for this sequence.
  if args.mode == '':
    tmp_imagedir = os.path.join(args.tmp_imagedir, args.dataset)
  else:
    tmp_imagedir = os.path.join(args.tmp_imagedir, args.dataset, args.mode)

  if not os.path.exists(tmp_imagedir):
    os.makedirs(tmp_imagedir)
  if args.mode == '':
    tmp_depthdir = os.path.join(args.tmp_imagedir,  args.dataset, 'depth', args.mode)
  else:
    tmp_depthdir = os.path.join(args.tmp_imagedir,  args.dataset, 'depth', args.mode)

  if not os.path.exists(tmp_depthdir):
    os.makedirs(tmp_depthdir)
  # Create a base directory to hold all sequence videos if it doesn't exist.
  if args.mode == '':
    vidbase = os.path.join(args.expdir, args.dataset, args.viddir)
  else:
    vidbase = os.path.join(args.expdir, args.dataset, args.viddir, args.mode)

  if not os.path.exists(vidbase):
    os.makedirs(vidbase)

    # Setup depth directory
  if args.mode == '':
    depthbase = os.path.join(args.expdir, args.dataset, args.depthdir)
  else:
    depthbase = os.path.join(args.expdir, args.dataset, args.depthdir, args.mode)

  if not os.path.exists(depthbase):
    os.makedirs(depthbase)
  # Get one directory per concurrent view and a sequence name.
  view_dirs, seqname = get_view_dirs(vidbase, tmp_imagedir)
  view_dirs_depth = get_view_dirs_depth(vidbase, tmp_depthdir)

  # Get an output path to each view's video.
  vid_paths = []
  for idx, _ in enumerate(view_dirs):
    vid_path = os.path.join(vidbase, '%s_view%d.mp4' % (seqname, idx))
    vid_paths.append(vid_path)
  depth_paths = []
  for idx, _ in enumerate(view_dirs_depth):
    depth_path = os.path.join(depthbase, '%s_view%d.mp4' % (seqname, idx))
    depth_paths.append(depth_path)

  # Optionally build paths to debug_videos.
  debug_path = None
  if args.debug_vids:
    if args.mode == '':
      debug_base = os.path.join(args.expdir, args.dataset, '%s_debug' % args.viddir)
    else:
      debug_base = os.path.join(args.expdir, args.dataset, '%s_debug' % args.viddir, args.mode)
    if not os.path.exists(debug_base):
      os.makedirs(debug_base)
    debug_path = '%s/%s.mp4' % (debug_base, seqname)
    debug_path_depth = '%s/%s_depth.mp4' % (debug_base, seqname)

  return view_dirs, vid_paths, debug_path, seqname, view_dirs_depth, depth_paths, debug_path_depth

def get_view_dirs(vidbase, tmp_imagedir):
  """Creates and returns one view directory per webcam."""
  # Create and append a sequence name.
  if args.seqname:
    seqname = args.seqname
  else:
    # If there's no video directory, this is the first sequence.
    if not os.listdir(vidbase):
      seqname = '0'
    else:
      # Otherwise, get the latest sequence name and increment it.
      seq_names = [i.split('_')[0] for i in os.listdir(vidbase)]
      latest_seq = sorted(map(int, seq_names), reverse=True)[0]
      seqname = str(latest_seq+1)
    print('No seqname specified, using: %s' % seqname)
  view_dirs = [os.path.join(
      tmp_imagedir, '%s_view%d' % (seqname, v)) for v in range(args.num_views)]
  # for d in view_dirs:
  #   if not os.path.exists(d):
  #     os.makedirs(d)
  return view_dirs, seqname


def get_view_dirs_depth(depthbase, tmp_depthdir):
  """Creates and returns one view directory per webcam."""
  # Create and append a sequence name.
  if args.seqname != -1:
    seqname = args.seqname
  else:
    # If there's no video directory, this is the first sequence.
    if not os.listdir(depthbase):
      seqname = '0'
    else:
      # Otherwise, get the latest sequence name and increment it.
      seq_names = [i.split('_')[0] for i in os.listdir(depthbase)]
      latest_seq = sorted(map(int, seq_names), reverse=True)[0]
      seqname = str(latest_seq+1)
    print('No seqname specified, using: %s' % seqname)
  view_dirs_depth = [os.path.join(
      tmp_depthdir, '%s_view%d' % (seqname, v)) for v in range(args.num_views)]
  # for d in view_dirs_depth:
  #   if not os.path.exists(d):
  #     os.makedirs(d)
  return view_dirs_depth


def collect_images_parallel(device_ids):

  topic_img_list = ['/camera' + device_id + '/color/image_raw' for device_id in device_ids]
  topic_depth_list = ['/camera' + device_id + '/aligned_depth_to_color/image_raw' for device_id in device_ids]
  img_subs_list = [img_subscriber(topic=topic_img) for topic_img in topic_img_list]
  depth_subs_list = [depth_subscriber(topic=topic_depth) for topic_depth in topic_depth_list]
  depth_scale = 0.001 # not precisely, but up to e-8
  print( "Depth Scale is: " , depth_scale)
  rospy.sleep(2)
  # We will be removing the background of objects more than
  #  clipping_distance_in_meters meters away
  clipping_distance_in_meters = 1.5 #1 meter
  clipping_distance = clipping_distance_in_meters / depth_scale

  # Take some ramp images to allow cams to adjust for brightness etc.
  # for i in range(100):
  #   # Get frameset of color and depth
  #   idx = 0
  #   # set_trace()
  #   for img_subs, depth_subs in zip(img_subs_list, depth_subs_list):
  #     color_image = img_subs.img
  #     depth_image = depth_subs.img
  #     idx += 1
  #   # Warm camera up (picture quality is bad in beginning frames)
  #   print('Taking ramp image %d.' % i)
          
  frame_count = 0
  # Streaming loop
  start_time = time.time()
  depth_stacked = []



  try:
    # rospy.sleep(0.5)
    print("Start collecting images...")
    while True:
      curr_time = time.time()
      color_view_buffer = []
      depth_view_buffer = []
      depth_raw_buffer = []

      for img_subs, depth_subs in zip(img_subs_list, depth_subs_list):

        color_image = img_subs.img
        depth_image = depth_subs.img
        depth_raw_buffer.append(np.asarray(depth_image))
        depth_image[np.where(depth_image > clipping_distance)] = 0
        depth_rescaled = ((depth_image  - 0) / (clipping_distance - 0)) * (255 - 0) + 0
        # depth_image_3d = np.dstack((depth_rescaled,depth_rescaled,depth_rescaled)) #depth image is 1 channel, color is 3 channels
        depth_image_3d = depth_rescaled
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        color_view_buffer.append(color_image)
        depth_view_buffer.append(depth_rescaled)
        # Render images
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
      GLOBAL_IMAGE_BUFFER.append(color_view_buffer)
      GLOBAL_DEPTH_BUFFER.append(depth_view_buffer)
      GLOBAL_DEPTH_RAW_BUFFER.append(depth_raw_buffer)

      # cv2.imwrite(os.path.join(view_dir, '{0:08d}.png'.format(frame_count)), color_image)
      # cv2.imwrite(os.path.join(view_dir_depth, '{0:08d}.png'.format(frame_count)), depth_image_3d)

      run_time = time.time() - curr_time
      time.sleep(max(1.0/FPS - run_time, 0.0))
      frame_count += 1
      current = time.time()
      if frame_count % 100 == 0:

        # print('Collected %s of video, %d frames at ~%.2f fps.' % (
        #     timer(start_time, current), frame_count, frame_count/(current-start_time)))
        print('Collected %s of video, %d frames at ~%.2f fps, time since start: %f' % (
            timer(start_time, current), frame_count, frame_count/(current-start_time), current-start_time))
  finally:
    pass

def save_baxter_values(limb, seqname):
    trial_limb = limb
    trial_arm = baxter_interface.Limb(trial_limb)
    joint_names = trial_arm.joint_names()
    q = [trial_arm.joint_angle(j) for j in joint_names]
    dq = [trial_arm.joint_velocity(j) for j in joint_names]
    pos = list(trial_arm.endpoint_pose()['position'])
    orn = list(trial_arm.endpoint_pose()['orientation'])
    vals = {
          'limb': limb,
          'joint_angles': q,
          'joint_vels': dq,
          'ee_pos': pos,
          'ee_orn': orn,
    }
    basedir = os.path.join(args.expdir, args.dataset, 'baxter_init_vals', args.mode)
    if not os.path.exists(basedir):
      os.makedirs(basedir)
    with open(join(basedir, '{}.json'.format(seqname)), 'w') as fp:
        json.dump(vals, fp)

def save_tf_frame_values(view_idx, seqname, device_indices, tf_listener):
  rospy.sleep(0.1)
  world_frame = '/base'
  camera_frame = '/' + device_indices[view_idx] + '_color_optical_frame'
  # trans, quat_xyzw = tf_listener.lookupTransform(world_frame, camera_frame, rospy.Time(0))
  # T_camera_world = tf_listener.fromTranslationRotation(trans, quat_xyzw)
  basedir = os.path.join(args.expdir, args.dataset, 'tf_frame_vals', args.mode)
  T_camera_world = np.zeros((4, 4))
  if not os.path.exists(basedir):
    os.makedirs(basedir)
  with open(join(basedir, '{}_view{}.json'.format(seqname, view_idx)), 'w') as fp:
      json.dump(T_camera_world.tolist(), fp)
  np.save(join(basedir, '{}_view{}.npy'.format(seqname, view_idx)), T_camera_world)

def main():
  # Initialize the camera capture objects.
  # Get one output directory per view.
  rospy.init_node("data_collection", disable_signals=True)
  try:
    tf_listener = tf.TransformListener()

    ctx = rs.context()
    ds5_dev = rs.device()
    devices = ctx.query_devices()
    # device_ids = ['817612071456', '819112072363', '801212070655']  # Manuela's lab USB IDs
    device_ids = ['817612071456', '826212070528', '826212070219']
    # device_indices = ['1', '2', '3']
    #device_indices = ['1', '2']
    device_indices = ['2']

    collect_images_parallel(device_indices)

  except KeyboardInterrupt:
    print("Make videos..")
    assert len(GLOBAL_DEPTH_BUFFER) == len(GLOBAL_IMAGE_BUFFER)


    view_dirs, vid_paths, debug_path, seqname, view_dirs_depth, depth_paths, debug_path_depth = setup_paths_w_depth()
    # Save baxter joint values
    save_baxter_values(limb='right', seqname=seqname)
    for view_idx in range(len(device_indices)):
      save_tf_frame_values(view_idx, seqname, device_indices=device_indices, tf_listener=tf_listener)

    writers = [imageio.get_writer(vidpath, fps=FPS) for vidpath in vid_paths]
    writers_depth = [imageio.get_writer(depthpath, fps=FPS) for depthpath in depth_paths]
    writer_debug = imageio.get_writer(debug_path, fps=FPS) 
    writers_depth_raw = [[]] * len(writers)
    for t in range(0, len(GLOBAL_DEPTH_BUFFER)):
      stacked_images = GLOBAL_IMAGE_BUFFER[t][0]

      for view_idx in range(len(device_indices)):
        writers[view_idx].append_data(GLOBAL_IMAGE_BUFFER[t][view_idx])
        writers_depth[view_idx].append_data(GLOBAL_DEPTH_BUFFER[t][view_idx].astype(np.uint8))
        writers_depth_raw[view_idx].append(GLOBAL_DEPTH_RAW_BUFFER[t][view_idx])
        if view_idx > 0:
          stacked_images = np.concatenate([stacked_images, GLOBAL_IMAGE_BUFFER[t][view_idx]], axis=1)
      writer_debug.append_data(stacked_images)
    for i, depthpath in enumerate(depth_paths):
      np.save(depthpath.split('.mp4')[0] + '.npy', writers_depth_raw[i])

    for idx in range(len(writers)):
      writers[idx].close()
      writers_depth[idx].close()
      writer_debug.close()

    for p, q in zip(vid_paths, depth_paths):
      print('Writing final color video to: %s' % p)
      print('Writing final depth video to: %s' % q)
    if debug_path:
      print('Writing debug video to: %s' % debug_path)


    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)  # pylint: disable=protected-access


if __name__ == '__main__':
  main()
