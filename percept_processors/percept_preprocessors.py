"""
Classes for preprocessing percepts.
"""
__authors__ = ["Dustin Webb", "Tom Le Paine"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb", "Tom Le Paine"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

# Standard
import Image
from time import time
import cPickle
from subprocess import call
import glob
import os
import os.path as op

# Third-party
import numpy as np
from theano import config

# Internal
import hedgehog.pylearn2.utils as utils


def ensure_dir_exists(directory):
    if not op.exists(directory):
        os.makedirs(directory)


class PerceptPreprocessor(object):
    """
    Base class for percept preprocessing.

    movie_dir: string
        Location wherein to save videos.
    frame_name_template: string
        Optional FFMPEG format string for identifying frame files.
    save_percepts: boolean
        Whether to save and write the percepts.
    """
    def __init__(
        self,
        movie_dir,
        frame_name_template='frame_%07d.png',
        percept_name_template='percept_%07d.png',
        save_percepts=False
    ):
        # Validate parameters and store
        assert(movie_dir)
        self.movie_dir = movie_dir

        assert(frame_name_template)
        self.frame_name_template = frame_name_template

        assert(percept_name_template)
        self.percept_name_template = percept_name_template

        self.save_percepts = save_percepts
        self.percept_count = 0

        self.frames_tmp_dir = os.path.join(self.movie_dir, 'frames_tmp')
        self.percepts_tmp_dir = os.path.join(self.movie_dir, 'percepts_tmp')

        self.reset_frames()
        self.reset_percepts()

    def reset_frames(self):
        self.frames = []

    def reset_percepts(self):
        self.percepts = []

    def write_images(self, loc, name_template, images, idx_offset=0):
        """
        Writes frames to disk.

        loc: string
            Directory wherein frames should be saved.
        """
        if not os.path.exists(loc):
            os.makedirs(loc)

        print("Saving frames to %s..." % loc),

        tic = time()

        for idx, image in enumerate(images):
            img_obj = Image.fromarray(image)
            image_file = os.path.join(loc, name_template % (idx_offset+idx))
            img_obj.save(open(image_file, 'w'))

        toc = time()

        print 'Done. Took %0.2f sec.' % (toc-tic)

        return len(images)

    def remove_frames(self, loc):
        """
        Removes frames from disk.

        loc: string
            Directory which contains frames to be deleted.
        """
        print("Removing frames from %s..." % loc),

        frame_name_glob = os.path.join(loc, '*.png')

        tic = time()

        ret = call(['rm'] + glob.glob(frame_name_glob))

        toc = time()

        print 'Done with status: %d. Took %0.2f sec.' % (ret, toc-tic)

    def create_video(self, src, dest, name):
        """
        Creates a video in specified location from specified frames.

        src: string
            Location of frames to be converted to video.
        dest: string
            Location wherein to store the video.
        name:
            Name of the video.
        """
        assert(os.path.exists(src))

        if not os.path.exists(dest):
            os.makedirs(dest)

        frame_files = os.path.join(src, self.frame_name_template)
        video_file = os.path.join(
            dest,
            name
        )

        print("Creating video (%s)..." % video_file),

        tic = time()

        ret = call([
            'ffmpeg',
            '-v',
            '0',
            '-i',
            frame_files,
            video_file
        ])

        toc = time()

        info = (ret, toc-tic)  # Appease pep8 on the next line
        print("Done with status: %d. Took %0.2f sec." % info),

    def save_video(self, name, reset_frames_buffer=True):
        """
        Writes frames to disk and then creates a video from them.

        name: string
            Name of the video.
        reset_frames_buffer: boolean
            Whether to reset the frames buffer. Useful for making partial
            videos.
        """
        ensure_dir_exists(self.frames_tmp_dir)
        self.write_images(
            self.frames_tmp_dir,
            self.frame_name_template,
            self.frames
        )
        self.create_video(self.frames_tmp_dir, self.movie_dir, name)
        self.remove_frames(self.frames_tmp_dir)

        if reset_frames_buffer:
            self.reset_frames()

    def write_percepts(self):
        """
        Writes percepts to disk.
        """
        ensure_dir_exists(self.percepts_tmp_dir)
        if self.save_percepts:
            self.percept_count += self.write_images(
                self.percepts_tmp_dir,
                self.percept_name_template,
                self.percepts,
                self.percept_count
            )


class DeepMindPreprocessor(PerceptPreprocessor):
    """
    Percept preprocessor for specifications defined in the paper Playing Atari
    with Deep Reinforcement Learning.

    img_dims: tuple
        Crop size for image.
    movie_dir: string
        Location wherein to store videos.
    """
    def __init__(self, img_dims, movie_dir, save_percepts=False):
        super(DeepMindPreprocessor, self).__init__(
            movie_dir,
            save_percepts=save_percepts
        )

        assert(type(img_dims) == tuple and len(img_dims) == 2)
        self.img_dims = img_dims
        self.offset = 128  # Start of image in observation
        self.atari_frame_size = (210, 160)
        self.reduced_frame_size = (110, 84)  # Prescribed by paper
        self.crop_start = (20, 0)  # Crop start prescribed by authors

        palette_path = os.path.dirname(os.path.realpath(__file__))
        palette_path = os.path.join(palette_path, 'palettes')
        palette_path = os.path.join(palette_path, 'stella_palette.pkl')
        print "Loading palette (%s)..." % palette_path
        self.palette = cPickle.load(open(palette_path, 'rb'))

    def get_frame(self, observation):
        """
        Extracts frame from RL-Glue observation.

        observation: Observation
            RL-Glue observation object.
        """
        #  TODO confirm this does a deep copy
        image = utils.observation_to_image(
            observation,
            self.offset,
            self.atari_frame_size
        )

        # Apply Atari palette
        image /= 2  # Calculate idxs into Atari 2600 pallete
        image = self.palette[image]

        self.frames.append(image)

        # Resize and crop
        image = utils.resize_image(image, self.reduced_frame_size)
        image = utils.crop_image(image, self.crop_start, self.img_dims)

        if self.save_percepts:
            self.percepts.append(image)

        # Convert to black and white
        image = np.sqrt(np.sum((image**2), axis=2))
        image = image.astype(np.uint8)

        # 26. Was the max value
        # in the frames at the time of inspection
        image = (image.astype(config.floatX) / 26.) - 0.5

        return image
