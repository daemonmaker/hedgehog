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

# Third-party
import numpy as np

# Internal
import hedgehog.pylearn2.utils as utils


class PerceptPreprocessor(object):
    """
    Base class for percept preprocessing.

    movie_dir: string
        Location wherein to save videos.
    frame_name_template: string
        Optional FFMPEG format string for identifying frame files.
    """
    def __init__(self, movie_dir, frame_name_template='frame_%07d.png'):
        # Validate parameters and store
        assert(movie_dir)
        self.movie_dir = movie_dir

        assert(frame_name_template)
        self.frame_name_template = frame_name_template

        self.tmp_dir = os.path.join(self.movie_dir, 'frames_tmp')

        self.reset_frames()

    def reset_frames(self):
        self.frames = []

    def write_frames(self, loc):
        """
        Writes frames to disk.

        loc: string
            Directory wherein frames should be saved.
        """
        if not os.path.exists(loc):
            os.makedirs(loc)

        print("Saving frames to %s..." % loc),

        tic = time()

        for idx, frame in enumerate(self.frames):
            img_obj = Image.fromarray(image)
            frame = os.path.join(loc, self.frame_name_template % idx)
            img_obj.save(open(frame, 'w'))

        toc = time()

        print 'Done. Took %0.2f sec.' % (toc-tic)

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
            '-i',
            frame_files,
            video_file
        ])

        toc = time()

        info = (ret, toc-tic)  # Appease pep8 on the next line
        print("Done with status: %d. Took %0.2f sec." % info),

    def save_video(self, name, reset_frames_buffer=True):
        """
        name: string
            Name of the video.
        reset_frames_buffer: boolean
            Whether to reset the frames buffer. Useful for making partial
            videos.
        """
        self.write_frames(self.tmp_dir)
        self.create_video(self.tmp_dir, self.movie_dir, name)
        self.remove_frames(self.tmp_dir)

        if reset_frames_buffer:
            self.reset_frames()


class DeepMindPreprocessor(PerceptPreprocessor):
    """
    Percept preprocessor for specifications defined in the paper Playing Atari
    with Deep Reinforcement Learning.

    img_dims: tuple
        Crop size for image.
    movie_dir: string
        Location wherein to store videos.
    """
    def __init__(self, img_dims, movie_dir):
        super(DeepMindPreprocessor, self).__init__(movie_dir)

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

        # Convert to black and white
        image = np.sqrt(np.sum((image**2), axis=2))
        image = image.astype(np.uint8)

        # Resize and crop
        image = utils.resize_image(image, self.reduced_frame_size)
        image = utils.crop_image(image, self.crop_start, self.img_dims)

        return image
