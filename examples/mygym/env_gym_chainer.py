import warnings
import gym
from chainerrl import env
from chainerrl import spaces

import skimage
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.color import gray2rgb
import collections

import numpy as np
from collections import deque


try:
    import cv2

    def imresize(img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

except Exception:
    from PIL import Image, ImageFilter

    warnings.warn(
        'Since cv2 is not available PIL will be used instead to resize images.'
        ' This might affect the resulting images.')

    def imresize(img, size):
        return np.asarray(Image.fromarray(img).resize(
            size,
            Image.BILINEAR).filter(ImageFilter.EDGE_ENHANCE))
    

class GymEnvironment(env.Env):
    """ Small wrapper for gym environments

    preprocesss screens and holds it onto a screen
    buffer of size agent_history_length from
    which the environment state is constructed

    """

    def __init__(self, env, res_width, res_height, agent_history_length,
                 render=False):
        """ Initialization stuff
        
        Parameters:
        -----------
        env: gym environment
        res_width: resized width
        res_height: resized height
        agent_history_length: buffer length

        """
        self.env = env
        self.res_width = res_width
        self.res_height = res_height
        self.n_last_screens = agent_history_length
        self.legal_actions = range(env.action_space.n)

        self._terminal = False

        # Screen buffer of size agent_history_length
        self.last_screens = deque()
        self.last_raw_screen = None

        # Render agent input
        self.viewer = None
        self.render = render

    def _render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # Pick last screen
        img = self.last_screens[-1]

        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(imresize(gray2rgb(img), (128, 128)))
        #self.viewer.imshow(gray2rgb(rgb2gray(self.last_raw_screen), alpha=False))
        #self.viewer.imshow(imresize(self.last_raw_screen, (self.res_width, self.res_height)))
        
    def _get_preprocessed_frame(self, observation, preserve_range=False):
        """
        
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """

        rgb_img = observation
        
        img = rgb_img[:, :, 0] * 0.2126 + rgb_img[:, :, 1] * \
            0.0722 + rgb_img[:, :, 2] * 0.7152
        img = img.astype(np.uint8)
        img = imresize(img, (self.res_width, self.res_height))

        #image = np.array(resize(rgb2gray(observation), (self.res_width,
        #                                                self.res_height),
        #                        preserve_range=preserve_range),
        #                 dtype="float32")
        return img

    def current_screen(self):
        """

        """
        return self._get_preprocessed_frame(self.last_raw_screen,
                                            preserve_range=False)
    
    def initialize(self):
        """

        """
        self.last_raw_screen = self.env.reset()
        
        # Clear the state buffer
        self.last_screens = deque()

        self.last_screens = collections.deque(
            [np.zeros((self.res_width, self.res_height),
                      dtype="float32")] * 3 +
            [self.current_screen()],
            maxlen=self.n_last_screens)
        
        self._reward = 0
        self._terminal = False
    
    def step(self, action):
        """ Execute an action
        
        action: id of an action

        """

        (self.last_raw_screen, self._reward,
         self._terminal, info) = self.env.step(action)

        # We must have last screen here unless it's terminal
        if not self.is_terminal:
            self.last_screens.append(self.current_screen())

        if self.render:
            self._render()
            
        return self.state, self.reward, self.is_terminal, {}
    
    def reset(self):
        self.initialize()

        return self.state

    def close(self):
        self.env.close()
        
    @property
    def state(self):
        assert len(self.last_screens) == self.n_last_screens
        #print "MY STATE ", len(self.last_screens)
        
        return list(self.last_screens)

    @property
    def is_terminal(self):
        return self._terminal

    @property
    def reward(self):
        return self._reward

    @property
    def number_of_actions(self):
        return len(self.legal_actions)

