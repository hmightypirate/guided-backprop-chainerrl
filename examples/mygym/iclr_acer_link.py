import chainer
from chainer import functions as F
from chainer import links as L
from chainerrl.initializers import LeCunNormal


class ICLRACERHead(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

#        w = chainer.initializers.HeNormal()

        # Layers
        # Input should be 84 x 84 (n channels)
        # 1st layer: (84 - 8) / 4 + 1, 20 x 20 x 32
        # 2nd layer: (20 - 4)/2 + 1, 9 x 9 x 64
        # 3rd layer: (9 - 3)/1 + 1, 7 x 7 x 64 = 3136
        
        layers = [
            L.Convolution2D(n_input_channels,
                            out_channels=32,
                            ksize=8,
                            stride=4,
#                            initialW=w,
                            initial_bias=bias),
            L.Convolution2D(32,
                            out_channels=64,
                            ksize=4,
                            stride=2,
#                            initialW=w,
                            initial_bias=bias),
            
            L.Convolution2D(64,
                            out_channels=64,
                            ksize=3,
                            stride=1,
#                            initialW=w,
                            initial_bias=bias),
            
            L.Linear(3136, n_output_channels,   # FIXME, check INPUT
#                     initialW=LeCunNormal(1e-3),
                     initial_bias=bias),
        ]

        super(ICLRACERHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class ICLRACERHeadMini(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

#        w = chainer.initializers.HeNormal()

        # Layers
        # Input should be 60 x 60 (n channels)
        # 1st layer: (60 - 8) / 4 + 1, 14 x 14 x 32
        # 2nd layer: (14 - 4)/2 + 1, 6 x 6 x 64
        # 3rd layer: (6 - 3)/1 + 1, 4 x 4 x 64 = 256
        
        layers = [
            L.Convolution2D(n_input_channels,
                            out_channels=32,
                            ksize=8,
                            stride=4,
#                            initialW=w,
                            initial_bias=bias),
            L.Convolution2D(32,
                            out_channels=64,
                            ksize=4,
                            stride=2,
#                            initialW=w,
                            initial_bias=bias),
            
            L.Convolution2D(64,
                            out_channels=64,
                            ksize=3,
                            stride=1,
#                            initialW=w,
                            initial_bias=bias),
            
            L.Linear(1024, n_output_channels,   # FIXME, check INPUT
#                     initialW=LeCunNormal(1e-3),
                     initial_bias=bias),
        ]

        super(ICLRACERHeadMini, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h

