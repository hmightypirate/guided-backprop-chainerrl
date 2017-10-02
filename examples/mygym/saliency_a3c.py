import logging
import chainer
from chainerrl.agents import a3c
from chainer import functions as F
import numpy as np

# Render imports
import gym
import skimage
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from chainerrl.misc.batch_states import batch_states


class SaliencyA3C(a3c.A3C):

    process_idx = None
    saved_attributes = ['model', 'optimizer']
    
    def __init__(self, model, optimizer, t_max, gamma, beta=1e-2,
                 process_idx=0, phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 keep_loss_scale_same=False,
                 normalize_grad_by_t_max=False,
                 use_average_reward=False, average_reward_tau=1e-2,
                 act_deterministically=False,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 batch_states=batch_states):
        
        super(SaliencyA3C, self).__init__(
            model=model,
            optimizer=optimizer,
            t_max=t_max,
            gamma=gamma,
            beta=beta,
            process_idx=process_idx,
            phi=phi,
            pi_loss_coef=pi_loss_coef,
            v_loss_coef=v_loss_coef,
            keep_loss_scale_same=keep_loss_scale_same,
            normalize_grad_by_t_max=normalize_grad_by_t_max,
            use_average_reward=use_average_reward,
            average_reward_tau=average_reward_tau,
            act_deterministically=act_deterministically,
            average_entropy_decay=average_entropy_decay,
            average_value_decay=average_value_decay,
            batch_states=batch_states
        )

        # Render agent input
        self.viewer_value = None
        self.viewer_logit = None
        self.render = True

    def act(self, obs):
        # Use the process-local model for acting
        if chainer.config.train:
            # Training
            with chainer.no_backprop_mode():
                statevar = self.batch_states([obs], np, self.phi)
                pout, _ = self.model.pi_and_v(statevar)
                
                if self.act_deterministically:
                    return pout.most_probable.data[0]
                else:
                    return pout.sample().data[0]
        else:
            # Testing
            self.model.cleargrads()

            statevar = chainer.Variable(self.batch_states(
                [obs], np, self.phi))
            pout, vout = self.model.pi_and_v(statevar)

            # Compute saliency maps
            saliency_map_logit = self._obtain_saliency_logit(pout, statevar)

            self.model.cleargrads()
            
            saliency_map_value = self._obtain_saliency_value(vout, statevar)

            # Compute saliency prob
            
            # render
            self._render(statevar.data[-1][-1], saliency_map_logit,
                         saliency_map_value)
            
            if self.act_deterministically:
                return pout.most_probable.data[0]
            else:
                return pout.sample().data[0]
        
    def _obtain_saliency_value(self, v, statevar):
        v.backward()

        inputgrad = statevar.grad

        # print "LINALGNORM ", np.linalg.norm(inputgrad[-1][-1], ord=2)
        # print "NPMAX ", np.max(inputgrad[-1][-1])
        
        # Pick last image
        return np.abs(inputgrad[-1][-1])

    def _obtain_saliency_logit(self, p, statevar):
        
        p_max_logit = F.max(p.logits)
        p_max_logit.backward()
        inputgrad = statevar.grad

        return np.abs(inputgrad[-1][-1])

    def _render(self, raw_img, saliency_logit, saliency_value, close=False):
        img = np.zeros((raw_img.shape[0], raw_img.shape[1], 3))
                       
        if close:
            if self.viewer_value is not None:
                self.viewer_value.close()
                self.viewer_value = None

            if self.viewer_logit is not None:
                self.viewer_logit.close()
                self.viewer_logit = None
            return

        # Pick last screen
        img[:, :, 0] = saliency_value * 200
        img[:, :, 1] = raw_img * 255
        img[:, :, 2] = raw_img * 255

        img = resize(img, (250, 250))

#        img = gray2rgb(raw_img*255)
        img = img.astype(np.uint8)
        
        from gym.envs.classic_control import rendering

        if self.viewer_value is None:
            self.viewer_value = rendering.SimpleImageViewer()
        self.viewer_value.imshow(img)

        # show logit
        # Pick last screen

        img = np.zeros((raw_img.shape[0], raw_img.shape[1], 3))
        img[:, :, 0] = raw_img * 255
        img[:, :, 1] = saliency_logit * 200
        img[:, :, 2] = raw_img * 255

        img = resize(img, (250, 250))

#        img = gray2rgb(raw_img*255)
        img = img.astype(np.uint8)

        if self.viewer_logit is None:
            self.viewer_logit = rendering.SimpleImageViewer()
        self.viewer_logit.imshow(img)

        
