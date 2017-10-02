# guided-backprop-chainerrl
Implementation of Guided Backpropagation in Chainer (ChainerRL)

## Introduction

This is an implementation of Guided Backpropagation over an A3C agent in ChainerRL (https://github.com/chainer/chainerrl). Guided Backpropagation is a technique to compute a saliency map of a network, so you actually can see which parts of the images are more interesting to the agent.

For more info, see for example (https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb)

## Dependencies

* gym: pip install gym
* gym['atari']: pip install gym['atari']
* chainerrl: pip install chainerrl 


## Training

To train a model try the following parameter

```
python examples/mygym/train_a3c_gym.py 4 --env Boxing-v0 --outdir outdirboxing --t-max 5 --lr 7e-4 --min_reward -500 --beta 1e-2 --reward-scale-factor 1.0  --rmsprop-epsilon 0.1 

```

Train could take a lot of time. Some games could produce good results after a few hours or just a day of training. 

## Testing:

To test a model just add --demo to the sequence of parameters and --load to load the best checkpoint. ChainerRL stores the checkpoint inside "outdir", in a folder with the date the training/test was launched. The best checkpoint is the one with the highest id number (a model is not saved until it improves the best model so far).

```
python examples/mygym/train_a3c_gym.py 4 --env Boxing-v0 --outdir outdirboxing --t-max 5 --lr 7e-4 --min_reward -500 --beta 1e-2 --reward-scale-factor 1.0  --rmsprop-epsilon 0.1 --demo --load <path to your checkpoint>
```

## Explanation of the files

* examples/mygym/train_a3c_gym.py: this is the launcher. You can see the whole list of parameters by typing straightaway (python examples/mygym/train_a3c_gym.py)
* examples/mygym/guided_relu.py: implements the guided relu. The guided relu behaves as a standard relu function during training and implements guided backpropagation during testing phase. For this to work properly the agent should turn off the training flag of chainer (chainer.config.train). See train_a3c_gym.py file.
* examples/mygym/env_gym_chainer.py: environment which links gym with chainer. It allows to use several frames as input (in a similar way as the ALE environment). That was the easiest way a found to use multiple frames as input with the gym framework in ChainerRL.
* examples/mygym/iclr_acer_link.py: implements the network (it is only the shared part between the actor and the critic). 
* examples/mygym/saliency_a3c.py: basically overrides the act function of the A3C agent in chainerRL to calculate/draw the saliency maps during testing. Otherwise it behaves as a normal a3c agent.
