#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import itertools as it
import pickle
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from skimage.color import rgb2gray
from lasagne.init import GlorotUniform,HeUniform, Constant
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify,sigmoid
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import theano
from theano import tensor
from tqdm import trange
from time import time, sleep

from PIL import Image,ImageGrab
import scipy.misc
import os
import win32api, win32con

'''=================global param==================='''
# Q-learning settings
g_learning_rate = 0.00025
g_discount_factor = 0.95
g_epochs = 100
g_learning_steps_per_epoch = 50000
g_replay_memory_size = 10000

# NN learning settings
g_batch_size = 64

# Training regime
g_test_episodes_per_epoch = 10

# Other parameters
g_frame_repeat = 2
g_resolution = (3, 228, 122)
g_episodes_to_watch = 10

# Configuration
g_game_box = (142,124,911,487) #(x1,y1,w,h)
g_red_blood_box = (91,31,312,16) # (x,y,w,h) relative to the game_box
g_blue_blood_box = (487,31,312,16) # (x,y,w,h) relative to the game_box
g_single_frame_time = 0.02
g_with_start_model = False
g_start_epoch = 0

# keys recorded by values
g_available_keys = (65,# A 
                    87,# W 
                    83,# S 
                    68,# D 
                    74,# J 
                    75,# K 
                    85,# U 
                    73)# I
'''=================================================='''
# Close and Open a new game
def mousePos(cord):
    win32api.SetCursorPos(cord)
    
def leftClick():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    sleep(.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
    
def doubleClick():
    leftClick()
    sleep(.1)
    leftClick()
    
def close_game():
    mousePos((1151,9))
    leftClick()
    
def open_game():
    mousePos((212,205))
    doubleClick()
    
# Get current game image
def get_screen():
    im = np.array(ImageGrab.grab((g_game_box[0],g_game_box[1],
                                  g_game_box[0]+g_game_box[2]-1,
                                  g_game_box[1]+g_game_box[3]-1)))
    return im

# Check player blood
def check_player_blood(img, box):
    img = img[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2]), 1]
    width, height = img.shape
    blood_points = 0
    for x in range(width):
        for y in range(height):
            if(img[x,y] > 128): # blood point
                blood_points += 1
    return blood_points*100/(width*height)

# Key push
def key_do(keys, frame_repeat):
    for key in keys:
        win32api.keybd_event(key,0,0,0)
    sleep(frame_repeat*g_single_frame_time)
    for key in keys:
        win32api.keybd_event(key,0,win32con.KEYEVENTF_KEYUP,0)
        
# Make actions
def make_action(action, frame_repeat):
    keys = []
    for i in range(len(action)):
        if(action[i]):
            keys = keys + [g_available_keys[i],]
    key_do(keys,frame_repeat)
    
# Converts and downsamples the input image
def preprocess(img):
    #img = rgb2gray(img)
    img = skimage.transform.resize(img, g_resolution)
    img = img.astype(np.float32)
    return img

class ReplayMemory:
    def __init__(self, capacity):
        state_shape = (capacity, g_resolution[0], g_resolution[1], g_resolution[2])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.bool_)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


def create_network(available_actions_count):
    # Create the input variables
    s1 = tensor.tensor4("States")
    a = tensor.vector("Actions", dtype="int32")
    q2 = tensor.vector("Next State's best Q-Value")
    r = tensor.vector("Rewards")
    isterminal = tensor.vector("IsTerminal", dtype="int8")

    # Create the input layer of the network.
    dqn = InputLayer(shape=[None, g_resolution[0], g_resolution[1], g_resolution[2]], input_var=s1)

    # Add 2 convolutional layers with Sigmoid or ReLu activation
    dqn = Conv2DLayer(dqn, num_filters=8, filter_size=[12, 12],
                      nonlinearity=sigmoid,
                      W=GlorotUniform(),
                      b=Constant(.1), stride=3)
    
    dqn = MaxPool2DLayer(dqn, pool_size=[2,2],stride=1)
    
    dqn = Conv2DLayer(dqn, num_filters=16, filter_size=[7, 7],
                      nonlinearity=sigmoid,
                      W=GlorotUniform(),
                      b=Constant(.1), stride=2)

    dqn = MaxPool2DLayer(dqn, pool_size=[2,2],stride=1)

    # Add a single fully-connected layer.
    dqn = DenseLayer(dqn, num_units=128, nonlinearity=sigmoid,
                      W=GlorotUniform(),
                     b=Constant(.1))

    # Add the output layer (also fully-connected).
    # (no nonlinearity as it is for approximating an arbitrary real function)
    dqn = DenseLayer(dqn, num_units=available_actions_count, nonlinearity=None)

    # Define the loss function
    q = get_output(dqn)
    # target differs from q only for the selected action. The following means:
    # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
    target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + g_discount_factor * (1 - isterminal) * q2)
    loss = squared_error(q, target_q).mean()

    # Update the parameters according to the computed gradient using RMSProp.
    params = get_all_params(dqn, trainable=True)
    updates = rmsprop(loss, params, g_learning_rate)

    # Compile the theano functions
    print "Compiling the network ..."
    function_learn = theano.function([s1, q2, a, r, isterminal], loss, updates=updates, name="learn_fn")
    function_get_q_values = theano.function([s1], q, name="eval_fn")
    function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
    print "Network compiled."

    def simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, g_resolution[0], g_resolution[1], g_resolution[2]]))

    # Returns Theano objects for the net and functions.
    return dqn, function_learn, function_get_q_values, simple_get_best_action


def learn_from_transition(s1, a, s2, s2_isterminal, r):
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, s2_isterminal, r)

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > g_batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(g_batch_size)
        q2 = np.max(get_q_values(s2), axis=1)
        # the value of q2 is ignored in learn if s2 is terminal
        learn(s1, q2, a, r, isterminal)
        
def get_reward(red_blood1, blue_blood1, red_blood2, blue_blood2):
    r = 0.0
    if(red_blood2 <= 0):
        r = -200.0
    if(blue_blood2 <= 0):
        r = 200.0
    return r - 1 + (red_blood2-red_blood1) + 3*(blue_blood1-blue_blood2)

def perform_learning_step(epoch, acum_reward):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * g_epochs  # 10% of learning time 2
        eps_decay_epochs = 0.6 * g_epochs  # 60% of learning time 12

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    unprocess_s1 = get_screen()
    s1 = preprocess(unprocess_s1)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
        
    make_action(actions[a], g_frame_repeat)

    # reward mechanism
    unprocess_s2 = get_screen()
    red_blood1 = check_player_blood(unprocess_s1, g_red_blood_box)
    blue_blood1 = check_player_blood(unprocess_s1, g_blue_blood_box)
    red_blood2 = check_player_blood(unprocess_s2, g_red_blood_box)
    blue_blood2 = check_player_blood(unprocess_s2, g_blue_blood_box)    

    reward = get_reward(red_blood1, blue_blood1, red_blood2, blue_blood2)
    acum_reward += reward

    isterminal = (red_blood2 <=0 or blue_blood2 <=0)
    s2 = preprocess(unprocess_s2) if not isterminal else None
    learn_from_transition(s1, a, s2, isterminal, reward)

    if(red_blood2 <= 0):
        return -1, acum_reward
    if(blue_blood2 <= 0):
        return 1, acum_reward
    else:
        return 0, acum_reward
    
# New an game environment.
def new_game():
    time_looping = time()
    while(True):
        image = get_screen()
        red_blood = check_player_blood(image, g_red_blood_box)
        blue_blood = check_player_blood(image, g_blue_blood_box)
        if(red_blood >= 100 and blue_blood >= 100):
            break

        time_wait = time() - time_looping
        if(time_wait > 10):
            close_game()
            open_game()
            mousePos((30,60))
            leftClick()
        
# Creates and initializes environment, this need you to open a new game
def initialize_game():
    print "Initializing game..."
    while(True):
        image = get_screen()
        red_blood = check_player_blood(image, g_red_blood_box)
        blue_blood = check_player_blood(image, g_blue_blood_box)
        if(red_blood >= 100 and blue_blood >= 100):
            break

# begin the game
initialize_game()

# Action = which buttons are pressed
n = len(g_available_keys)
actions = [list(a) for a in it.product([0, 1], repeat=n)]
    
# Create replay memory which will store the transitions
memory = ReplayMemory(g_replay_memory_size)

net, learn, get_q_values, get_best_action = create_network(len(actions))

# Load the network's parameters from a file
if(g_with_start_model):
    params = pickle.load(open('weights.dump', "r"))
    set_all_param_values(net, params)

print "Starting the training!"

time_start = time()
for epoch in range(g_start_epoch, g_epochs):
    print "\nEpoch %d\n-------" % (epoch + 1)
    train_episodes_finished = 0
    train_scores = []

    print "Training..."
    new_game()
    acum_reward = 0.0
    for learning_step in trange(g_learning_steps_per_epoch):
        game_state,acum_reward = perform_learning_step(epoch, acum_reward)
        if(game_state == 1 or game_state == -1):
            score = acum_reward
            train_scores.append(score)
            new_game()
            acum_reward = 0.0
            train_episodes_finished += 1

    print "%d training episodes played." % train_episodes_finished

    train_scores = np.array(train_scores)

    print "Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
        "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max()

##    print "\nTesting..."
##    test_episode = []
##    test_scores = []
##    for test_episode in trange(g_test_episodes_per_epoch):
##        new_game()
##        total_reward = 0.0
##        is_episode_finished = False
##        while not is_episode_finished:
##            unprocess_s1 = get_screen()
##            s1 = preprocess(unprocess_s1)
##            best_action_index = get_best_action(s1)
##            make_action(actions[best_action_index], g_frame_repeat)
##            unprocess_s2 = get_screen()
##            
##            red_blood1 = check_player_blood(unprocess_s1, g_red_blood_box)
##            blue_blood1 = check_player_blood(unprocess_s1, g_blue_blood_box)
##            red_blood2 = check_player_blood(unprocess_s2, g_red_blood_box)
##            blue_blood2 = check_player_blood(unprocess_s2, g_blue_blood_box)
##
##            total_reward += get_reward(red_blood1, blue_blood1, red_blood2, blue_blood2)
##            is_episode_finished = ((red_blood2 <= 0) or (blue_blood2 <= 0))
##            
##        test_scores.append(total_reward)
##
##    test_scores = np.array(test_scores)
##    print "Results: mean: %.1f±%.1f," % (
##        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max()

    print "Saving the network weigths..."
    pickle.dump(get_all_param_values(net), open('weights.dump', "w"))

    print "Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0)
    
print "======================================"
print "Training finished. It's time to watch!"

# Load the network's parameters from a file
params = pickle.load(open('weights.dump', "r"))
set_all_param_values(net, params)

for _ in range(g_episodes_to_watch):
    new_game()
    total_reward = 0.0
    while not is_episode_finished:
        unprocess_s1 = get_screen()
        s1 = preprocess(unprocess_s1)
        best_action_index = get_best_action(s1)
        make_action(actions[best_action_index], g_frame_repeat)
        unprocess_s2 = get_screen()
            
        red_blood1 = check_player_blood(unprocess_s1, g_red_blood_box)
        blue_blood1 = check_player_blood(unprocess_s1, g_blue_blood_box)
        red_blood2 = check_player_blood(unprocess_s2, g_red_blood_box)
        blue_blood2 = check_player_blood(unprocess_s2, g_blue_blood_box)

        total_reward += get_reward(red_blood1, blue_blood1, red_blood2, blue_blood2)
        is_episode_finished = (red_blood2 <= 0) or (blue_blood2 <= 0)

    # Sleep between episodes
    print "Total score: ", total_reward
