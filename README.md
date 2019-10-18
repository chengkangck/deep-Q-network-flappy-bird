# deep-Q-network-flappy-bird

#Overview
This project follows the description of the Deep Q Learning algorithm described in Playing Atari with Deep Reinforcement Learning [1]
and shows that this learning algorithm can be further generalized to the notorious Flappy Bird.
Deep Reinforcement learning is first proposed in this paper.Deep-Q network is proposed and achieved in this paper, which achieved in learning
Atrari game by image input.

Installation Dependencies:
Python 2.7 or 3
TensorFlow 1.7
pygame

What is Deep Q-Network?
It is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.

For those who are interested in deep reinforcement learning, I highly recommend to read the following post:

Demystifying Deep Reinforcement Learning

Deep Q-Network Algorithm
The pseudo-code for the Deep Q Learning algorithm, as given in [1], can be found below:

Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for

