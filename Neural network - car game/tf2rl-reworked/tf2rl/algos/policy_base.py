import numpy as np
import tensorflow as tf


class Policy(tf.keras.Model):
    def __init__(
            self,
            name,
            memory_capacity,
            update_interval=1,
            batch_size=256,
            discount=0.99,
            max_grad=10.,
            n_epoch=1,
            gpu=0):
        super().__init__()
        self.policy_name = name
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.discount = discount
        self.n_epoch = n_epoch
        self.max_grad = max_grad
        self.memory_capacity = memory_capacity
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

    def get_action(self, observation, test=False):
        raise NotImplementedError



class OffPolicyAgent(Policy):
    """
    Base class for off-policy agents
    """

    def __init__(
            self,
            memory_capacity,
            **kwargs):
        super().__init__(memory_capacity=memory_capacity, **kwargs)
