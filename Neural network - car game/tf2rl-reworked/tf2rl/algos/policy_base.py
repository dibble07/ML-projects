# import tensorflow as tf

# class Policy(tf.keras.Model):
#     def __init__(self, max_grad=10., gpu=0):
#         super().__init__()
#         self.max_grad = max_grad
#         self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

# class OffPolicyAgent(Policy):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

import tensorflow as tf

class Policy(tf.keras.Model):
    def __init__(self):
        super().__init__()

class OffPolicyAgent(Policy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
