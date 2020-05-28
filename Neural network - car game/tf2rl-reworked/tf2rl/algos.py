import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

def update_target_variables(target_variables, source_variables, tau=1.0, use_locking=False, name="update_target_variables"):

    if not isinstance(tau, float):
        raise TypeError("Tau has wrong type (should be float) {}".format(tau))
    if not 0.0 < tau <= 1.0:
        raise ValueError("Invalid parameter tau {}".format(tau))
    if len(target_variables) != len(source_variables):
        raise ValueError("Number of target variables {} is not the same as "
                         "number of source variables {}".format(
                             len(target_variables), len(source_variables)))

    same_shape = all(trg.get_shape() == src.get_shape()
                     for trg, src in zip(target_variables, source_variables))
    if not same_shape:
        raise ValueError("Target variables don't have the same shape as source "
                         "variables.")

    def update_op(target_variable, source_variable, tau):
        if tau == 1.0:
            return target_variable.assign(source_variable, use_locking)
        else:
            return target_variable.assign(
                tau * source_variable + (1.0 - tau) * target_variable, use_locking)

    update_ops = [update_op(target_var, source_var, tau)
                  for target_var, source_var
                  in zip(target_variables, source_variables)]
    return tf.group(name="update_all_variables", *update_ops)


def huber_loss(x, delta=1.):
    delta = tf.ones_like(x) * delta
    less_than_max = 0.5 * tf.square(x)
    greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
    return tf.where(
        tf.abs(x) <= delta,
        x=less_than_max,
        y=greater_than_max)


class Policy(tf.keras.Model):
    def __init__(self):
        super().__init__()


class OffPolicyAgent(Policy):
    def __init__(self):
        super().__init__()


class QFunc(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[32, 32], name="QFunc", enable_dueling_dqn=False, n_atoms=51):
        super().__init__(name=name)
        self._enable_dueling_dqn = enable_dueling_dqn

        self.l1 = Dense(units[0], name="L1", activation="relu")
        self.l2 = Dense(units[1], name="L2", activation="relu")
        self.l3 = Dense(action_dim, name="L3", activation="linear")

        if enable_dueling_dqn:
            self.l4 = Dense(1, name="L3", activation="linear")

        self(inputs=tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        if self._enable_dueling_dqn:
            advantages = self.l3(features)
            v_values = self.l4(features)
            q_values = v_values + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        else:
            q_values = self.l3(features)
        return q_values


class DQN(OffPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            q_func=None,
            name="DQN",
            lr=0.001,
            units=[32, 32],
            epsilon=0.1,
            epsilon_min=None,
            epsilon_decay_step=int(1e6),
            target_replace_interval=int(5e3),
            optimizer=None,
            enable_double_dqn=False,
            enable_dueling_dqn=False,
            discount=None,
            **kwargs):
        super().__init__(**kwargs)

        q_func = q_func if q_func is not None else QFunc
        # Define and initialize Q-function network
        kwargs_dqn = {
            "state_shape": state_shape,
            "action_dim": action_dim,
            "units": units,
            "enable_dueling_dqn": enable_dueling_dqn}
        self.q_func = q_func(**kwargs_dqn)
        self.q_func_target = q_func(**kwargs_dqn)
        self.q_func_optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam(learning_rate=lr)
        update_target_variables(self.q_func_target.weights,self.q_func.weights, tau=1.)

        self._action_dim = action_dim
        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

        # Set hyper-parameters
        if epsilon_min is not None:
            assert epsilon > epsilon_min
            self.epsilon_min = epsilon_min
            self.epsilon_decay_rate = (
                epsilon - epsilon_min) / epsilon_decay_step
            self.epsilon = max(epsilon, self.epsilon_min)
        else:
            self.epsilon = epsilon
            self.epsilon_min = epsilon
            self.epsilon_decay_rate = 0.
        self.target_replace_interval = target_replace_interval
        self.n_update = 0

        # DQN variants
        self._enable_double_dqn = enable_double_dqn

        # Oops
        self.discount=discount
        self.max_grad = 10

    def get_action(self, state, test=False, tensor=False):
        if not tensor:
            assert isinstance(state, np.ndarray)
        is_single_input = state.ndim == self._state_ndim

        if not test and np.random.rand() < self.epsilon:
            if is_single_input:
                action = np.random.randint(self._action_dim)
            else:
                action = np.array([np.random.randint(self._action_dim)
                                   for _ in range(state.shape[0])], dtype=np.int64)
            if tensor:
                return tf.convert_to_tensor(action)
            else:
                return action

        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_input else state
        action = self._get_action_body(tf.constant(state))
        if tensor:
            return action
        else:
            if is_single_input:
                return action.numpy()[0]
            else:
                return action.numpy()

    @tf.function
    def _get_action_body(self, state):
        q_values = self.q_func(state)
        return tf.argmax(q_values, axis=1)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        td_errors, q_func_loss = self._train_body(
            states, actions, next_states, rewards, done, weights)

        # tf.summary.scalar(name=self.policy_name +
        #                   "/q_func_Loss", data=q_func_loss)

        self.n_update += 1
        # Update target networks
        if self.n_update % self.target_replace_interval == 0:
            update_target_variables(
                self.q_func_target.weights, self.q_func.weights, tau=1.)

        # Update exploration rate
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_min)
        # self.epsilon = max(self.epsilon - self.epsilon_decay_rate * self.update_interval, self.epsilon_min)
        # tf.summary.scalar(name=self.policy_name+"/epsilon", data=self.epsilon)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        # with tf.device(self.device):
        with tf.GradientTape() as tape:
            td_errors = self._compute_td_error_body(
                states, actions, next_states, rewards, done)
            q_func_loss = tf.reduce_mean(
                huber_loss(td_errors,
                           delta=self.max_grad) * weights)

        q_func_grad = tape.gradient(
            q_func_loss, self.q_func.trainable_variables)
        self.q_func_optimizer.apply_gradients(
            zip(q_func_grad, self.q_func.trainable_variables))

        return td_errors, q_func_loss

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        if isinstance(actions, tf.Tensor):
            actions = tf.expand_dims(actions, axis=1)
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        return self._compute_td_error_body(
            states, actions, next_states, rewards, dones)

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        batch_size = states.shape[0]
        not_dones = 1. - tf.cast(dones, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.int32)
        # with tf.device(self.device):
        indices = tf.concat(
            values=[tf.expand_dims(tf.range(batch_size), axis=1),
                    actions], axis=1)
        current_Q = tf.expand_dims(
            tf.gather_nd(self.q_func(states), indices), axis=1)

        if self._enable_double_dqn:
            max_q_indexes = tf.argmax(self.q_func(next_states),
                                      axis=1, output_type=tf.int32)
            indices = tf.concat(
                values=[tf.expand_dims(tf.range(batch_size), axis=1),
                        tf.expand_dims(max_q_indexes, axis=1)], axis=1)
            target_Q = tf.expand_dims(
                tf.gather_nd(self.q_func_target(next_states), indices), axis=1)
            target_Q = rewards + not_dones * self.discount * target_Q
        else:
            target_Q = rewards + not_dones * self.discount * tf.reduce_max(
                self.q_func_target(next_states), keepdims=True, axis=1)
        target_Q = tf.stop_gradient(target_Q)
        td_errors = current_Q - target_Q
        return td_errors


class Actor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action, units=[400, 300], name="Actor"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(action_dim, name="L3")

        self.max_action = max_action

        # with tf.device("/cpu:0"):
        self(tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action


class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[400, 300], name="Critic"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(1, name="L3")

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        # with tf.device("/cpu:0"):
        self([dummy_state, dummy_action])

    def call(self, inputs):
        states, actions = inputs
        features = tf.concat([states, actions], axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        return features


class DDPG(OffPolicyAgent):
    def __init__(self, state_shape, action_dim, name="DDPG", max_action=1., lr_actor=0.001, lr_critic=0.001, actor_units=[400, 300], critic_units=[400, 300], sigma=0.1, tau=0.005, discount=None, **kwargs):
        super().__init__(**kwargs)

        # Define and initialize Actor network
        self.actor = Actor(state_shape, action_dim, max_action, actor_units)
        self.actor_target = Actor(state_shape, action_dim, max_action, actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        update_target_variables(self.actor_target.weights,self.actor.weights, tau=1.)

        # Define and initialize Critic network
        self.critic = Critic(state_shape, action_dim, critic_units)
        self.critic_target = Critic(state_shape, action_dim, critic_units)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        update_target_variables(self.critic_target.weights, self.critic.weights, tau=1.)

        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau

        # Oops
        self.discount=discount
        self.max_grad = 10
        # self.device = "/gpu:0"#"/cpu:0"

    def get_action(self, state, test=False, tensor=False):
        is_single_state = len(state.shape) == 1
        if not tensor:
            assert isinstance(state, np.ndarray)
        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(
            tf.constant(state), self.sigma * (1. - test),
            tf.constant(self.actor.max_action, dtype=tf.float32))
        if tensor:
            return action
        else:
            return action.numpy()[0] if is_single_state else action.numpy()

    @tf.function
    def _get_action_body(self, state, sigma, max_action):
        # with tf.device(self.device):
        action = self.actor(state)
        action += tf.random.normal(shape=action.shape, mean=0., stddev=sigma, dtype=tf.float32)
        return tf.clip_by_value(action, -max_action, max_action)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        actor_loss, critic_loss, td_errors = self._train_body(states, actions, next_states, rewards, done, weights)
        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        # with tf.device(self.device):
        with tf.GradientTape() as tape:
            td_errors = self._compute_td_error_body(
                states, actions, next_states, rewards, done)
            critic_loss = tf.reduce_mean(huber_loss(td_errors, delta=self.max_grad) * weights)

        critic_grad = tape.gradient(
            critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            next_action = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, next_action]))

        actor_grad = tape.gradient(
            actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # Update target networks
        update_target_variables(self.critic_target.weights, self.critic.weights, self.tau)
        update_target_variables(self.actor_target.weights, self.actor.weights, self.tau)

        return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        td_errors = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.abs(np.ravel(td_errors.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        # with tf.device(self.device):
        not_dones = 1. - dones
        target_Q = self.critic_target(
            [next_states, self.actor_target(next_states)])
        target_Q = rewards + (not_dones * self.discount * target_Q)
        target_Q = tf.stop_gradient(target_Q)
        current_Q = self.critic([states, actions])
        td_errors = target_Q - current_Q
        return td_errors
