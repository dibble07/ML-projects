import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model

def update_target_variables(target_variables, source_variables, tau=1.0, use_locking=False):

    def update_op(target_variable, source_variable, tau):
        return target_variable.assign(tau * source_variable + (1.0 - tau) * target_variable, use_locking)

    update_ops = [update_op(target_var, source_var, tau) for target_var, source_var in zip(target_variables, source_variables)]
    return tf.group(name="update_all_variables", *update_ops)


def huber_loss(x, delta=1.):
    delta = tf.ones_like(x) * delta
    less_than_max = 0.5 * tf.square(x)
    greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
    return tf.where(
        tf.abs(x) <= delta,
        x=less_than_max,
        y=greater_than_max)


class QFunc(tf.keras.Model):
    def __init__(self, action_dim, units, enable_dueling_dqn=False):

        super().__init__()
        self._enable_dueling_dqn = enable_dueling_dqn

        self.l1 = Dense(units[0], name="L1", activation="relu")
        self.l2 = Dense(units[1], name="L2", activation="relu")
        self.l3 = Dense(action_dim, name="L3", activation="linear")

        if enable_dueling_dqn:
            self.l4 = Dense(1, name="L3", activation="linear")
        
    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        q_values_temp = self.l3(features)
        if self._enable_dueling_dqn:
            advantages = q_values_temp
            v_values = self.l4(features)
            q_values = v_values + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        else:
            q_values = q_values_temp
        return q_values


class DQN(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units, target_replace_interval, enable_dueling_dqn,
    discount, load_model, epsilon_init_step, epsilon_init, epsilon_final, epsilon_decay):
        super().__init__()

        # Assign variables
        self.alg_name = "DQN"
        self.epsilon_init_step = epsilon_init_step
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.discount=discount
        self.max_grad = 10
        self._action_dim = action_dim
        self._state_ndim = np.array(state_shape).shape[0]
        self.target_replace_interval = target_replace_interval
        self.n_update = 0

        # Define and initialize Q-function network
        kwargs_dqn = {"action_dim": action_dim, "units": units, "enable_dueling_dqn": enable_dueling_dqn}
        if load_model is None:
            self.q_func = QFunc(**kwargs_dqn)
            self.q_func_target = QFunc(**kwargs_dqn)
        else:
            self.q_func = tf.keras.models.load_model(f"DQN_{load_model}")
            self.q_func_target = tf.keras.models.load_model(f"DQN_{load_model}")
        self.q_func_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        update_target_variables(self.q_func_target.weights,self.q_func.weights, tau=1.)

    def get_action(self, state, test):

        is_single_input = state.ndim == self._state_ndim

        if not test and np.random.rand() < self.epsilon([self.n_update])[0]:
            if is_single_input:
                action = np.random.randint(self._action_dim)
            else:
                action = np.array([np.random.randint(self._action_dim) for _ in range(state.shape[0])], dtype=np.int64)
            return action

        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_input else state
        action = self._get_action_body(tf.constant(state))
        if is_single_input:
            return action.numpy()[0]
        else:
            return action.numpy()

    @tf.function
    def _get_action_body(self, state):
        q_values = self.q_func(state)
        return tf.argmax(q_values, axis=1)

    def train(self, states, actions, next_states, rewards, done, weights):
        if weights is None:
            weights = np.ones_like(rewards)
        td_errors, q_func_loss = self._train_body(states, actions, next_states, rewards, done, weights)

        self.n_update += 1
        # Update target networks
        if self.n_update % self.target_replace_interval == 0:
            update_target_variables(self.q_func_target.weights, self.q_func.weights, tau=1.)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.GradientTape() as tape:
            td_errors = self._compute_td_error_body(states, actions, next_states, rewards, done)
            q_func_loss = tf.reduce_mean(huber_loss(td_errors,delta=self.max_grad) * weights)

        q_func_grad = tape.gradient(q_func_loss, self.q_func.trainable_variables)
        self.q_func_optimizer.apply_gradients(zip(q_func_grad, self.q_func.trainable_variables))

        return td_errors, q_func_loss

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        return self._compute_td_error_body(states, actions, next_states, rewards, dones)

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        batch_size = states.shape[0]
        not_dones = 1. - tf.cast(dones, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.int32)
        indices = tf.concat(values=[tf.expand_dims(tf.range(batch_size), axis=1), actions], axis=1)
        current_Q = tf.expand_dims(tf.gather_nd(self.q_func(states), indices), axis=1)

        max_q_indexes = tf.argmax(self.q_func(next_states),axis=1, output_type=tf.int32)
        indices = tf.concat(values=[tf.expand_dims(tf.range(batch_size), axis=1),tf.expand_dims(max_q_indexes, axis=1)], axis=1)
        target_Q = tf.expand_dims(tf.gather_nd(self.q_func_target(next_states), indices), axis=1)
        target_Q = rewards + not_dones * self.discount * target_Q
        target_Q = tf.stop_gradient(target_Q)
        td_errors = current_Q - target_Q
        return td_errors

    def save_agent(self, filename):
        self.q_func.save(f"{self.alg_name}_{filename}")

    def epsilon(self, eps):
        epsilon = []
        eps_range = self.epsilon_init - self.epsilon_final
        for ep in eps:
            if ep < self.epsilon_init_step:
                epsilon.append(0)
            else:
                epsilon.append(eps_range*np.exp(-self.epsilon_decay*(ep-self.epsilon_init_step))+self.epsilon_final)
        return epsilon


class Actor(tf.keras.Model):
    def __init__(self, action_dim, units):
        super().__init__()
        self.l1 = Dense(units[0], name="L1", activation="relu")
        self.l2 = Dense(units[1], name="L2", activation="relu")
        self.l3 = Dense(action_dim, name="L3", activation="linear")

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        action = self.l3(features)
        return tf.clip_by_value(action, -1, 1)


class Critic(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.l1 = Dense(units[0], name="L1", activation="relu")
        self.l2 = Dense(units[1], name="L2", activation="relu")
        self.l3 = Dense(1, name="L3", activation="linear")

    def call(self, inputs):
        states, actions = inputs
        features = tf.concat([states, actions], axis=1)
        features = self.l1(features)
        features = self.l2(features)
        features = self.l3(features)
        return features


class DDPG(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action, actor_units, critic_units, sigma, tau, discount, load_model):
        super().__init__()

        # Assign variables
        self.alg_name = "DDPG"
        self.sigma = sigma
        self.tau = tau
        self.discount=discount
        self.max_grad = 10
        self.max_action = max_action

        # Define and initialize Actor network
        if load_model is None:
            self.actor = Actor(action_dim, actor_units)
        else:
            self.actor = tf.keras.models.load_model(f"{load_model}/actor")
        self.actor_target = Actor(action_dim, actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.tau)
        update_target_variables(self.actor_target.weights,self.actor.weights, tau=1.)

        # Define and initialize Critic network
        if load_model is None:
            self.critic = Critic(critic_units)
        else:
            self.critic = tf.keras.models.load_model(f"{load_model}/critic")
        self.critic_target = Critic(critic_units)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.tau)
        update_target_variables(self.critic_target.weights, self.critic.weights, tau=1.)


    def get_action(self, state, test):
        is_single_state = len(state.shape) == 1
        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_state else state
        action = self._get_action_body(tf.constant(state), self.sigma * (1. - test),tf.constant(self.max_action, dtype=tf.float32))
        return action.numpy()[0] if is_single_state else action.numpy()

    @tf.function
    def _get_action_body(self, state, sigma, max_action):
        action = self.actor(state)*max_action
        action += tf.random.normal(shape=action.shape, mean=0., stddev=sigma, dtype=tf.float32)
        return tf.clip_by_value(action, -max_action, max_action)

    def train(self, states, actions, next_states, rewards, done, weights):
        weights = np.ones_like(rewards) if weights is None else weights
        actor_loss, critic_loss, td_errors = self._train_body(states, actions, next_states, rewards, done, weights)
        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.GradientTape() as tape:
            td_errors = self._compute_td_error_body(states, actions, next_states, rewards, done)
            critic_loss = tf.reduce_mean(huber_loss(td_errors, delta=self.max_grad) * weights)

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            next_action = self.actor(states)*self.max_action
            actor_loss = -tf.reduce_mean(self.critic([states, next_action]))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # Update target networks
        update_target_variables(self.critic_target.weights, self.critic.weights, self.tau)
        update_target_variables(self.actor_target.weights, self.actor.weights, self.tau)

        return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones)
        return np.abs(np.ravel(td_errors.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        not_dones = 1. - dones
        target_Q = self.critic_target([next_states, self.actor_target(next_states)*self.max_action])
        target_Q = rewards + (not_dones * self.discount * target_Q)
        target_Q = tf.stop_gradient(target_Q)
        current_Q = self.critic([states, actions])
        td_errors = target_Q - current_Q
        return td_errors

    def save_agent(self, filename):
        self.actor.save(f"{self.alg_name}_{filename}/actor")
        self.critic.save(f"{self.alg_name}_{filename}/critic")