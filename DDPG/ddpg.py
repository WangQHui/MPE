import random
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
import time
from DDPG.replay_buffer import ReplayBuffer

tf.keras.backend.set_floatx('float32')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Actor Model
def actor(state_shape, action_dim):
    state = Input(shape=state_shape)
    x = Dense(64, activation='relu')(state)
    x = Dense(32, activation='relu')(x)
    output = Dense(action_dim, activation='tanh')(x)  # 在网络末端，用tanh函数，把输出映射到[-1.0,1.0]之间
    model = Model(inputs=state, outputs=output)
    return model


# Critic Model
def critic(state_shape, action_dim):
    input = []
    input.append(Input(shape=state_shape))
    input.append(Input(shape=action_dim))
    concat = Concatenate(axis=-1)(input)
    x = Dense(128, activation='relu')(concat)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=input, outputs=output)
    return model




# update network parameters
def update_target_weights(model, target_model, tau):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


# DDPGAgent Class
class DDPGAgent():
    def __init__(self, name, obs_shape, act_space, agent_num, agent_index, parameters, create_summary_writer=False):
        self.name = name
        self.obs_shape = obs_shape
        self.act_space = act_space
        self.agent_num = agent_num
        self.agent_index = agent_index
        self.parameters = parameters

        self.actor = actor(state_shape=obs_shape[self.agent_index].shape[0], action_dim=act_space[self.agent_index].shape[0])
        self.critic = critic(state_shape=obs_shape[self.agent_index].shape[0], action_dim=act_space[self.agent_index].shape[0])
        # print(critic.summary())
        self.actor_target = actor(state_shape=obs_shape[self.agent_index].shape[0], action_dim=act_space[self.agent_index].shape[0])
        self.critic_target = critic(state_shape=obs_shape[self.agent_index].shape[0], action_dim=act_space[self.agent_index].shape[0])

        self.actor_optimizer = Adam(learning_rate=parameters['lr_actor'])
        self.critic_optimizer = Adam(learning_rate=parameters['lr_critic'])
        # self.critic.compile(loss="mean_squared_error", optimizer=self.critic_optimizer)

        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(act_space[self.agent_index].shape[0]), sigma=parameters['sigma'])

        self.update_networks_weight(tau=1)

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(parameters["buffer_size"])
        self.max_replay_buffer_len = parameters['max_replay_buffer_len']
        self.replay_sample_index = None

        # 为每一个agent构建tensorboard可视化训练过程
        if create_summary_writer:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/DDPG_Summary_' + current_time + "agent" + str(self.agent_index)
            self.summary_writer = tf.summary.create_file_writer(train_log_dir)

    def update_networks_weight(self, tau=1):
        update_target_weights(model=self.actor, target_model=self.actor_target, tau=tau)
        update_target_weights(model=self.critic, target_model=self.critic_target, tau=tau)

    def action(self, obs, evaluation=False):
        obs = np.expand_dims(obs, axis=0).astype(np.float32)
        mu, pi = self.__get_action(obs)
        a = mu if evaluation else pi
        return np.array(a[0])

    @tf.function
    def __get_action(self, obs):
        mu = self.actor(obs)
        noise = np.asarray([self.action_noise() for i in range(mu.shape[0])])
        # print(noise)
        pi = tf.clip_by_value(mu + noise, -1, 1)
        return mu, pi

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done[0]))

    # save_model("models/ddpg_actor_agent_", "models/ddpg_critic_agent_")
    def save_model(self, a_fn, c_fn):
        self.actor.save(a_fn+str(self.agent_index)+".h5")
        self.critic.save(c_fn+str(self.agent_index)+".h5")

    def load_actor(self, a_fn):
        self.actor.load_weights(a_fn+str(self.agent_index)+".h5")

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, train_step):
        T = time.time()
        self.replay_sample_index = self.replay_buffer.make_index(self.parameters['batch-size'])
        # collect replay sample from all agents
        obs_n = []
        act_n = []
        obs_next_n = []
        rew_n = []
        done_n = []
        act_next_n = []

        for i in range(self.agent_num):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(self.replay_sample_index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
            done_n.append(done)
            rew_n.append(rew)

        for i, agent in enumerate(agents):
            # print(i)
            # print(len(obs_next_n))
            target_mu = agents[i].actor_target(obs_next_n[i])
            # action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
            act_next_n.append(target_mu)
        # print(time.time()-T)
        '''
        obs_n = tf.convert_to_tensor(obs, dtype=tf.float32)
        act_n = tf.convert_to_tensor(act, dtype=tf.float32)
        rew_n = tf.convert_to_tensor(rew, dtype=tf.float32)
        obs_next_n = tf.convert_to_tensor(obs_next, dtype=tf.float32)
        done_n = tf.convert_to_tensor(done, dtype=tf.float32)
        '''
        summaries = self.train((obs_n, act_n, rew_n, obs_next_n, done_n, act_next_n))

        # if train_step % 10 == 0:  # only update every 100 steps
        self.update_networks_weight(tau=self.parameters["tau"])

        with self.summary_writer.as_default():
            for key in summaries.keys():
                tf.summary.scalar(key, summaries[key], step=train_step)
        self.summary_writer.flush()

    @tf.function
    def train(self, memories):
        obs_n, act_n, rew_n, obs_next_n, done_n, act_next_n = memories
        with tf.GradientTape() as tape:
            # compute Q_target
            # print(len(obs_next_n), len(act_next_n))
            q_target = self.critic_target([obs_next_n[self.agent_index]]+[act_next_n[self.agent_index]])
            dc_r = rew_n[self.agent_index] + self.parameters['gamma'] * q_target * (1 - done_n[self.agent_index])
            # compute Q
            q = self.critic([obs_n[self.agent_index]] + [act_n[self.agent_index]])
            td_error = q - dc_r
            q_loss = tf.reduce_mean(tf.square(td_error))
        q_grads = tape.gradient(q_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(q_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            mu = self.actor(obs_n[self.agent_index])
            act_n = [mu if i == self.agent_index else act for i, act in enumerate(act_n)]
            # act_n[self.agent_index] = mu
            q_actor = self.critic([obs_n[self.agent_index]] + [act_n[self.agent_index]])
            actor_loss = -tf.reduce_mean(q_actor)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # train q network
        # self.agent_num = 1
        # target_q = 0.0
        # for i in range(self.agent_num):
        #     target_act_next_n = [agents[i].actor['target_act'](obs_next_n[i]) for i in range(len(obs_next_n))]
        #     target_q_next = self.critic['target_q_values'](*(obs_next_n + target_act_next_n))
        #     target_q += rew_n[self.agent_index] + self.parameters['gamma'] * (1 - done_n[self.agent_index]) * target_q_next
        # target_q /= self.agent_num
        # q_loss = self.critic(*(obs_n + act_n + [target_q]))
        #
        # # train p network
        # actor_loss = self.actor(*(obs_n + act_n))

        # with tf.GradientTape() as tape:
        #     # compute Q_target
        #     q_target = [self.critic_target(obs_next+act_next)
        #     dc_r = rew + self.parameters['gamma'] * q_target * (1 - done)
        #     # compute Q
        #     q = self.critic(obs + act)
        #     td_error = q - dc_r
        #     q_loss = tf.reduce_mean(tf.square(td_error))
        # q_grads = tape.gradient(q_loss, self.critic.trainable_variables)
        # self.critic_optimizer.apply_gradients(zip(q_grads, self.critic.trainable_variables))
        #
        # with tf.GradientTape() as tape:
        #     mu = self.actor(obs)
        #     act = [mu if i == self.agent_index else act for i, act in enumerate(act)]
        #     # act_n[self.agent_index] = mu
        #     q_actor = self.critic(obs + act)
        #     actor_loss = -tf.reduce_mean(q_actor)
        # actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/q_loss', q_loss],
        ])

        return summaries


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
