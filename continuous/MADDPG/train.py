import numpy as np
import tensorflow as tf
import time
from continuous.MADDPG.maddpg import MADDPGAgent
from make_env import make_env

train_parameters = {
    "max-episode-len": 25,
    "num-episodes": 100000,
    }
model_parameters = {
    "buffer_size": 100000,
    "lr_actor": 1.0e-2,
    "lr_critic": 1.0e-2,
    "sigma": 0.15,
    "gamma": 0.95,
    "batch-size": 1024,
    "max_replay_buffer_len": 10240,
    "tau": 0.01
}


def train():
    # 初始化环境
    env = make_env(scenario_name="Predator_prey_4v4")

    # 初始化MADDPGAgent
    maddpgagents = [MADDPGAgent(name="agent_"+str(i),
                                obs_shape=env.observation_space,
                                act_space=env.action_space,
                                agent_num=env.n,
                                agent_index=i,
                                parameters=model_parameters,
                                create_summary_writer=True)
                    for i in range(env.n)]

    print('Starting training...')
    episode = 0
    epoch = 0
    train_step = 0
    while episode < train_parameters["num-episodes"]:
        t_start = time.time()
        obs_n = env.reset()
        episode_steps = 0
        episode_rewards = [0.0]  # sum of rewards for all agents
        group_rewards = [[0.0], [0.0]]
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward

        for agent in maddpgagents:
            agent.action_noise.reset()

        while episode_steps < train_parameters["max-episode-len"]:
            T = time.time()
            # get action
            action_n = [agent.action(obs, evaluation=False) for agent, obs in zip(maddpgagents, obs_n)]
            # print(time.time()-T)
            # environment step
            # print(action_n)
            new_obs_n, rew_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            # collect experience
            for i, agent in enumerate(maddpgagents):
                agent.experience(obs_n[i], action_n[i], [rew_n[i]], new_obs_n[i], [done_n[i]])

            # 记录reward数据
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            group_rewards[0][-1] += sum(rew_n[:4])
            group_rewards[1][-1] += sum(rew_n[4:])

            # update all trainers
            if epoch % 50 == 0:
                for agent in maddpgagents:
                    agent.preupdate()
                train_able = False
                for i in range(env.n):
                    if len(maddpgagents[i].replay_buffer) > maddpgagents[i].max_replay_buffer_len:  # replay buffer is not large enough
                        train_able = True
                        maddpgagents[i].update(maddpgagents, train_step)
                if train_able:
                    train_step += 1

            if done:
                break

            obs_n = new_obs_n
            episode_steps += 1
            epoch += 1

            # print(time.time()-T)
        if episode % 50 == 0:
            print("episode {}: {} total reward, {} epoch, {} episode_steps, {} train_steps, {} time".format(
                episode, agent_rewards, epoch, episode_steps, train_step, time.time()-t_start))

        episode += 1

        for i, maddpgagent in enumerate(maddpgagents):
            with maddpgagent.summary_writer.as_default():
                tf.summary.scalar('Main/total_reward', episode_rewards[-1], step=episode)
                tf.summary.scalar('Main/Agent_reward', agent_rewards[i][-1], step=episode)
                if i < 4:
                    tf.summary.scalar('Main/group_reward', group_rewards[0][-1], step=episode)
                else:
                    tf.summary.scalar('Main/group_reward', group_rewards[1][-1], step=episode)
            maddpgagent.summary_writer.flush()

        if episode % 50 == 0:
            # 保存模型参数
            for agent in maddpgagents:
                agent.save_model("models/maddpg_actor_agent_", "models/maddpg_critic_agent_")

    # 关闭summary，回收资源
    for i, maddpgagent in enumerate(maddpgagents):
        maddpgagent.summary_writer.close()
    env.close()
    # 保存模型参数
    for agent in maddpgagents:
        agent.save_model("models/maddpg_actor_agent_", "models/maddpg_critic_agent_")


def inference(episode_num=100, max_episode_steps=100):
    # 初始化环境
    env = make_env(scenario_name="Predator_prey_4v4")

    # 初始化MADDPGAgent
    maddpgagents = [MADDPGAgent(name="agent" + str(i),
                                obs_shape=env.observation_space,
                                act_space=env.action_space,
                                agent_num=env.n,
                                agent_index=i,
                                parameters=model_parameters,
                                create_summary_writer=False)
                    for i in range(env.n)]
    for agent in maddpgagents:
        agent.load_actor(a_fn="models/maddpg_actor_agent_")

    episode = 0
    while episode < episode_num:
        rewards = np.zeros(env.n, dtype=np.float32)
        cur_state = env.reset()
        step = 0
        while step < max_episode_steps:
            # get action
            action_n = [agent.action(obs, evaluation=True) for agent, obs in zip(maddpgagents, cur_state)]
            # print(action_n)
            # environment step
            next_state, reward, done, _ = env.step(action_n)
            env.render()
            time.sleep(0.03)
            cur_state = next_state
            rewards += np.asarray(reward, dtype=np.float32)
            step += 1
        episode += 1
        print("episode {}: {} total reward, {} steps".format(episode, rewards, step))
    env.close()


if __name__ == '__main__':
    train()
    inference()
