import os
import torch.nn as nn
import torch
import torch.nn.functional as F

class DRQN(nn.Module):
    # 所有的agent共享同一网络，input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(DRQN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # GRUCell(input_size, hidden_size)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)  # GRUCell 的输入要求（current_input, last_hidden_state）
        q = self.fc2(h)  # h 是这一时刻的隐状态，用于输到下一时刻的RNN网络中去，q 是真实行为Q值输出
        return q, h



class QmixNN(nn.Module):
    def __init__(self, state_shape, args):
        super(QmixNN, self).__init__()
        self.args = args
        self.state_shape = state_shape
        """
                生成的hyper_w1需要是一个矩阵，但是torch NN的输出只能是向量；
                因此先生成一个（行*列）的向量，再reshape

                args.n_agents是使用hyper_w1作为参数的网络的输入维度， args.qmix_hidden_dim是网络隐藏层参数个数
                从而经过hyper_w1得到（经验条数， args.n_agents * args.qmix_hidden_dim)的矩阵
        """
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            # 经过hyper_w2 得到的（经验条数， 1）的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))

        else:
            self.hyper_w1 = nn.Linear(state_shape, args.n_agents * args.qmix_hidden_dim)
            # 经过hyper_w2 得到的（经验条数， 1）的矩阵
            self.hyper_w2 = nn.Linear(state_shape, args.qmix_hidden_dim)

        # hyper_b1 得到的（经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(state_shape, args.qmix_hidden_dim)
        # hyper_b1 得到的（经验条数， 1）的矩阵需要同样维度的hyper_b1
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, args.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.qmix_hidden_dim, 1)
                                      )

    """
        input:(batch_size, n_agents, qmix_hidden_dim)
        q_values:(episode_num, max_episode_Len, n_agents)
        states shape:(episode_num, max_episode_len, state_shape)
    """

    def forward(self, q_values, states):
        # print(args.state_shape)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents)
        states = states.reshape(-1, self.state_shape)  # (episode_num * max_episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states))  # abs输出绝对值，保证w非负
        b1 = self.hyper_b1(states)  # 不需要进行限制

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total


class QMIX:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape

        state_shape = self.obs_shape * self.n_agents

        # 根据参数决定DQRN的输入维度
        if self.args.last_action:
            input_shape += self.n_actions
        if self.args.reuse_networks:
            input_shape += self.n_agents

        # 神经网络
        self.eval_rnn = DRQN(input_shape, args)
        self.target_rnn = DRQN(input_shape, args)

        self.eval_qmix = QmixNN(state_shape, args)
        self.target_qmix = QmixNN(state_shape, args)

        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.eval_rnn.to(self.device)
            self.target_rnn.to(self.device)
            self.eval_qmix.to(self.device)
            self.target_qmix.to(self.device)
        else:
            self.device = torch.device("cpu")

        # 使target网络和eval网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix.load_state_dict(self.eval_qmix.state_dict())
        # 获取所有参数
        self.eval_parameters = list(self.eval_qmix.parameters()) + list(self.eval_rnn.parameters())
        # 获取优化器

        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，为每个agent维护一个eval_hidden
        # 学习时，为每个agent维护一个eval_hidden, target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print("Init qmix networks finished")

    def learn(self, batch, max_episode_len, train_step, episode=None):
        """
        在learn的时候，抽取到的数据是四维的，四个维度分别为
        1——第几个episode
        2——episode中第几个transition
        3——第几个agent的数据
        4——具体obs维度。
        因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，
        然后一次给神经网络传入每个episode的同一个位置的transition
        :param batch: train data，obs: 四维（第几个episode，episode中的第几个transition，第几个agent，具体obs的维度）
        :param max_episode_len: max episode length
        :param train_step: step record for updating target network parameters
        :param episode:
        :return:
        """
        # 获得episode的数目
        episode_num = batch['o'].shape[0]
        # 初始化隐藏状态
        self.init_hidden(episode_num)

        # 把batch里的数据转化为tensor
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        o, o_next, u, r, terminated = batch['o'], batch['o_next'], batch['u'], \
                                      batch['r'], batch['terminated']

        # 得到每个agent当前与下个状态的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        o = o.to(self.device)
        u = u.to(self.device)
        r = r.to(self.device)
        o_next = o_next.to(self.device)
        terminated = terminated.to(self.device)


        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        # print("q_evals1 shape: ", q_evals.size()) #[batch_size, max_episode_len, n_agents, n_actions]
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q，取所有行为中最大的 Q 值
        q_targets = q_targets.max(dim=3)[0]
        # print("q_evals2 shape: ", q_evals.size()) # [batch_size, max_episode_len, n_agents]

        # qmix更新过程，evaluate网络输入的是每个agent选出来的行为的q值，target网络输入的是每个agent最大的q值，和DQN更新方式一样
        q_total_eval = self.eval_qmix(q_evals, o)
        q_total_target = self.target_qmix(q_targets, o_next)

        # 计算一步 qmix 的target
        targets = r + self.args.gamma * q_total_target * (1 - terminated)
        # 参数更新
        td_error = (q_total_eval - targets.detach())

        # L2的损失函数，不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (td_error ** 2).mean()

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # 在指定周期更新 target network 的参数
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix.load_state_dict(self.eval_qmix.state_dict())

    def init_hidden(self, episode_num):
        """
        为每个episode初始化一个eval_hidden,target_hidden
        :param episode_num:
        :return:
        """
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition in range(max_episode_len):
            # 为每个obs加上agent编号和last_action
            inputs, inputs_ = self.get_inputs(batch, transition)
            inputs = inputs.to(self.device)  # [batch_size*n_agents, obs_shape+n_agents+n_actions]
            inputs_ = inputs_.to(self.device)

            self.eval_hidden = self.eval_hidden.to(self.device)
            self.target_hidden = self.target_hidden.to(self.device)

            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_, self.target_hidden)

            # 形状变化，把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            # 添加transition信息
            q_evals.append(q_eval)
            q_targets.append(q_target)

        # 将 q_eval 和 q_target 列表中的max_episode_len个数组（episode_num, n_agents, n_actions)
        # 堆叠为(batch_size, max_episode_len, n_agents, n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，onehot_u要用到上一条故取出所有
        obs, obs_, u_onehot = batch['o'][:, transition_idx],\
                              batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        inputs, inputs_ = [], []
        inputs.append(obs)
        inputs_.append(obs_)
        # 经验池的大小
        episode_num = obs.shape[0]

        # obs添上一个动作，agent编号
        if self.args.last_action:
            # 如果是第一条经验，就让前一个动作为0向量
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx-1])
            inputs_.append(u_onehot[:, transition_idx])
        if self.args.reuse_networks:
            """
            因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量即可，
            比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。
            而agent_0的数据正好在第0行，那么需要加的agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            """
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # 把batch_size, n_agents的agents的obs拼接起来
        # 因为这里所有的所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        # (batch_size, n_agents, n_actions) ->形状为(batch_size*n_agents, n_actions)
        inputs = torch.cat([x.reshape(episode_num*self.n_agents, -1) for x in inputs], dim=1)
        inputs_ = torch.cat([x.reshape(episode_num*self.n_agents, -1) for x in inputs_], dim=1)

        return inputs, inputs_

    def get_params(self):
        return {'eval_rnn': self.eval_rnn.state_dict(),
                'eval_qmix': self.eval_qmix.state_dict()}

    def load_params(self, params_dict):
        # Get parameters from save_dict
        self.eval_rnn.load_state_dict(params_dict['eval_rnn'])
        self.eval_qmix.load_state_dict(params_dict['eval_qmix'])
        # Copy the eval networks to target networks
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix.load_state_dict(self.eval_qmix.state_dict())
