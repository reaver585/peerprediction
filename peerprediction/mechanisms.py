import numpy as np
import peerprediction.tools.PeerSampler as ps

############################################################################
# This code performs evaluation of subjective truthfulness of OA mechanism #
############################################################################

class SubjectiveOA(object):

    # Constructor
    # parameters: reports - ndarray (dim=N*M or M)
    #             beliefs - ndarray (dim=N*M or M)
    #             sampling - estimation method will be used if sampling=True
    #             num_samp - number of peers to sample if sampling=True
    #
    # attributes: truthfulness_ - ndarray (dim=N*M or M), where entry (i, j) = 1 if
    #                             mechanism is truthful for agent i on task j,
    #                             (i, j) = 0 if mechanism is untruthful and
    #                             (i, j) = 0.5 if truthful and untruthful reporting are equivalent
    def __init__(self, reports, beliefs, sampling=False, num_samp=1000):
        # print('Subjective OA mechanism, sampling={}, num_samp={}\n'.format(sampling, num_samp))

        if len(reports.shape) == 1:
            reports = np.array([reports])

        if len(beliefs.shape) == 1:
            beliefs = np.array([beliefs])

        if reports.shape != beliefs.shape:
            print('Shapes of reports and beliefs do not match, implement exception!')

        num_agents = reports.shape[0]
        num_tasks = reports.shape[1]

        self.payments_ = np.zeros((num_agents, num_tasks))
        self.payments_c_ = np.zeros((num_agents, num_tasks))

        for i in range(num_agents):
            if sampling:
                peers = ps.sample_peers(beliefs[i, :], num_samp)
                # print(np.mean(peers, axis=0))
                self.payments_[i, :] = \
                self.estimated_mean_reward(reports[i, :], peers)

                self.payments_c_[i, :] = \
                self.estimated_mean_reward(1 - reports[i, :], peers)
            else:
                self.payments_[i, :] = \
                self.expected_reward(reports[i, :], beliefs[i, :])

                self.payments_c_[i, :] = \
                self.expected_reward(1 - reports[i, :], beliefs[i, :])

        self.strong_truthfulness_ = (self.payments_ > self.payments_c_).astype(int)
        self.weak_truthfulness_ = (self.payments_ >= self.payments_c_).astype(int)
        if sampling:
            self.truthfulness_ = self.strong_truthfulness_
        else:
            self.truthfulness_ = (self.strong_truthfulness_ + self.weak_truthfulness_) / 2

    @staticmethod
    def expected_reward(reports, beliefs):
        reward = beliefs * (reports == 1) + \
        (1 - beliefs) * (reports == 0)
        return reward

    @staticmethod
    def estimated_mean_reward(reports, peers):
        rewards = np.equal(reports, peers)
        reward = np.mean(rewards, axis=0)
        return reward

############################################################################
# This code performs evaluation of subjective truthfulness of SM mechanism #
############################################################################

class SubjectiveSM(object):

    # Constructor
    # parameters: reports - ndarray (dim=N*M or M, M >= 2)
    #             beliefs - ndarray (dim=N*M or M, M >= 2)
    #             sampling - estimation method will be used if sampling=True
    #             num_samp - number of peers to sample if sampling=True
    #
    # attributes: truthfulness_ - ndarray (dim=N*M or M), where entry (i, j) = 1 if
    #                             mechanism is truthfull for agent i on task j,
    #                             (i, j) = 0 if mechanism is untruthful and
    #                             (i, j) = 0.5 if truthful and untruthful reporting are equivalent
    def __init__(self, reports, beliefs, sampling=False, num_samp=1000):
        # print('Subjective SM mechanism, sampling={}, num_samp={}\n'.format(sampling, num_samp))

        if len(reports.shape) == 1:
            reports = np.array([reports])

        if len(beliefs.shape) == 1:
            beliefs = np.array([beliefs])

        if reports.shape != beliefs.shape:
            print('Shapes of reports and beliefs do not match, implement exception!')

        num_agents = reports.shape[0]
        num_tasks = reports.shape[1]

        self.payments_ = np.zeros((num_agents, num_tasks))
        self.payments_c_ = np.zeros((num_agents, num_tasks))
        self.priors = np.zeros((num_agents, num_tasks))

        for i in range(num_agents):
            if sampling:
                peers = ps.sample_peers(beliefs[i, :], num_samp)
                self.priors[i, :] = self.estimated_mean_priors(peers)
                sh_post = self.perform_shadowing(reports[i, :], self.priors[i, :])
                comp_post = self.perform_shadowing(1 - reports[i, :], self.priors[i, :])

                self.payments_[i, :] = \
                self.estimated_mean_reward(sh_post, peers)

                self.payments_c_[i, :] = \
                self.estimated_mean_reward(comp_post, peers)
            else:
                self.priors[i, :] = self.expected_priors(beliefs[i, :])
                sh_post = self.perform_shadowing(reports[i, :], self.priors[i, :])
                comp_post = self.perform_shadowing(1 - reports[i, :], self.priors[i, :])

                self.payments_[i, :] = \
                self.expected_reward(sh_post, beliefs[i, :])

                self.payments_c_[i, :] = \
                self.expected_reward(comp_post, beliefs[i, :])

        self.strong_truthfulness_ = (self.payments_ > self.payments_c_).astype(int)
        self.weak_truthfulness_ = (self.payments_ >= self.payments_c_).astype(int)
        if sampling:
            self.truthfulness_ = self.strong_truthfulness_
        else:
            self.truthfulness_ = (self.strong_truthfulness_ + self.weak_truthfulness_) / 2

    @staticmethod
    def expected_priors(beliefs):
        num_tasks = beliefs.shape[0]
        expanded = np.tile(beliefs, (num_tasks, 1))
        u = np.triu(expanded, 1)
        l = np.tril(expanded, -1)
        no_diag = u + l

        priors = np.mean(no_diag, axis=1) * (num_tasks / (num_tasks - 1))
        return priors

    @staticmethod
    def estimated_mean_priors(peers):
        num_tasks = peers.shape[1]
        num_samples = peers.shape[0]
        priors = np.zeros(num_tasks)
        for i in range(num_tasks):
            truncated = np.delete(peers, i, axis=1)
            prior = np.sum(truncated)/(num_samples*(num_tasks-1))
            priors[i] = prior
        return priors

    @staticmethod
    def perform_shadowing(reports, priors, d=0.1):
        d_mask = (2 * reports - 1) * d
        sh_posteriors = priors + d_mask
        return sh_posteriors

    @staticmethod
    def expected_reward(sh_posteriors, beliefs):
        reward = beliefs * (sh_posteriors * (2 - sh_posteriors)) + \
        (1 - beliefs) * (1 - sh_posteriors ** 2)
        return reward

    @staticmethod
    def estimated_mean_reward(sh_posteriors, peers):
        rewards = 1 - (sh_posteriors - peers) ** 2
        reward = np.mean(rewards, axis=0)
        return reward

############################################################################
# This code performs evaluation of subjective truthfulness of DG mechanism #
############################################################################

class SubjectiveDG(object):

    # Constructor
    # parameters: reports - ndarray (dim=N*M or M, M >= 3)
    #             beliefs - ndarray (dim=N*M or M, M >= 3)
    #             sampling - estimation method will be used if sampling=True
    #             num_samp - number of peers to sample if sampling=True
    #
    # attributes: truthfulness_ - ndarray (dim=N*M or M), where entry (i, j) = 1 if
    #                             mechanism is truthfull for agent i on task j,
    #                             (i, j) = 0 if mechanism is untruthful and
    #                             (i, j) = 0.5 if truthful and untruthful reporting are equivalent
    def __init__(self, reports, beliefs, sampling=False, num_samp=100000):
        # print('Subjective DG mechanism, sampling={}, num_samp={}\n'.format(sampling, num_samp))

        if len(reports.shape) == 1:
            reports = np.array([reports])

        if len(beliefs.shape) == 1:
            beliefs = np.array([beliefs])

        if reports.shape != beliefs.shape:
            print('Shapes of reports and beliefs do not match, implement exception!')

        num_agents = reports.shape[0]
        num_tasks = reports.shape[1]

        self.payments_ = np.zeros(num_agents)
        self.payments_c_ = np.zeros(num_agents)

        for i in range(num_agents):
            if sampling:
                self.payments_[i] = \
                self.estimated_mean_reward(reports[i, :], beliefs[i, :], num_samp)

                self.payments_c_[i] = \
                self.estimated_mean_reward(1 - reports[i, :], beliefs[i, :], num_samp)
            else:
                self.payments_[i] = \
                self.expected_reward(reports[i, :], beliefs[i, :])

                self.payments_c_[i] = \
                self.expected_reward(1 - reports[i, :], beliefs[i, :])

        self.strong_truthfulness_ = (self.payments_ > self.payments_c_).astype(int)
        self.weak_truthfulness_ = (self.payments_ >= self.payments_c_).astype(int)
        if sampling:
            self.truthfulness_ = self.strong_truthfulness_
        else:
            self.truthfulness_ = (self.strong_truthfulness_ + self.weak_truthfulness_) / 2

    @staticmethod
    def expected_reward_iteration(reports, beliefs, b_id, p1_id, p2_id):
        bonus = beliefs[b_id] * (reports[b_id] == 1) + \
        (1 - beliefs[b_id]) * (reports[b_id] == 0)
        penalty = beliefs[p2_id] * (reports[p1_id] == 1) + \
        (1 - beliefs[p2_id]) * (reports[p1_id] == 0)
        reward = bonus - penalty
        return reward

    @staticmethod
    def expected_reward(reports, beliefs):
        num_tasks = reports.shape[0]
        counter = 0
        accumulator = 0
        for i in range(num_tasks):
            for j in range(num_tasks):
                if j == i:
                    continue
                for k in range(num_tasks):
                    if k == i or k == j:
                        continue
                    accumulator += SubjectiveDG.expected_reward_iteration(reports,
                                                beliefs, i, j, k)
                    counter += 1
        accumulator /= counter
        return accumulator

    @staticmethod
    def estimated_reward_iteration(reports, beliefs, b_id, p1_id, p2_id):
        sample = np.random.binomial(1, p=beliefs)
        reward = 1 * (reports[b_id] == sample[b_id]) - \
                 1 * (reports[p1_id] == sample[p2_id])
        return reward

    @staticmethod
    def estimated_mean_reward(reports, beliefs, num_rep):
        num_tasks = reports.shape[0]
        accumulator = 0
        for i in range(num_rep):
            [b_id, p1_id, p2_id] = np.random.choice(np.arange(num_tasks),
                                             size=3, replace=False)
            accumulator += SubjectiveDG.estimated_reward_iteration(reports,
                                    beliefs, b_id, p1_id, p2_id)
        res = accumulator / num_rep
        return res
