import os
import time
import pyjags
import warnings
import numpy as np
import multiprocessing as mp
import peerprediction.tools.PeerSampler as ps



###############################################
# Aggregation algorithm Spectral Meta-Learner #
###############################################

class SpectralMetaLearner(object):

    # Method initializing aggregating
    # parameters: reports - ndarray (dim=N*M or N >= 2, M >= 2))
    def aggregate(self, ls):
        labels = ls * 2 - 1
        self.labels = labels
        self.find_eig()

    def create_cov_mat(self):
        n_voters = self.labels.shape[0]
        n_tasks = self.labels.shape[1]

        voters_means = np.mean(self.labels, axis=1)
        centered_labels = self.labels - voters_means[:, None]

        normalizer = 1.0 / (n_tasks - 1)
        cov_mat = np.dot(centered_labels,
                  np.transpose(centered_labels)) * normalizer

        self.cov_mat_ = cov_mat

    def create_left(self, n_unk):
        n_eq = int((n_unk-1)*(n_unk)/2)
        mat = np.zeros((n_eq, n_unk))
        i = 0
        j = 1
        counter = 0

        while counter < n_eq:
            mat[counter, i] = 1
            mat[counter, j] = 1
            if j == n_unk-1:
                i += 1
                j = i+1
            else:
                j += 1
            counter += 1

        return mat

    def create_both(self):
        n_unk = self.cov_mat_.shape[0]
        n_eq = int((n_unk-1)*(n_unk)/2)
        left_side = self.create_left(n_unk)
        right_side = np.zeros(n_eq)

        counter = 0

        while counter < n_eq:
            idx = np.where(left_side[counter, :] == 1)[0]
            right_side[counter] = self.cov_mat_[idx[0], idx[1]]
            counter += 1

        return left_side, right_side

    def solve_linear_system(self):
        left_side, right_side = self.create_both()
        selector = right_side != 0
        new_l = left_side[selector, :]
        new_r = np.log(np.abs(right_side[selector]))

        pseudo_inv = np.linalg.pinv(new_l)
        ols_solution = np.dot(pseudo_inv, new_r)

        return ols_solution

    def generate_rank1_mat(self):
        diag = self.solve_linear_system()
        n = self.cov_mat_.shape[0]
        rank1_mat = np.copy(self.cov_mat_)

        self.overflow_ = False

        for i in range(n):
            diagonal_element = np.exp(2*diag[i])
            if np.isinf(diagonal_element):
                self.overflow_ = True
                warnings.warn("Warning: One of the diagonal elements of rank-1 \
                matrix overflowed. Original covariance matrix will be used instead \
                for eigendecompostition")
                break
            else:
                rank1_mat[i, i] = diagonal_element

        self.rank1_mat_ = rank1_mat

    def find_eig(self):
        self.create_cov_mat()
        self.generate_rank1_mat()

        if self.overflow_:
            w, v = np.linalg.eig(self.cov_mat_)
        else:
            w, v = np.linalg.eig(self.rank1_mat_)

        idx = np.argmax(w)
        largest_eig = v[:, idx]

        if np.mean(largest_eig) < 0:
            largest_eig = -largest_eig

        self.eig_ = largest_eig

    # Shows aggregated reports
    # Returs: recovered_ground_truth_ - ndarray (dim=M)
    def predict(self):
        pred = (np.dot(np.transpose(self.labels), self.eig_) > 0).astype(int)
        return pred

##############################################################
# Aggregation algorithm Hierarchical General Condorcet Model #
##############################################################

class HGCM(object):

    # Constructor
    # parameters: adapt - number of adapting iterations for Gibbs Sampler
    #             iterations - number of sampling iterations for Gibbs Sampler
    #             chains - number of chains for Gibbs Sampler
    #             thin - thinning interval for Gibbs Sampler
    #             progress_bar - shows progress bar if True
    #             threads - number of threads to use
    def __init__(self, adapt=1000, iterations=1000, chains=4, thin=1, progress_bar=True, threads=1):

        self.adapt = adapt
        self.iterations = iterations
        self.chains = chains
        self.thin = thin
        self.progress_bar = progress_bar
        self.threads = threads

    # Method initializing aggregating
    # parameters: reports - ndarray (dim=N*M or N >= 2, M >= 2))
    def aggregate(self, reports):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.abspath(os.path.join(script_path, os.pardir))
        path = os.path.join(script_dir, 'hgcm_model.jags')

        n = reports.shape[0]
        m = reports.shape[1]
        Xtheta = np.transpose(np.array([np.ones(n)]))
        Xg = np.transpose(np.array([np.ones(n)]))
        Xdelta = np.transpose(np.array([np.ones(m)]))

        model = pyjags.Model(file=path, data=dict(Y=reports, n=n, m=m,
                             nrofdeltacov=1, nrofgcov=1, nrofthetacov=1,
                             Xtheta=Xtheta, Xg=Xg, Xdelta=Xdelta), chains=self.chains,
                             adapt=self.adapt, progress_bar=self.progress_bar, threads=self.threads)

        self.run_sampling(model)

    def run_sampling(self, model):
        samples = model.sample(self.iterations, vars=['Z'], thin=self.thin)
        self.recovered_ground_truth_ = (np.mean(np.mean(samples['Z'], axis=2), axis=1) > 0.5).astype(int)

    # Shows aggregated reports
    # Returs: recovered_ground_truth_ - ndarray (dim=M)
    def predict(self):
        return self.recovered_ground_truth_

########################################
# Aggregation algorithm Two-coin Model #
########################################

class TwoCoinModel(object):

    # Constructor
    # parameters: epsilon - convergence threshold
    #             iterations - number of iterations of EM algorithm to do.
    #                          Can stop earlier if convergence criterion met
    def __init__(self, epsilon=0.01, iterations=10000):
        self.epsilon = epsilon
        self.iterations = iterations
        self.iterations_done = 0

    # Method initializing aggregating
    # parameters: reports - ndarray (dim=N*M or N >= 2, M >= 2))
    def aggregate(self, reports):

        myus = np.mean(reports, axis=0)

        for i in range(self.iterations):
            alphas, betas, p = self.update_parameters(reports, myus)
            new_myus = self.update_myus(reports, alphas, betas, p)

            diff = np.sum(np.abs(myus - new_myus))
            myus = new_myus

            if diff < self.epsilon:
                break

            self.iterations_done += 1

        if self.iterations_done == self.iterations and diff >= self.epsilon:
            warnings.warn("Warning: EM algorithm did not converge")

        self.myus_ = myus

    # Shows aggregated reports
    # Returs: recovered_ground_truth_ - ndarray (dim=M)
    def predict(self):
        pred = (self.myus_ > 0.5).astype(int)
        return pred

    def update_parameters(self, reports, myus):
        alphas = np.dot(reports, myus) / np.sum(myus)
        betas = np.dot(1 - reports, 1 - myus) / np.sum(1 - myus)
        p = np.mean(myus)

        return alphas, betas, p

    def update_myus(self, reports, alphas, betas, p):
        a_temp = (alphas ** np.transpose(reports)) * \
        ((1 - alphas) ** np.transpose(1 - reports))
        a = np.prod(np.transpose(a_temp), axis=0)

        b_temp = (betas ** np.transpose(1 - reports)) * \
        ((1 - betas) ** np.transpose(reports))
        b = np.prod(np.transpose(b_temp), axis=0)

        myus_numer = a * p
        myus_denom = a * p + b * (1 - p)
        myus = myus_numer / myus_denom

        return myus

#########################################################################################
# Class performing evaluation of subjective truthfulness of aggregation-based mechanism #
#########################################################################################

class SubjectiveAggregator(object):

    # Constructor
    # parameters: reports - ndarray (dim=N*M or M, M >= 2)
    #             beliefs - ndarray (dim=N*M or M, M >= 2)
    #             agg - aggregation algorithm class, should have methods aggregate() and predict()
    #             sampler - sampling method to use. Can be 'simple' for Bernoulli sampling or 'smart_v1'
    #                       for smart sampling using original dataset
    #             n_sampled_peers - number of peers to sample when using sampling
    #             inner_loops - number of times sampling and aggregation should be repeated
    #             kwargs - additional arguments for aggregation algorithm agg
    def __init__(self, reports, beliefs, agg, sampler='simple', n_sampled_peers=100, inner_loops=1, **kwargs):

        if len(reports.shape) == 1:
            reports = np.array([reports])

        if len(beliefs.shape) == 1:
            beliefs = np.array([beliefs])

        if reports.shape != beliefs.shape:
            print('Shapes of reports and beliefs do not match.')

        self.reports = reports
        self.beliefs = beliefs
        self.agg = agg
        self.sampler = sampler
        self.n_sampled_peers = n_sampled_peers
        self.inner_loops = inner_loops
        self.kwargs = kwargs

    def sample_agents(self, beliefs):
        np.random.seed()

        if self.sampler == 'simple':
            sampled_agents = ps.sample_peers(beliefs, self.n_sampled_peers)
        elif self.sampler == 'smart_v1':
            sampled_agents = ps.smart1(beliefs, self.n_sampled_peers, self.reports)

        return sampled_agents

    def run_for_single_agent(self, beliefs, assignment):

        all_pred = []

        current_belief = beliefs[assignment]

        for i in range(self.inner_loops):
            sampled_agents = self.sample_agents(current_belief)
            local_agg = self.agg(**self.kwargs)

            local_agg.aggregate(sampled_agents)
            pred = local_agg.predict()
            if self.inner_loops > 1:
                all_pred.append(pred)
            else:
                print('agent {} done'.format(assignment))
                return pred

        print('agent {} done'.format(assignment))
        return np.transpose(np.array(all_pred))

    def run_process(self, beliefs, assignments, output_queue):

        total_tasks = assignments.shape[0]

        for i in range(total_tasks):
            result = self.run_for_single_agent(beliefs, assignments[i])
            output_queue.put((assignments[i], result))

    # Initiates main computation.
    # parameters: num_processes - number of processors to use for concurrency
    def run_computation(self, num_processes=1):

        num_agents = self.reports.shape[0]
        to_be_done = np.arange(num_agents)

        results = []

        while to_be_done.shape[0] > 0:

            que = mp.Queue()
            proc_assignments = np.array_split(to_be_done, num_processes)

            processes = [mp.Process(target=self.run_process, args=(self.beliefs, proc_assignments[i], que)) for i in range(num_processes)]

            for p in processes:
                p.start()

            ### IMPORTANT, prevents deadlock
            liveprocs = list(processes)
            while liveprocs:
                try:
                    while 1:
                        results.append(que.get(False))
                except mp.queues.Empty:
                    pass

                # Give tasks a chance to put more data in
                time.sleep(0.5)

                if not que.empty():
                    continue
                liveprocs = [p for p in liveprocs if p.is_alive()]
            ### END IMPORTANT

            for p in processes:
                p.join()

            while not que.empty():
                results.append(que.get())

            agents_done = np.array([r[0] for r in results])
            agents_left = np.setdiff1d(to_be_done, agents_done)
            to_be_done = agents_left

        results.sort(key = lambda t: t[0])
        results = [r[1] for r in results]
        results = np.array(results)

        if self.inner_loops > 1:
            self.raw_aggregated_reports_ = np.transpose(results, axes=(2, 0, 1))
        else:
            self.raw_aggregated_reports_ = results

    # Outputs results of main computation
    # Returns: ndarray (dim=N*M or M), where entry (i, j) = 1 if
    #          mechanism is truthfull for agent i on task j, 0 otherwise.
    def show_subjective_truthfulness(self):
        truthfulness = (np.mean(self.reports == self.raw_aggregated_reports_, axis=0) \
                        > 0.5).astype(int)
        return truthfulness

#################################################################################
# This code allows using aggregation algorithms as mechanisms for paying agents #
#################################################################################

class AggregationMechanism(object):

    # Constructor
    # parameters: reports - ndarray (dim=N*M or N >= 2, M >= 2)
    #             agg - aggregation algorithm class, should have methods aggregate() and predict()
    #             kwargs - additional arguments for aggregation algorithm agg
    #             sampling - estimation method will be used if sampling=True
    #             num_samp - number of peers to sample if sampling=True
    def __init__(self, reports, agg, **kwargs):
        self.reports = reports
        self.agg = agg
        self.kwargs = kwargs

    # Method to initialize calculation of payments
    def produce_payments(self):
        n = self.reports.shape[0]
        m = self.reports.shape[1]

        payments = []

        for i in range(n):
            cur_rep = self.reports[i, :]
            other_reps = np.delete(self.reports, i, axis=0)

            local_agg = self.agg(**self.kwargs)
            local_agg.aggregate(other_reps)

            aggregated = local_agg.predict()

            cur_payment = (cur_rep == aggregated).astype(int)
            payments.append(cur_payment)

        self.payments = np.array(payments)

    # Shows payment matrix
    # Returns: payments_ - ndarray (dim=N*M), where entry (i, j) is reward of agent i
    #                      on task j
    def show_payments(self):
        return self.payments
