# PeerPrediction

Python module for evaluation of subjective truthfulness of peer prediction mechanisms.

Dependencies for the module are:

* python >= 3.6

* numpy >= 1.13.3

* matplotlib >= 2.0.2

* pyjags >= 1.2.2

Additional dependencies for pyjags might include:

* JAGS >= 4.3.0

* gcc >= 4.8.1

## Contents:

* [class peerprediction.mechanisms.SubjectiveOA](#class-peerpredictionmechanismssubjectiveoareports-beliefs-samplingfalse-num_samp1000)

* [class peerprediction.mechanisms.SubjectiveSM](#class-peerpredictionmechanismssubjectivesmreports-beliefs-samplingfalse-num_samp1000)

* [class peerprediction.mechanisms.SubjectiveDG](#class-peerpredictionmechanismssubjectivedgreports-beliefs-samplingfalse-num_samp100000)

* [class peerprediction.aggregation.SpectralMetaLearner](#class-peerpredictionaggregationspectralmetalearner)

* [class peerprediction.aggregation.HGCM](#class-peerpredictionaggregationhgcmadapt1000-iterations1000-chains4-thin1-progress_bartrue-threads1)

* [class peerprediction.aggregation.TwoCoinModel](#class-peerpredictionaggregationtwocoinmodelepsilon001-iterations10000)

* [class peerprediction.aggregation.SubjectiveAggregator](#class-peerpredictionaggregationsubjectiveaggregatorreports-beliefs-agg-samplersimple-n_sampled_peers100-inner_loops1-kwargs)

* [class peerprediction.aggregation.AggregationMechanism](#class-peerpredictionaggregationaggregationmechanismreports-agg-kwargs)

* [module peerprediction.tools.Simplex](#module-peerpredictiontoolssimplex)

***

### class peerprediction.mechanisms.SubjectiveOA(reports, beliefs, sampling=False, num_samp=1000)

Performs evaluation of subjective truthfulness of Output Agreement mechanism.

<table style="width: 100%">
  <tr>
    <td rowspan="8" valign="top">Parameters:</td>
    <td><b>reports:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Report(s) of a single or multiple agents. Shape of the array should be N (number of agents) * M (number of tasks) for N >= 1 and M >= 1. Entry (i, j) shows report of agent i on task j. Each report can be either 0 or 1.</td>
  </tr>
  <tr>
    <td><b>beliefs:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Belief(s) of a single or multiple agents on other agents observing signal 1. Shape of the array should be equivalent to shape of reports. Entry (i, j) shows belief of agent i on task j. Each belief can be between 0 and 1.</td>
  </tr>
  <tr>
    <td><b>sampling:</b> boolean (default=False)</td>
  </tr>
  <tr>
    <td>Closed-form method used if sampling=False. Estimation method is used if sampling=True.</td>
  </tr>
  <tr>
    <td><b>num_samp:</b> integer (default=1000)</td>
  </tr>
  <tr>
    <td>Number of peers to sample if estimation method is used (sampling=True). Should be a positive number.</td>
  </tr>
  <tr>
    <td rowspan="6" valign="top">Attributes:</td>
    <td><b>truthfulness_:</b> numpy ndarray</td>
  </tr>
  <tr>
    <td>Shape of this array is N * M. Entry (i, j) = 1 if mechanism is truthful for agent i on task j, (i, j) = 0 if mechanism is untruthful and (i, j) = 0.5 if truthful and untruthful reporting produce equal rewards.</td>
  </tr>
  <tr>
    <td><b>payments_:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Shape of this array is N * M. Entry (i, j) shows payment to agent i for task j given her original report.</td>
  </tr>
  <tr>
    <td><b>payments_c_:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Shape of this array is N * M. Entry (i, j) shows payment to agent i for task j given her complementary (opposite) report.</td>
  </tr>
</table>

***

### class peerprediction.mechanisms.SubjectiveSM(reports, beliefs, sampling=False, num_samp=1000)

Performs evaluation of subjective truthfulness of Shadowing Method.

<table>
  <tr>
    <td rowspan="8" valign="top">Parameters:</td>
    <td><b>reports:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Report(s) of a single or multiple agents. Shape of the array should be N (number of agents) * M (number of tasks) for N >= 1 and M >= 2. Entry (i, j) shows report of agent i on task j. Each report can be either 0 or 1.</td>
  </tr>
  <tr>
    <td><b>beliefs:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Belief(s) of a single or multiple agents on other agents observing signal 1. Shape of the array should be equivalent to shape of reports. Entry (i, j) shows belief of agent i on task j. Each belief can be between 0 and 1.</td>
  </tr>
  <tr>
    <td><b>sampling:</b> boolean (default=False)</td>
  </tr>
  <tr>
    <td>Closed-form method used if sampling=False. Estimation method is used if sampling=True.</td>
  </tr>
  <tr>
    <td><b>num_samp:</b> integer (default=1000)</td>
  </tr>
  <tr>
    <td>Number of peers to sample if estimation method is used (sampling=True). Should be a positive number.</td>
  </tr>
  <tr>
    <td rowspan="6" valign="top">Attributes:</td>
    <td><b>truthfulness_:</b> numpy ndarray</td>
  </tr>
  <tr>
    <td>Shape of this array is N * M. Entry (i, j) = 1 if mechanism is truthful for agent i on task j, (i, j) = 0 if mechanism is untruthful and (i, j) = 0.5 if truthful and untruthful reporting produce equal rewards.</td>
  </tr>
  <tr>
    <td><b>payments_:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Shape of this array is N * M. Entry (i, j) shows payment to agent i for task j given her original report.</td>
  </tr>
  <tr>
    <td><b>payments_c_:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Shape of this array is N * M. Entry (i, j) shows payment to agent i for task j given her complementary (opposite) report.</td>
  </tr>
</table>

***

### class peerprediction.mechanisms.SubjectiveDG(reports, beliefs, sampling=False, num_samp=100000)

Performs evaluation of subjective truthfulness of Dasgupta-Ghosh mechanism with symmetric strategies.

<table>
  <tr>
    <td rowspan="8" valign="top">Parameters:</td>
    <td><b>reports:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Report(s) of a single or multiple agents. Shape of the array should be N (number of agents) * M (number of tasks) for N >= 1 and M >= 3. Entry (i, j) shows report of agent i on task j. Each report can be either 0 or 1.</td>
  </tr>
  <tr>
    <td><b>beliefs:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Belief(s) of a single or multiple agents on other agents observing signal 1. Shape of the array should be equivalent to shape of reports. Entry (i, j) shows belief of agent i on task j. Each belief can be between 0 and 1.</td>
  </tr>
  <tr>
    <td><b>sampling:</b> boolean (default=False)</td>
  </tr>
  <tr>
    <td>Closed-form method used if sampling=False. Estimation method is used if sampling=True.</td>
  </tr>
  <tr>
    <td><b>num_samp:</b> integer (default=100000)</td>
  </tr>
  <tr>
    <td>Number of peers to sample if estimation method is used (sampling=True). Should be a positive number.</td>
  </tr>
  <tr>
    <td rowspan="6" valign="top">Attributes:</td>
    <td><b>truthfulness_:</b> numpy ndarray</td>
  </tr>
  <tr>
    <td>Shape of this array is M. Entry i = 1 if mechanism is truthful for agent i, i = 0 if mechanism is untruthful and i = 0.5 if truthful and untruthful reporting produce equal rewards. Assumes agent adopts same symmetric strategy across all the tasks.</td>
  </tr>
  <tr>
    <td><b>payments_:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Shape of this array is M. Entry i shows payment to agent i for all tasks assuming agent adopts same symmetric strategy across all tasks given her original reports.</td>
  </tr>
  <tr>
    <td><b>payments_c_:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Shape of this array is M. Entry i shows payment to agent i for all tasks assuming agent adopts same symmetric strategy across all tasks given her complementary (opposite) reports.</td>
  </tr>
</table>

***

### class peerprediction.aggregation.SpectralMetaLearner()

Aggregation algorithm Spectral Meta-Learner. Main idea of aggregation utilizes eigendecomposition.

Methods:

* aggregate(reports): Start aggregation of reports of multiple agents into one ndarray of size M.

<table>
  <tr>
    <td rowspan="2" valign="top">Parameters:</td>
    <td><b>reports:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Report(s) of multiple agents. Shape of the array should be N (number of agents) * M (number of tasks) for N >= 2 and M >= 2. Entry (i, j) shows report of agent i on task j. Each report can be either 0 or 1.</td>
  </tr>
  <tr>
    <td>Returns:</td>
    <td><b>None</b></td>
  </tr>
</table>

* predict(): Returns ndarray of size M of aggregated reports.

<table>
  <tr>
    <td>Parameters:</td>
    <td><b>None</b></td>
  </tr>
  <tr>
    <td rowspan="2" valign="top">Returns:</td>
    <td><b>pred:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Aggregated reports for each of the tasks.</td>
  </tr>
</table>

***

### class peerprediction.aggregation.HGCM(adapt=1000, iterations=1000, chains=4, thin=1, progress_bar=True, threads=1)

Aggregation algorithm Hierarchical General Condorcet Model. Relies on Gibbs Sampling (pyjags) for performing aggregation.

<table>
  <tr>
    <td rowspan="12" valign="top">Parameters:</td>
    <td><b>adapt:</b> non-negative integer (default=1000)</td>
  </tr>
  <tr>
    <td>Number of adaptation (burnout) iterations for Gibbs Sampler.</td>
  </tr>
  <tr>
    <td><b>iterations:</b> non-negative integer (default=1000)</td>
  </tr>
  <tr>
    <td>Number of sampling iterations for Gibbs Sampler.</td>
  </tr>
  <tr>
    <td><b>chains:</b> non-negative integer (default=4)</td>
  </tr>
  <tr>
    <td>Number of chains for Gibbs Sampler</td>
  </tr>
  <tr>
    <td><b>thin:</b> non-negative integer (default=1)</td>
  </tr>
  <tr>
    <td>Thinning interval for Gibbs Sampler.</td>
  </tr>
  <tr>
    <td><b>progress_bar:</b> boolean (default=True)</td>
  </tr>
  <tr>
    <td>Shows sampler’s progress bar if True.</td>
  </tr>
  <tr>
    <td><b>threads:</b> non-negative integer (default=1)</td>
  </tr>
  <tr>
    <td>Number of threads used for sampling. One thread can sample from at most one chain.</td>
  </tr>
</table>

Methods:

* aggregate(reports): Start aggregation of reports of multiple agents into one ndarray of size M.

<table>
  <tr>
    <td rowspan="2" valign="top">Parameters:</td>
    <td><b>reports:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Report(s) of multiple agents. Shape of the array should be N (number of agents) * M (number of tasks) for N >= 2 and M >= 2. Entry (i, j) shows report of agent i on task j. Each report can be either 0 or 1.</td>
  </tr>
  <tr>
    <td>Returns:</td>
    <td><b>None</b></td>
  </tr>
</table>

* predict(): Returns ndarray of size M of aggregated reports.

<table>
  <tr>
    <td>Parameters:</td>
    <td><b>None</b></td>
  </tr>
  <tr>
    <td rowspan="2" valign="top">Returns:</td>
    <td><b>pred:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Aggregated reports for each of the tasks.</td>
  </tr>
</table>

***

### class peerprediction.aggregation.TwoCoinModel(epsilon=0.01, iterations=10000)

Aggregation algorithm TwoCoinModel. Employs classical EM algorithm for maximizing likelihood of aggregated reports.

<table>
  <tr>
    <td rowspan="4" valign="top">Parameters:</td>
    <td><b>epsilon:</b> non-negative float (default=0.01)</td>
  </tr>
  <tr>
    <td>Convergence threshold for Expectation-Maximization algorithm. EM algorithm converges if 1-norm of <i>diff</i> is less than epsilon. <i>diff</i> is defined as <i>(mu<sub>t</sub> - mu<sub>t-1</sub>)</i>, where <i>mu<sub>t</sub></i> is a vector of posterior estimates at current iteration, and <i>mu<sub>t-1</sub></i> is a vector of posterior estimates at previous iteration.</td>
  </tr>
  <tr>
    <td><b>iterations:</b> non-negative integer (default=10000)</td>
  </tr>
  <tr>
    <td>Number of iterations for EM algorithm. Algorithm may stop if convergence criterion epsilon is met before total number of iterations is done. If convergence criterion is not met after performing all iterations, warning is displayed.</td>
  </tr>

Methods:

* aggregate(reports): Start aggregation of reports of multiple agents into one ndarray of size M.

<table>
  <tr>
    <td rowspan="2" valign="top">Parameters:</td>
    <td><b>reports:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Report(s) of multiple agents. Shape of the array should be N (number of agents) * M (number of tasks) for N >= 2 and M >= 2. Entry (i, j) shows report of agent i on task j. Each report can be either 0 or 1.</td>
  </tr>
  <tr>
    <td>Returns:</td>
    <td><b>None</b></td>
  </tr>
</table>

* predict(): Returns ndarray of size M of aggregated reports.

<table>
  <tr>
    <td>Parameters:</td>
    <td><b>None</b></td>
  </tr>
  <tr>
    <td rowspan="2" valign="top">Returns:</td>
    <td><b>pred:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Aggregated reports for each of the tasks.</td>
  </tr>
</table>

***

### class peerprediction.aggregation.SubjectiveAggregator(reports, beliefs, agg, sampler='simple', n_sampled_peers=100, inner_loops=1, **kwargs)

Evaluates subjective truthfulness of peer prediction mechanisms using aggregation algorithms.

<table>
  <tr>
    <td rowspan="14" valign="top">Parameters:</td>
    <td><b>reports:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Report(s) of a single or multiple agents. Shape of the array should be N (number of agents) * M (number of tasks) for N >= 1 and M >= 2. Entry (i, j) shows report of agent i on task j. Each report can be either 0 or 1.</td>
  </tr>
  <tr>
    <td><b>beliefs:</b> numpy ndarray of floats</td>
  </tr>
  <tr>
    <td>Belief(s) of a single or multiple agents on other agents observing signal 1. Shape of the array should be equivalent to shape of reports. Entry (i, j) shows belief of agent i on task j. Each belief can be between 0 and 1.</td>
  </tr>
  <tr>
    <td><b>agg:</b> python class performing aggregation</td>
  </tr>
  <tr>
    <td>This is the aggregation algorithm that will be used for evaluation. Possible choices are 3 previous classes of aggregation algorithms mentioned in this API. It is also possible to use any other aggregation algorithm, that has methods aggregate(reports) and predict().</td>
  </tr>
  <tr>
    <td><b>sampler:</b> string, can be ’simple’ or ’smart_v1’ (default=’simple’)</td>
  </tr>
  <tr>
    <td>Method of sampling virtual peers for current agent according to her beliefs. If ’simple’ then peers will be sampled i.i.d. from Bernoulli process. If ’smart_v1’ then original dataset of reports will be used to emulate peers. In ’smart_v1’ we expand the original dataset and flip as few reports as possible such that they converge to current agent’s belief.</td>
  </tr>
  <tr>
    <td><b>n_sampled_peers:</b> non-negative integer (default=100)</td>
  </tr>
  <tr>
    <td>Number of virtual peers to sample for each of the actual agents.</td>
  </tr>
  <tr>
    <td><b>inner_loops:</b> non-negative integer (default=1)</td>
  </tr>
  <tr>
    <td>Number of times sampling procedure should be repeated. If 1, then values against which current agent will be scored come from single aggregation of sampled peers. For any inner loops > 1 there will be more than one aggregation. In this case final values against which agent will be scored are defined by majority vote over all performed aggregations.</td>
  </tr>
  <tr>
    <td><b>kwargs:</b> python dictionary</td>
  </tr>
  <tr>
    <td>Additional arguments required by agg class.</td>
  </tr>
</table>

Methods:

* run_computation(num processes=1): Initiates evaluation of subjective truthfulness.

<table>
  <tr>
    <td rowspan="2" valign="top">Parameters:</td>
    <td><b>num_processes:</b> non-negative integer (default=1)</td>
  </tr>
  <tr>
    <td>Number of processes to use while running the computation. In order to speed-up computation recommended value for this parameter should be equal to number of available cores.
</td>
  </tr>
  <tr>
    <td>Returns:</td>
    <td><b>None</b></td>
</table>

* show_subjective_truthfulness(): Shows information on subjective truthfulness of the mechanism.

<table>
  <tr>
    <td>Parameters:</td>
    <td><b>None</b></td>
  </tr>
  <tr>
    <td rowspan="2" valign="top">Returns:</td>
    <td><b>truthfulness:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Shape of this array is N * M. Entry (i, j) = 1 if mechanism is truthful for agent i on task j or 0 otherwise.</td>
  </tr>
</table>

***

### class peerprediction.aggregation.AggregationMechanism(reports, agg, **kwargs)

Peer prediction mechanism using aggregation algorithms. Does not evaluate subjective truthfulness or require beliefs of the agents. Produces actual payments for each of the agents on each of the tasks.

<table>
  <tr>
    <td rowspan="6" valign="top">Parameters:</td>
    <td><b>reports:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Reports of multiple agents. Shape of the array should be N (number of agents) * M (number of tasks) for N >= 3 and M >= 2. Entry (i, j) shows report of agent i on task j. Each report can be either 0 or 1.</td>
  </tr>
  <tr>
    <td><b>agg:</b> python class performing aggregation</td>
  </tr>
  <tr>
    <td>This is the aggregation algorithm that will be used for evaluation. Possible choices are 3 previous classes of aggregation algorithms mentioned in this API documentation. It is also possible to use any other aggregation algorithm, that has methods aggregate(reports) and predict().</td>
  </tr>
  <tr>
    <td><b>kwargs:</b> python dictionary</td>
  </tr>
  <tr>
    <td>Additional arguments required by agg class.</td>
  </tr>
</table>

Methods:

* produce_payments(): Produces rewards for each agent on each task.

<table>
  <tr>
    <td>Parameters:</td>
    <td><b>None</b></td>
  </tr>
  <tr>
    <td>Returns:</td>
    <td><b>None</b></td>
  </tr>
</table>

* show_payments(): Returns array containing payments.

<table>
  <tr>
    <td>Parameters:</td>
    <td><b>None</b></td>
  </tr>
  <tr>
    <td rowspan="2" valign="top">Returns:</td>
    <td><b>payments:</b> numpy ndarray of integers</td>
  </tr>
  <tr>
    <td>Shape of this array is N * M. Entry (i, j) corresponds to reward of agent i on task j.</td>
  </tr>
</table>

***

### module peerprediction.tools.Simplex

Methods:

* draw simplex(pay_mat, save=False, savename=’simplex’): Draws probability simplex with belief model constraints based on payment matrix corresponding to some mechanism. Works for environments with tree signals.

<table>
  <tr>
    <td rowspan="6" valign="top">Parameters:</td>
    <td><b>pay_mat:</b> numpy ndarray of shape 3 * 3</td>
  </tr>
  <tr>
    <td>Payment matrix, specifying the mechanism. Entry (i, j) shows which reward current agent gets if she reports i and her peer agent reports j.</td>
  </tr>
  <tr>
    <td><b>save:</b> boolean (default=False)</td>
  </tr>
  <tr>
    <td>Shows probability simplex if False. Saves the image in *.png format if True.</td>
  </tr>
  <tr>
    <td><b>savename:</b> string (default=’simplex’)</td>
  </tr>
  <tr>
    <td>Savename for *.png file (only if save=True).</td>
  </tr>
  <tr>
    <td>Returns:</td>
    <td><b>None</b></td>
  </tr>
</table>
