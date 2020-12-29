from CalculationKernel import CalculationKernel, HiddenPrints
from model.LatencyModel import LatencyModel
from agent.ExchangeAgent import ExchangeAgent
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from agent.ValueAgent import ValueAgent
from util.order import LimitOrder
from util import util
from util import OrderBook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import time
import multiprocessing as mp
from gurobipy import *
from scipy.stats import kstwobign


class CalibrateValueAgent:
    def __init__(self, symbol="AAPL", fund_path=None, freq="min", num_agents=100, lambda_a=1e-12,
                 starting_cash=1e7, historical_date="20190628", seed=None):
        if fund_path == None:
            fund_path = "../data/mid_prices/fundamental_{}_{}.bz2".format(symbol, historical_date)
        try:
            self.fundamental_series = pd.read_pickle(fund_path, compression="bz2")  # read historical data
            self.fundamental_series.fillna(method="ffill", inplace=True)
            self.fundamental_returns = np.log(1 + self.fundamental_series.pct_change()).dropna().values[:, 0]
        except FileNotFoundError:
            raise FileNotFoundError("Path of fundamental prices does not exist.")

        self.symbol = symbol  # only support one symbol
        self.freq = freq  # calibration freq
        self.num_agents = num_agents  # number of value agents
        self.lambda_a = lambda_a  # arrival rate of agents
        self.starting_cash = starting_cash  # starting cash of agents
        self.midnight = pd.to_datetime(historical_date)  # historical date

        # Calibrate OU process
        self.r_bar, self.kappa, self.sigma_s = CalibrateValueAgent.calibrateOU(self.fundamental_series, freq)

        self.seed = seed

        # What is the earliest available time for an agent to act during the
        # simulation?
        self.kernelStartTime = self.midnight

        # When should the Kernel shut down?  (This should be after market close.)
        # Here we go for 5 PM the same day.
        self.kernelStopTime = self.midnight + pd.to_timedelta('17:00:00')

        # This will configure the kernel with a default computation delay
        # (time penalty) for each agent's wakeup and recvMsg.  An agent
        # can change this at any time for itself.  (nanoseconds)
        self.defaultComputationDelay = 1000000000  # one second

        print("Calibrate freq: {}".format(self.freq))
        logprint("Calibrate freq: {}\n".format(self.freq))
        print("Configuration seed: {}".format(seed))
        logprint("Configuration seed: {}\n".format(seed))
        logprint("Computer cores: {}\n".format(mp.cpu_count()))
        print("Calibrated OU parameters: r_bar = {}, kappa = {}, sigma_s = {}".format(self.r_bar, self.kappa,
                                                                                      self.sigma_s))
        logprint("Calibrated OU parameters: r_bar = {}, kappa = {}, sigma_s = {}\n".format(self.r_bar, self.kappa,
                                                                                           self.sigma_s))

    def calibrateModelDRO(self, sigma_n_grid, batch_size=16, parallel=True):
        n1 = len(self.fundamental_returns)
        fundamental_returns_sorted = np.sort(self.fundamental_returns)
        fund_quantile = np.searchsorted(fundamental_returns_sorted, self.fundamental_returns, side='right') / n1
        quantile_dict = dict()
        for sigma_n in sigma_n_grid:
            time_start = time.time()
            dist_sim = self.generateReturnDistribution(sigma_n, batch_size, parallel)
            dist_sim = np.sort(dist_sim)
            n2 = len(dist_sim)
            quantile_dict[sigma_n] = np.searchsorted(dist_sim, self.fundamental_returns, side='right') / n2
            time_end = time.time()
            print("sigma_n {} finished with total time {}, loss {}.".format(sigma_n, time_end - time_start, np.max(
                quantile_dict[sigma_n] - fund_quantile)))
            logprint("sigma_n {} finished with total time {}, loss {}.\n".format(sigma_n, time_end - time_start, np.max(
                quantile_dict[sigma_n] - fund_quantile)))

        print("Quantile calculation finished. Start optimization")
        m = Model("DRO")

        q = m.addVar(vtype=GRB.CONTINUOUS, name='q')
        W = dict()
        W_sum = 0
        quantile_avg = dict()
        m.addConstr(q >= 0, name="postive_q")
        for i in range(len(sigma_n_grid)):
            W[i] = m.addVar(vtype=GRB.CONTINUOUS, name='W' + str(i))
            m.addConstr(W[i] >= 0, "postive_W" + str(i))
            W_sum += W[i]
            for j in range(len(fund_quantile)):
                if j in quantile_avg:
                    quantile_avg[j] += W[i] * quantile_dict[sigma_n_grid[i]][j]
                else:
                    quantile_avg[j] = W[i] * quantile_dict[sigma_n_grid[i]][j]

        m.addConstr(W_sum == 1, name="sum_prob")
        for j in range(len(fund_quantile)):
            m.addConstr(fund_quantile[j] - q / np.sqrt(n1 * n2 / (n1 + n2)) <= quantile_avg[j], name="qCons1_" + str(j))
            m.addConstr(fund_quantile[j] + q / np.sqrt(n1 * n2 / (n1 + n2)) >= quantile_avg[j], name="qCons2_" + str(j))

        m.setObjective(q, GRB.MINIMIZE)
        m.optimize()
        print("Optimization finished.")

        W_optimal = [W[i].x for i in W]
        plt.plot(sigma_n_grid, W_optimal)
        plt.xlabel("$\sigma_n^2$")
        plt.ylabel("Weights")
        plt.savefig(log_file + "_weights.png")
        plt.show()

        return m.objVal, kstwobign.sf(m.objVal), W_optimal

    def calibrateModelGS(self, sigma_n_grid, batch_size=16, parallel=True):
        loss_dict = dict()
        for sigma_n in sigma_n_grid:
            time_start = time.time()
            loss_dict[sigma_n] = self.evaluateLoss(sigma_n, batch_size, parallel)
            time_end = time.time()
            print("sigma_n {} finished with total time {}, loss {}.".format(sigma_n, time_end - time_start,
                                                                            loss_dict[sigma_n]))
        sigma_n, _ = self.plot_score_curve(loss_dict)
        return {self.symbol: {'r_bar': self.r_bar, 'kappa': self.kappa, 'fund_var_sigma_s': self.sigma_s,
                              'noise_var_sigma_n': sigma_n}}

    def generateReturnDistribution(self, sigma_n, batch_size=16, parallel=True):
        if not parallel:
            dist_sim = np.array([])
            for i in range(batch_size):
                mid_prices = self.generateMidPrices(sigma_n)
                mid_prices["return"] = np.log(1 + mid_prices["price"].pct_change())
                mid_prices.dropna(inplace=True)
                dist_sim = np.hstack([mid_prices["return"].values, dist_sim])
        else:
            dist_sim = np.array([])
            num_cores = int(mp.cpu_count())
            iter = batch_size // num_cores
            for i in range(iter):
                pool = mp.Pool(num_cores)
                results = [pool.apply_async(self.generateMidPrices, args=(sigma_n,)) for j in range(num_cores)]
                for p in results:
                    mid_prices = p.get()
                    mid_prices["return"] = np.log(1 + mid_prices["price"].pct_change())
                    mid_prices.dropna(inplace=True)
                    # print(mid_prices.head(30))
                    dist_sim = np.hstack([mid_prices["return"].values, dist_sim])
                pool.close()
                pool.join()

        z1 = len(self.fundamental_returns[self.fundamental_returns != 0.]) / len(self.fundamental_returns)
        z2 = len(dist_sim[dist_sim != 0.]) / len(dist_sim)
        print(self.fundamental_returns.shape, dist_sim.shape)
        logprint("{},{}\n".format(self.fundamental_returns.shape, dist_sim.shape))
        print(z1, z2)
        logprint("{},{}\n".format(z1, z2))

        return dist_sim

    def evaluateLoss(self, sigma_n, batch_size=16, parallel=True):
        dist_sim = self.generateReturnDistribution(sigma_n, batch_size, parallel)
        return CalibrateValueAgent.KSDistance(self.fundamental_returns, dist_sim)

    def generateMidPrices(self, sigma_n):  # Set random seed
        if not self.seed:
            self.seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
        np.random.seed(self.seed)

        # Note: sigma_s is no longer used by the agents or the fundamental (for sparse discrete simulation).
        symbols = {
            self.symbol: {'r_bar': self.r_bar, 'kappa': self.kappa, 'agent_kappa': 1e-15, 'sigma_s': 0.,
                          'fund_vol': self.sigma_s, 'megashock_lambda_a': 1e-15, 'megashock_mean': 0.,
                          'megashock_var': 1e-15, "random_state": np.random.RandomState(
                    seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

        util.silent_mode = True
        LimitOrder.silent_mode = True
        OrderBook.tqdm_used = False

        kernel = CalculationKernel("Calculation Kernel", random_state=np.random.RandomState(
            seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))

        ### Configure the agents.  When conducting "agent of change" experiments, the
        ### new agents should be added at the END only.
        agent_count = 0
        agents = []
        agent_types = []

        # Let's open the exchange at 9:30 AM.
        mkt_open = self.midnight + pd.to_timedelta('09:30:00')

        # And close it at 4:00 PM.
        mkt_close = self.midnight + pd.to_timedelta('16:00:00')

        # Configure an appropriate oracle for all traded stocks.
        # All agents requiring the same type of Oracle will use the same oracle instance.
        oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

        # Create the exchange.
        num_exchanges = 1
        agents.extend([ExchangeAgent(j, "Exchange Agent {}".format(j), "ExchangeAgent", mkt_open, mkt_close,
                                     [s for s in symbols], log_orders=False, book_freq=self.freq, pipeline_delay=0,
                                     computation_delay=0, stream_history=10, random_state=np.random.RandomState(
                seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                       for j in range(agent_count, agent_count + num_exchanges)])
        agent_types.extend(["ExchangeAgent" for j in range(num_exchanges)])
        agent_count += num_exchanges

        symbol = self.symbol
        s = symbols[symbol]

        # Some value agents.
        agents.extend([ValueAgent(j, "Value Agent {}".format(j),
                                  "ValueAgent {}".format(j),
                                  random_state=np.random.RandomState(
                                      seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                  log_orders=False, symbol=symbol, starting_cash=self.starting_cash,
                                  sigma_n=sigma_n, r_bar=s['r_bar'], kappa=s['agent_kappa'],
                                  sigma_s=s['fund_vol'],
                                  lambda_a=self.lambda_a) for j in range(agent_count, agent_count + self.num_agents)])
        agent_types.extend(["ValueAgent {}".format(j) for j in range(self.num_agents)])
        agent_count += self.num_agents

        # Config the latency model

        latency = None
        noise = None
        latency_model = None

        USE_NEW_MODEL = True

        ### BEGIN OLD LATENCY ATTRIBUTE CONFIGURATION ###

        ### Configure a simple message latency matrix for the agents.  Each entry is the minimum
        ### nanosecond delay on communication [from][to] agent ID.

        # Square numpy array with dimensions equal to total agent count.  Most agents are handled
        # at init, drawn from a uniform distribution from:
        # Times Square (3.9 miles from NYSE, approx. 21 microseconds at the speed of light) to:
        # Pike Place Starbucks in Seattle, WA (2402 miles, approx. 13 ms at the speed of light).
        # Other agents can be explicitly set afterward (and the mirror half of the matrix is also).

        if not USE_NEW_MODEL:
            # This configures all agents to a starting latency as described above.
            latency = np.random.uniform(low=21000, high=13000000, size=(len(agent_types), len(agent_types)))

            # Overriding the latency for certain agent pairs happens below, as does forcing mirroring
            # of the matrix to be symmetric.
            for i, t1 in zip(range(latency.shape[0]), agent_types):
                for j, t2 in zip(range(latency.shape[1]), agent_types):
                    # Three cases for symmetric array.  Set latency when j > i, copy it when i > j, same agent when i == j.
                    if j > i:
                        # Presently, strategy agents shouldn't be talking to each other, so we set them to extremely high latency.
                        if (t1 == "ZeroIntelligenceAgent" and t2 == "ZeroIntelligenceAgent"):
                            latency[i, j] = 1000000000 * 60 * 60 * 24  # Twenty-four hours.
                    elif i > j:
                        # This "bottom" half of the matrix simply mirrors the top.
                        latency[i, j] = latency[j, i]
                    else:
                        # This is the same agent.  How long does it take to reach localhost?  In our data center, it actually
                        # takes about 20 microseconds.
                        latency[i, j] = 20000

            # Configure a simple latency noise model for the agents.
            # Index is ns extra delay, value is probability of this delay being applied.
            noise = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]

        ### END OLD LATENCY ATTRIBUTE CONFIGURATION ###

        ### BEGIN NEW LATENCY MODEL CONFIGURATION ###

        else:
            # Get a new-style cubic LatencyModel from the networking literature.
            pairwise = (len(agent_types), len(agent_types))

            model_args = {'connected': True,

                          # All in NYC.
                          'min_latency': np.random.uniform(low=21000, high=100000, size=pairwise),
                          'jitter': 0.3,
                          'jitter_clip': 0.05,
                          'jitter_unit': 5,
                          }

            latency_model = LatencyModel(latency_model='cubic', random_state=np.random.RandomState(
                seed=np.random.randint(low=0, high=2 ** 31)), kwargs=model_args)

        # Start the kernel running.
        with HiddenPrints():
            midprices = kernel.runner(agents=agents, startTime=self.kernelStartTime,
                                      stopTime=self.kernelStopTime,
                                      agentLatencyModel=latency_model,
                                      agentLatency=latency, latencyNoise=noise,
                                      defaultComputationDelay=self.defaultComputationDelay,
                                      oracle=oracle, log_dir=None, return_value={self.symbol: "midprices"})

        return midprices[self.symbol]

    def plot_distribution(self, sigma_n, batch_size=16, parallel=True, rule_out_zero=False):
        fundamental_returns = self.fundamental_returns
        dist_sim = self.generateReturnDistribution(sigma_n, batch_size, parallel)

        if rule_out_zero:
            fundamental_returns = fundamental_returns[fundamental_returns != 0.]
            dist_sim = dist_sim[dist_sim != 0.]

        fig, ax = plt.subplots(1, 2, figsize=(12, 9))
        sns.distplot(fundamental_returns, ax=ax[0], kde=False, bins=np.arange(-0.01, 0.01, 0.0005))
        ax[0].set_title("Fundamental price return")
        ax[0].set_xlim([-0.01, +0.01])
        sns.distplot(dist_sim, ax=ax[1], kde=False, bins=np.arange(-0.01, 0.01, 0.0005))
        ax[1].set_title("Simulated return with $\sigma_n^2$ = {}".format(sigma_n))
        ax[1].set_xlim([-0.01, +0.01])
        plt.show()

    @staticmethod
    def calibrateOU(fundamental_series, freq="s"):
        # The most reasonable and accurate calibration requires freq="ns", since our
        # trading system take these parameters in time unit "ns". However it will be
        # out of memory to use "ns", so we take default freq="s" as a approximation.

        if 'FundamentalValue' in fundamental_series.columns:
            fundamental_series = fundamental_series[['FundamentalValue']]

        fundamental_series = fundamental_series.resample(freq).ffill()
        fundamental_series["diff"] = -fundamental_series['FundamentalValue'].diff(-1)
        fundamental_series.dropna(inplace=True)

        # Regression
        slope, intercept, r_value, p_value, std_err = linregress(fundamental_series['FundamentalValue'],
                                                                 fundamental_series["diff"])

        # Adjust parameters to freq "ns"
        freq_multiplier = pd.Timedelta("1" + freq) / pd.Timedelta("1ns")
        r_bar = -intercept / slope
        kappa = -slope / freq_multiplier
        sigma_s = (1 - r_value ** 2) * fundamental_series["diff"].var() / freq_multiplier

        return r_bar, kappa, sigma_s

    @staticmethod
    def KSDistance(data1, data2, rule_out_zero=False):
        if rule_out_zero:
            data1 = data1[data1 != 0.]
            data2 = data2[data2 != 0.]
        data1 = np.sort(data1)
        data2 = np.sort(data2)
        n1 = data1.shape[0]
        n2 = data2.shape[0]
        data_all = np.concatenate([data1, data2])
        cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
        cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0 * n2)
        d = np.max(np.absolute(cdf1 - cdf2))
        return d

    @staticmethod
    def plot_score_curve(score_dict, title=""):
        param_list = []
        score_list = []
        for param in score_dict:
            param_list.append(param)
            score_list.append(score_dict[param])

        best_idx = np.argmin(score_list)

        plt.plot(param_list, score_list)
        plt.plot([param_list[best_idx]], [score_list[best_idx]], marker='.', markersize=5, color="red")

        plt.annotate('Best $d_{KS}$', xy=(param_list[best_idx], score_list[best_idx]), xycoords='data',
                     xytext=(0.8, 0.25), textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.01, headwidth=5),
                     horizontalalignment='right', verticalalignment='top',
                     )

        plt.title(title)
        plt.xlabel("$\sigma_n^2$")
        plt.ylabel("Loss $d_{KS}$")
        plt.show()

        return param_list[best_idx], score_list[best_idx]


log_file = "log"


def logprint(message):
    with open(log_file + ".txt", "a") as f:
        f.write(message)


if __name__ == "__main__":
    logprint("----------------------------------------------------\n")
    model = CalibrateValueAgent(symbol="AAPL", historical_date="20190603", lambda_a=1e-12, num_agents=100)

    # model.plot_distribution(sigma_n=4225, batch_size=32)
    # model.plot_distribution(sigma_n=6000, batch_size=32)

    sigma_n_grid = np.arange(5, 201, 5)
    sigma_n_grid = sigma_n_grid ** 2

    time_start = time.time()

    # param_dict = model.calibrateModelGS(sigma_n_grid, batch_size=8)
    # print(param_dict)

    result_dict = model.calibrateModelDRO(sigma_n_grid, batch_size=8)
    print(result_dict)
    logprint("{}\n".format(result_dict))

    time_end = time.time()
    print('totally time', time_end - time_start)
    logprint("totally time {}\n".format(time_end - time_start))
    logprint("----------------------------------------------------\n")

    # b_128_grid {'AAPL': {'r_bar': 17422.92347392952, 'kappa': 1.6968932483558243e-13, 'fund_var_sigma_s': 1.329102726489007e-08, 'noise_var_sigma_n': 4225}}
    # B_64_DRO l=1e-12 5-100 (2.13854335003672, 0.00021313428616781392, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5792141951837781, 0.4207858048162219, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # B_64_DRO l=2e-12 50-150 (2.0458293235535803, 0.00046304419184845717, [0.0, 0.0, 0.0, 0.366818873668189, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.633181126331811, 0.0, 0.0, 0.0, 0.0])
    # B_64_DRO l=3e-12 5-150 (1.9091424694165953, 0.0013651339042317647, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3895720345808983, 0.0, 0.0, 0.0, 0.1158245180690439, 0.0, 0.4946034473500578, 0.0, 0.0, 0.0])
    # B_64_DRO l=4e-12 5-200 (1.8171170974372886, 0.0027103380409097344, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.34760528781425004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2169846231369408, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4354100890488091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # B_64_DRO l=5e-12 5-200 (1.6709905878651525, 0.007511861594051844, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8553052037412611, 0.0, 0.0, 0.0, 0.09624938772657188, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04844540853216711, 0.0, 0.0])
    # B_64_DRO l=6e-12 5-200 (1.5587925392302957, 0.015506102138205387, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5421611689837513, 0.0, 0.0, 0.04799071058505739, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40984812043119123, 0.0, 0.0, 0.0, 0.0])
