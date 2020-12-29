from CalculationKernel import CalculationKernel, HiddenPrints
from model.LatencyModel import LatencyModel
from agent.ExchangeAgent import ExchangeAgent
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from agent.ValueAgent import ValueAgent
from agent.GuessAgent import GuessAgent
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
import gurobipy
from gurobipy import *
from scipy.stats import kstwobign

from sklearn.preprocessing import MinMaxScaler
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CalibrateTimeSeries:
    def __init__(self, date_range, symbol="AAPL", data_file="calibration_data", encode_length=8, num_agents=100,
                 lambda_a=1e-12, starting_cash=1e7, historical_date="20190102", seed=None):

        # Calibrate OU process
        OU_param_list = []
        for day in date_range:
            path = "fundamental_{}_{}.bz2".format(symbol, day.strftime("%Y%m%d"))
            path = os.path.join(data_file, path)
            if os.path.exists(path):
                prices = pd.read_pickle(path, compression="bz2")
                prices.index = prices.index.tz_convert(None)
                r_bar, kappa, sigma_s = CalibrateTimeSeries.calibrateOU(prices)
                OU_param_list.append([r_bar, kappa, sigma_s])
        OU_param_list = np.array(OU_param_list)
        self.r_bar, self.kappa, self.sigma_s = OU_param_list.mean(axis=0)

        self.data_feature = np.load("feature/AutoEncoderFeature_{}.npy".format(encode_length))
        with open('model/AutoEncoder_w_{}.h5'.format(encode_length), "rb") as f:
            self.params = pickle.load(f)

        self.symbol = symbol  # only support one symbol
        self.num_agents = num_agents  # number of value agents
        self.lambda_a = lambda_a  # arrival rate of agents
        self.starting_cash = starting_cash  # starting cash of agents
        self.midnight = pd.to_datetime(historical_date)  # historical date
        self.encode_length = encode_length  # feature space dimension
        self.freq = "min"  # frequency of sampling

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

        print("Configuration seed: {}".format(seed))
        logprint("Configuration seed: {}\n".format(seed))
        print("Calibrated OU parameters: r_bar = {}, kappa = {}, sigma_s = {}".format(self.r_bar, self.kappa,
                                                                                      self.sigma_s))
        logprint("Calibrated OU parameters: r_bar = {}, kappa = {}, sigma_s = {}\n".format(self.r_bar, self.kappa,
                                                                                           self.sigma_s))

    def encode(self, x):
        [w1, b1, w2, b2, w3, b3] = self.params
        activation = np.vectorize(lambda x: max(0., x))
        return activation(activation(x @ w1 + b1) @ w2 + b2) @ w3 + b3

    def calibrateModelDRO(self, sigma_n_grid, batch_size=16, parallel=True):
        n1 = self.data_feature.shape[0]
        data_feature_sorted = self.data_feature.copy()
        data_feature_sorted.sort(axis=0)
        empirical_quantile = np.array([np.arange(0., 1 - 1e-15, 1 / n1)] * self.encode_length).T
        # self.plotFeatureDistribution(data_feature_sorted)

        scaler = MinMaxScaler()
        simulate_quantile_list = []

        for sigma_n in sigma_n_grid:
            time_start = time.time()
            mid_prices_dist = self.generateTimeSeriesDistribution(sigma_n, batch_size, parallel)
            mid_prices_dist = np.array([scaler.fit_transform(mid_prices) for mid_prices in mid_prices_dist])

            feature_dist = self.encode(mid_prices_dist[:, :, 0])

            feature_dist.sort(axis=0)
            n2 = len(feature_dist)
            simulate_quantile = np.full_like(empirical_quantile, 0.)
            for i in range(self.encode_length):
                simulate_quantile[:, i] = np.searchsorted(feature_dist[:, i], data_feature_sorted[:, i],
                                                          side='right') / n2
            simulate_quantile_list.append(simulate_quantile)

            loss = np.max(np.abs(empirical_quantile - simulate_quantile))
            loss_argmax = np.argmax(np.abs(empirical_quantile - simulate_quantile))
            arg_x, arg_y = (loss_argmax // self.encode_length, loss_argmax % self.encode_length)
            time_end = time.time()
            print("sigma_n {} finished with time {}, loss is {} at {}.".format(sigma_n, time_end - time_start, loss,
                                                                               (arg_x, arg_y)))
            logprint(
                "sigma_n {} finished with time {}, loss is {} at {}.\n".format(sigma_n, time_end - time_start, loss,
                                                                               (arg_x, arg_y)))
            # self.plotFeatureDistribution(feature_dist)
            #  self.plotQuantileHeatmap(empirical_quantile, simulate_quantile)

        simulate_quantile_list = np.array(simulate_quantile_list)

        print("Quantile calculation finished. Start optimization.")
        logprint("Quantile calculation finished. Start optimization.\n")
        m = gurobipy.Model("DRO")

        q = m.addVar(vtype=GRB.CONTINUOUS, name='q')
        W = dict()
        W_sum = 0
        quantile_avg = dict()
        m.addConstr(q >= 0, name="postive_q")
        for i in range(len(sigma_n_grid)):
            W[i] = m.addVar(vtype=GRB.CONTINUOUS, name='W' + str(i))
            m.addConstr(W[i] >= 0, "postive_W" + str(i))
            W_sum += W[i]
            for j in range(n1):
                for k in range(self.encode_length):
                    if (j, k) in quantile_avg:
                        quantile_avg[(j, k)] += W[i] * simulate_quantile_list[i][j][k]
                    else:
                        quantile_avg[(j, k)] = W[i] * simulate_quantile_list[i][j][k]

        m.addConstr(W_sum == 1, name="sum_prob")
        for j in range(n1):
            for k in range(self.encode_length):
                m.addConstr(empirical_quantile[j][k] - q / np.sqrt(n1 * n2 / (n1 + n2)) <= quantile_avg[(j, k)],
                            name="qCons1_" + str(j) + str(k))
                m.addConstr(empirical_quantile[j][k] + q / np.sqrt(n1 * n2 / (n1 + n2)) >= quantile_avg[(j, k)],
                            name="qCons2_" + str(j) + str(k))

        m.setObjective(q, GRB.MINIMIZE)
        m.optimize()
        print("Optimization finished.")
        logprint("Optimization finished.\n")

        W_optimal = [W[i].x for i in W]
        plt.plot(sigma_n_grid, W_optimal)
        plt.xlabel("$\sigma_n^2$")
        plt.ylabel("Weights")
        plt.savefig("weights_l_{}_b_{}.png".format(self.lambda_a, batch_size))
        plt.show()

        return m.objVal, kstwobign.sf(m.objVal), W_optimal

    def plotFeatureDistribution(self, feature):
        if self.encode_length == 8:
            fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(12, 9))
            axes = ax.flatten()
            for i in range(feature.shape[1]):
                sns.distplot(feature[:, i], ax=axes[i])
            plt.show()
            return

        for i in range(feature.shape[1]):
            sns.distplot(feature[:, i])
            plt.show()
        return

    def plotQuantileHeatmap(self, empirical_quantile, simulate_quantile):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
        ax1 = sns.heatmap(empirical_quantile, ax=ax[0])
        ax1.tick_params(labelsize=8)
        ax1.tick_params(labelsize=8)
        ax2 = sns.heatmap(simulate_quantile, ax=ax[1])
        ax2.tick_params(labelsize=8)
        ax2.tick_params(labelsize=8)
        plt.show()

    def generateTimeSeriesDistribution(self, sigma_n, batch_size=16, parallel=True):
        if not parallel:
            mid_prices_dist = []
            for i in range(batch_size):
                mid_prices = self.generateMidPrices(sigma_n)
                mid_prices_dist.append(mid_prices.values)
        else:
            mid_prices_dist = []
            num_cores = int(mp.cpu_count())
            iter = batch_size // num_cores
            for i in range(iter):
                pool = mp.Pool(num_cores)
                results = [pool.apply_async(self.generateMidPrices, args=(sigma_n,)) for j in range(num_cores)]
                for p in results:
                    mid_prices = p.get()
                    mid_prices_dist.append(mid_prices.values)
                pool.close()
        return np.array(mid_prices_dist)

    def generateMidPrices(self, sigma_n):
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

        # Some value agents.
        # agents.extend([ValueAgent(j, "Value Agent {}".format(j),
        #                           "ValueAgent {}".format(j),
        #                           random_state=np.random.RandomState(
        #                               seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
        #                           log_orders=False, symbol=symbol, starting_cash=self.starting_cash,
        #                           sigma_n=sigma_n, r_bar=s['r_bar'], kappa=s['agent_kappa'],
        #                           sigma_s=s['fund_vol'],
        #                           lambda_a=self.lambda_a) for j in range(agent_count, agent_count + self.num_agents)])
        # agent_types.extend(["ValueAgent {}".format(j) for j in range(self.num_agents)])
        # agent_count += self.num_agents

        agents.extend([GuessAgent(j, "Guess Agent {}".format(j),
                                  "GuessAgent {}".format(j),
                                  random_state=np.random.RandomState(
                                      seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                  log_orders=False, symbol=symbol, starting_cash=self.starting_cash,
                                  sigma_n=sigma_n, lambda_a=self.lambda_a) for j in
                       range(agent_count, agent_count + self.num_agents)])
        agent_types.extend(["GuessAgent {}".format(j) for j in range(self.num_agents)])
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

        midprices_df = midprices[self.symbol]
        if len(midprices_df) == 390:
            new_row = pd.DataFrame({"price": midprices_df.iloc[0, 0]},
                                   index=[midprices_df.index[0] - pd.Timedelta("1min")])
            midprices_df = pd.concat([new_row, midprices_df])

        return midprices_df

    @staticmethod
    def calibrateOU(fundamental_series):
        # The most reasonable and accurate calibration requires freq="ns", since our
        # trading system take these parameters in time unit "ns". However it will be
        # out of memory to use "ns", so we take default freq="s" as a approximation.
        freq = "min"

        if 'FundamentalValue' in fundamental_series.columns:
            fundamental_series = fundamental_series[['FundamentalValue']]

        fundamental_series = fundamental_series * 100
        fundamental_series["diff"] = -fundamental_series['FundamentalValue'].diff(-1)
        mask = fundamental_series.groupby(fundamental_series.index.strftime("%Y%m%d")).apply(lambda x: x.index[-1])
        fundamental_series = fundamental_series.drop(mask.values, axis=0)

        # Regression
        slope, intercept, r_value, p_value, std_err = linregress(fundamental_series['FundamentalValue'],
                                                                 fundamental_series["diff"])

        # Adjust parameters to freq "ns"
        freq_multiplier = pd.Timedelta("1" + freq) / pd.Timedelta("1ns")
        r_bar = -intercept / slope
        kappa = -slope / freq_multiplier
        sigma_s = (1 - r_value ** 2) * fundamental_series["diff"].var() / freq_multiplier

        return r_bar, kappa, sigma_s


log_file = "log_ts"


def logprint(message):
    with open(log_file + ".txt", "a") as f:
        f.write(message)


if __name__ == "__main__":
    logprint("----------------------------------------------------\n")
    model = CalibrateTimeSeries(date_range=pd.date_range(start='1/1/2019', end='12/31/2019', freq="D"))
    sigma_n_grid = np.arange(5, 101, 5)
    sigma_n_grid = sigma_n_grid ** 2

    time_start = time.time()

    result_dict = model.calibrateModelDRO(sigma_n_grid, batch_size=8)
    print(result_dict)
    logprint("{}\n".format(result_dict))

    time_end = time.time()
    print('totally time', time_end - time_start)
    logprint("totally time {}\n".format(time_end - time_start))
    logprint("----------------------------------------------------\n")
