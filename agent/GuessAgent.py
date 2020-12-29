from agent.TradingAgent import TradingAgent
from util.util import log_print

import numpy as np
import pandas as pd


class GuessAgent(TradingAgent):

    def __init__(self, id, name, type, symbol='IBM', sigma_n=10000, starting_cash=100000,
                 lambda_a=0.005, log_orders=False, random_state=None):

        # Base class init.
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        # Store important parameters particular to the ZI agent.
        self.symbol = symbol  # symbol to trade
        self.sigma_n = sigma_n  # observation noise variance
        self.lambda_a = lambda_a  # mean arrival rate of ZI agents

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state = 'AWAITING_WAKEUP'

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time = None

        self.size = np.random.randint(20, 50)  # size that the agent will be placing

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()
        # self.exchangeID is set in TradingAgent.kernelStarting()

        super().kernelStarting(startTime)

        self.oracle = self.kernel.oracle

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

        # Print end of day valuation.
        H = int(round(self.getHoldings(self.symbol), -2) / 100)
        # May request real fundamental value from oracle as part of final cleanup/stats.

        # marked to fundamental
        rT = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=0, random_state=self.random_state)

        # final (real) fundamental value times shares held.
        surplus = rT * H

        log_print("surplus after holdings: {}", surplus)

        # Add ending cash value and subtract starting cash value.
        surplus += self.holdings['CASH'] - self.starting_cash
        surplus = float(surplus) / self.starting_cash

        self.logEvent('FINAL_VALUATION', surplus, True)

        log_print(
            "{} final report.  Holdings {}, end cash {}, start cash {}, final fundamental {}, surplus {}",
            self.name, H, self.holdings['CASH'], self.starting_cash, rT, surplus)

        # print("Final surplus", self.name, surplus)

    def wakeup(self, currentTime):
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(currentTime)

        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                log_print("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        self.setWakeup(currentTime + pd.Timedelta('{}ns'.format(int(round(delta_time)))))

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
            return

        self.cancelOrders()

        if type(self) == GuessAgent:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
        else:
            self.state = 'ACTIVE'

    def updateEstimates(self):
        # Called by a background agent that wishes to obtain a new fundamental observation,
        # update its internal estimation parameters, and compute a new total valuation for the
        # action it is considering.

        # The agent obtains a new noisy observation of the current fundamental value
        # and uses this to update its internal estimates in a Bayesian manner.
        obs_t = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=self.sigma_n,
                                         random_state=self.random_state)

        log_print("{} observed {} at {}", self.name, obs_t, self.currentTime)

        # Update internal estimates of the current fundamental value and our error of same.

        # If this is our first estimate, treat the previous wake time as "market open".
        if self.prev_wake_time is None: self.prev_wake_time = self.mkt_open

        # Random guess agent directly take the observation as estimation
        r_T = obs_t

        # Our final fundamental estimate should be quantized to whole units of value.
        r_T = int(round(r_T))

        # Finally (for the final fundamental estimation section) remember the current
        # time as the previous wake time.
        self.prev_wake_time = self.currentTime

        log_print("{} estimates r_T = {} as of {}", self.name, r_T, self.currentTime)

        return r_T

    def placeOrder(self):
        # estimate final value of the fundamental price
        # used for surplus calculation
        r_T = self.updateEstimates()

        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

        if bid and ask:
            mid = int((ask + bid) / 2)

            if r_T < mid:
                # fundamental belief that price will go down, place a sell order
                buy = False
                p = r_T  # submit a market order to sell, limit order inside the spread or deeper in the book
            elif r_T >= mid:
                # fundamental belief that price will go up, buy order
                buy = True
                p = r_T  # submit a market order to buy, a limit order inside the spread or deeper in the book
        else:
            # initialize randomly
            buy = np.random.randint(0, 1 + 1)
            p = r_T

        # Place the order
        self.placeLimitOrder(self.symbol, self.size, buy, p)

    def receiveMessage(self, currentTime, msg):
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receiveMessage(currentTime, msg)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == 'AWAITING_SPREAD':
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if msg.body['msg'] == 'QUERY_SPREAD':
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed: return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                self.placeOrder()
                self.state = 'AWAITING_WAKEUP'

        # Cancel all open orders.
        # Return value: did we issue any cancellation requests?

    def cancelOrders(self):
        if not self.orders: return False

        for id, order in self.orders.items():
            self.cancelOrder(order)

        return True

    def getWakeFrequency(self):
        return pd.Timedelta(self.random_state.randint(low=0, high=100), unit='ns')
