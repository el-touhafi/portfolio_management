import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import t


class PortfolioConstruction:
    """
    This class will serve as a constructor for the user-defined portfolio,
    using historical data as the basis for allocating its components.

    - `components_tics`: A list containing Yahoo Finance tickers.
    - `start_date` and `end_date`: Strings representing dates in the format `%Y-%m-%d`.
    - `dist_prev`: If set to 'normal' or 't_stud', the components will be allocated based
                    on a forecast rather than historical data.
                  Else for historical data use 'Markovitz'.

    Note: By specifying `dist_prev`, we replicate the Markowitz modern portfolio theory,
            but using the selected forecast rather than historical data.
    """
    def __init__(self, components_tics, start_date, end_date, dist_prev: str):
        self.components_tics = components_tics
        self.start_date = start_date
        self.dist_prev = dist_prev
        self.end_date = end_date
        self._daily_returns_compo = pd.DataFrame()
        self._cholesky_matrice = pd.DataFrame()
        self._numb_days = 252
        self.portfolio_comp_data = pd.DataFrame()
        self.t_param = None
        self.daily_covariance_compo = None
        self.daily_sigma_compo = None
        self.daily_correlation_compo = None
        self.daily_average_compo = None
        self.normal_prev_returns = pd.DataFrame()
        self.tstud_prev_returns = pd.DataFrame()

    def load_data(self, time_frame="1d", price_type="Adj Close"):
        """
        Load components data from Yahoo finance.
        """
        n = len(self.components_tics)
        if (n == 1) or (isinstance(self.components_tics, str)):
            df = yf.download(
                tickers=self.components_tics,
                start=self.start_date,
                end=self.end_date,
                interval=time_frame,
            )[price_type]
            columns = []
            columns.append(self.components_tics)
        else:
            df = yf.download(
                tickers=self.components_tics,
                start=self.start_date,
                end=self.end_date,
                interval=time_frame,
            )[price_type]
            columns = self.components_tics
        df = pd.DataFrame(df.values, columns=columns, index=np.array(df.index))
        df = df.dropna()
        self.portfolio_comp_data = df

    @property
    def daily_returns_compo(self):
        """
        Calculate the arithmetic returns of components.
        Formula: return = price(t) / price(t-1) - 1
        """

        self._daily_returns_compo = self.portfolio_comp_data.pct_change()
        return self._daily_returns_compo.dropna()

    def compo_prices_plot(self):
        """
        Plot components prices.
        """
        for compo in self.portfolio_comp_data.columns:
            plt.plot(
                self.portfolio_comp_data.index,
                self.portfolio_comp_data.loc[:,compo],
                label=compo,
                alpha=0.5,
            )
        plt.xticks(rotation=45)
        plt.legend()
        plt.ylabel("Price")
        plt.title("Components plot by date")
        plt.show()

    def compo_returns_plot(self):
        """
        Plot components returns.
        """
        for compo in self.daily_returns_compo.columns:
            plt.plot(
                self.daily_returns_compo.index,
                self.daily_returns_compo.loc[:, compo],
                label=compo,
                alpha=0.5,
            )
        plt.xticks(rotation=45)
        plt.legend()
        plt.ylabel("Return")
        plt.title("Components plot by date")
        plt.show()

    def compo_correlation_plot(self):
        """
        Plot components correlation.
        """
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            self.daily_returns_compo.corr(),
            vmin=-1,
            vmax=1,
            linewidth=1,
            annot=True,
            xticklabels=self.daily_returns_compo.columns,
            yticklabels=self.daily_returns_compo.columns,
        )
        plt.title("HeatMap Correlation")
        plt.show()

    def compute_stats(self):
        """
        Compute components statistics :
        - average returns : sum(returns) / number_of_periods
        - standard deviation : sqrt(sum((return_i - mean_return)^2) / (n - 1))
        - covariance : sum((X_i - mean_X) * (Y_i - mean_Y)) / (n - 1)
        - correlation : covariance(X, Y) / (std_dev(X) * std_dev(Y))
        - t_students params : - Degrees of freedom (ν): Determines the shape
                                of the distribution. As ν increases, the distribution
                                approaches the standard normal distribution.
                              - Location parameter (μ): Shifts the distribution
                                along the x-axis, representing the mean of the distribution.
                              - Scale parameter (σ): Controls the spread of the distribution,
                                similar to the standard deviation in a normal distribution.
        """
        self.daily_average_compo = self.daily_returns_compo.mean(axis=0)
        self.daily_correlation_compo = self.daily_returns_compo.corr()
        self.daily_sigma_compo = self.daily_returns_compo.std()
        self.daily_covariance_compo = self.daily_returns_compo.cov()
        self.t_param = pd.DataFrame(
            columns=self.daily_returns_compo.columns, index=["dof", "loc or mean", "scale or std"]
        )
        for i in range(len(self.daily_returns_compo.columns)):
            self.t_param.iloc[:, i] = t.fit(
                self.daily_returns_compo.iloc[:, i].to_numpy().astype(np.float64)
            )

    def compute(self):
        """
        This method will compute all the necessary components
        for running the process and provide the required data.
        It will also verify whether the covariance matrix is positive definite,
        allowing for the application of the Cholesky decomposition.
        """
        self.load_data()
        self.compute_stats()
        self.positive_definite_matrice_check()

    @property
    def numb_days(self):
        """
        Define the eligible number of days based on data to get annual returns, std ...
        """
        self._numb_days = (
            self.daily_returns_compo.index[-1] - self.daily_returns_compo.index[0]
        ).days
        if (
            self.daily_returns_compo.index[1] - self.daily_returns_compo.index[0]
        ).days > 10:
            self._numb_days = 12
        elif self._numb_days >= 252:
            self._numb_days = 252
        return self._numb_days

    def positive_definite_matrice_check(self):
        """
        check for positive definite covariance matrix of components.
        Matrix A(n,n) is positive definite <=> det(submatrix(A,i)) > 0 , i = {1,2,3,...,n}
        """
        semi_pos = None
        for i in range(len(self.daily_covariance_compo) - 1):
            determin = np.linalg.det(self.daily_covariance_compo.iloc[: i + 1, : i + 1].astype("float64"))
            if determin < 0:
                return "non positive definite"
            if determin == 0 :
                semi_pos = 'semi-positive definite'
        if semi_pos:
            return semi_pos
        return "positive definite"

    @property
    def cholesky_matrice(self):
        """
        if the covariance matrix is definite positive or semi-definite positive the cholesky
        decomposition will be calculated.
        Cholesky decomposition : A = transposed(U) * U
        """
        if self.positive_definite_matrice_check() in [
            "positive definite",
            "semi-positive definite",
        ]:
            return pd.DataFrame(
                np.linalg.cholesky(self.daily_correlation_compo),
                columns=self.daily_returns_compo.columns,
                index=self.daily_returns_compo.columns,
            )
        return self.positive_definite_matrice_check()

    def monte_carlo_sim_normal_dist(self):
        """
        This method will be used to forecast one year
        ahead by leveraging historical data and applying
        Monte Carlo simulation with normal distribution.
        """
        ind_norm_rand_var = pd.DataFrame(
            np.random.normal(0, 1, (253, len(self.daily_returns_compo.columns))),
            columns=self.daily_returns_compo.columns,
            index=np.array(
                pd.date_range(
                    start=self.daily_returns_compo.index[-1],
                    end=(
                        pd.to_datetime(self.daily_returns_compo.index[-1])
                        + pd.to_timedelta(252, unit="D")
                    ),
                    freq="D",
                )
            ),
        )
        if isinstance(self.cholesky_matrice, pd.DataFrame):
            depend_norm_rand_var = pd.DataFrame(
                np.dot(
                    self.cholesky_matrice, ind_norm_rand_var.transpose()
                ).transpose(),
                columns=self.daily_returns_compo.columns,
                index=ind_norm_rand_var.index,
            )
            return (
                self.daily_average_compo + depend_norm_rand_var * self.daily_sigma_compo
            )
        return self.cholesky_matrice

    def norm_dist_prev_mcs(self, numb_sim, quantile_scenario=0.5):
        """
        This method will be used to get the quantile based on the perception
        of the market to forecast the one year.
        """
        portfolios = pd.DataFrame()
        for i in range(numb_sim):
            portfolios = pd.concat(
                [portfolios, self.monte_carlo_sim_normal_dist().add_suffix(str(i))],
                axis=1,
            )
        steps = np.arange(
            0,
            (numb_sim) * len(self.portfolio_comp_data.columns),
            len(self.portfolio_comp_data.columns),
        )
        self.normal_prev_returns = pd.DataFrame()
        for i in range(len(self.portfolio_comp_data.columns)):
            selected_scenario = (
                np.quantile(portfolios.iloc[-1, steps + i], quantile_scenario)
                <= portfolios.iloc[-1, steps + i]
            )[
                (
                    np.quantile(portfolios.iloc[-1, steps + i], quantile_scenario)
                    <= portfolios.iloc[-1, steps + i]
                )
            ].index[
                -1
            ]
            self.normal_prev_returns = pd.concat(
                [self.normal_prev_returns, portfolios[selected_scenario]], axis=1
            )
        self.normal_prev_returns.columns = self.daily_returns_compo.columns

    def monte_carlo_sim_tstud_dist(self):
        """
        This method will be used to forecast one year
        ahead by leveraging historical data and applying
        Monte Carlo simulation with t_student distribution.
        """
        ind_tstud_rand_var = pd.DataFrame(
            np.random.standard_t(
                self.t_param.iloc[0], size=(253, len(self.daily_returns_compo.columns))
            ),
            columns=self.daily_returns_compo.columns,
            index=np.array(
                pd.date_range(
                    start=self.daily_returns_compo.index[-1],
                    end=(
                        pd.to_datetime(self.daily_returns_compo.index[-1])
                        + pd.to_timedelta(252, unit="D")
                    ),
                    freq="D",
                )
            ),
        )
        if isinstance(self.cholesky_matrice, pd.DataFrame):
            depend_tstud_rand_var = pd.DataFrame(
                np.dot(
                    self.cholesky_matrice, ind_tstud_rand_var.transpose()
                ).transpose(),
                columns=self.daily_returns_compo.columns,
                index=ind_tstud_rand_var.index,
            )
            return (
                self.daily_average_compo
                + depend_tstud_rand_var * self.daily_sigma_compo
            )
        return self.cholesky_matrice

    def tstud_dist_prev_mcs(self, numb_sim, quantile_scenario=0.75):
        """
        This method will be used to get the quantile based on the perception
        of the market to forecast the one year.
        """
        portfolios = pd.DataFrame()
        for i in range(numb_sim):
            portfolios = pd.concat(
                [portfolios, self.monte_carlo_sim_tstud_dist().add_suffix(str(i))],
                axis=1,
            )
        steps = np.arange(
            0,
            (numb_sim) * len(self.portfolio_comp_data.columns),
            len(self.portfolio_comp_data.columns),
        )
        self.tstud_prev_returns = pd.DataFrame()
        for i in range(len(self.portfolio_comp_data.columns)):
            selected_scenario = (
                np.quantile(portfolios.iloc[-1, steps + i], quantile_scenario)
                <= portfolios.iloc[-1, steps + i]
            )[
                (
                    np.quantile(portfolios.iloc[-1, steps + i], quantile_scenario)
                    <= portfolios.iloc[-1, steps + i]
                )
            ].index[
                -1
            ]
            self.tstud_prev_returns = pd.concat(
                [self.tstud_prev_returns, portfolios[selected_scenario]], axis=1
            )
        self.tstud_prev_returns.columns = self.daily_returns_compo.columns


# #
# p = PortfolioConstruction(
#     ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "META"], "2024-01-01", "2024-07-11",'normal'
# )
# p.compute()
# p.norm_dist_prev_mcs(1000,0.5)
# p.compo_prices_plot()
# p.compo_returns_plot()
# p.compo_correlation_plot()
# print(p.norm_dist_prev_mcs(10))
# print(p.tstud_dist_prev_mcs(10))
# plt.plot(p.tstud_prev_returns)
# plt.plot(p.normal_prev_returns)
# plt.show()
