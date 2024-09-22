import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from PortfolioConstruction import PortfolioConstruction


class PortfolioManagement(PortfolioConstruction):
    """
    This class will manage the portfolio by computing allocations
    according to Markowitz's modern portfolio theory,
    based on the user's specified components.
    It will calculate key metrics like the Sharpe ratio, Sortino ratio, and minimum variance
    to determine the optimal allocation according to the investor's preferences.
    Additionally, allocations can be determined through forecasting
    if the forecast distribution is defined; otherwise, historical data will be used.
    The class will also calculate the portfolio's beta
    and generate plots to illustrate the efficient frontier.
    """

    def __init__(
        self,
        components_tics,
        start_date,
        end_date,
        numb_sim,
        dist_prev,
        interest_rate,
        benchmark: str,
    ):
        super().__init__(components_tics, start_date, end_date, dist_prev)
        self.numb_sim = numb_sim
        self.benchmark = benchmark
        self.interest_rate = interest_rate
        self._semi_cov_matrix = pd.DataFrame()
        self.markovitz_portfolios = pd.DataFrame()
        self.markovitz_ef = pd.DataFrame()
        self.maxdrawdown_markovitz = None
        self.calmar_mcs = None
        self.benchmark_data = pd.DataFrame()
        self.beta_compo = None
        self.optim_calmar = None
        self.optim_treynor = None
        self.optim_maxdrawdown = None
        self.optim_minvar = None
        self.optim_sharpe = None
        self.optim_sortino = None

    def calculate_weights(self, average_returns, cov_returns):
        """
        This method will serve to compute weights with a sum of 1 using monte carlo simulation.
        """
        columns = ["Return", "Std"]
        columns.extend(
            (compo + "_weight" for compo in self.daily_returns_compo.columns)
        )
        markovitz_portfolios = np.zeros((self.numb_sim, len(columns)))
        for i in range(self.numb_sim):
            weights = np.random.dirichlet(
                np.ones(len(self.daily_returns_compo.columns))
            )
            markovitz_portfolios[i][0] = (
                np.dot(weights, average_returns) * self.numb_days
            )
            markovitz_portfolios[i][1] = np.sqrt(
                np.dot(np.dot(weights, cov_returns), np.transpose(weights))
                * self.numb_days
            )
            markovitz_portfolios[i][2:] = weights
        return markovitz_portfolios, columns

    def get_markovitz_portfolios(self):
        """
        This method will serve to compute weights based on forcasted data or historical.
        """
        if self.dist_prev == "normal":
            markovitz_portfolios, columns = self.calculate_weights(
                self.normal_prev_returns.mean(), self.normal_prev_returns.cov()
            )
        elif self.dist_prev == "t_stud":
            markovitz_portfolios, columns = self.calculate_weights(
                self.tstud_prev_returns.mean(), self.tstud_prev_returns.cov()
            )
        else:
            markovitz_portfolios, columns = self.calculate_weights(
                self.daily_average_compo, self.daily_covariance_compo
            )
        markovitz_portfolios = pd.DataFrame(
            markovitz_portfolios, index=range(self.numb_sim), columns=columns
        )
        self.markovitz_portfolios = markovitz_portfolios.sort_values(
            by="Std", ascending=True, ignore_index=True
        )

    def calculate_ef_weights(self, average_returns, cov_returns):
        """
        This method will serve to compute optimal weights
        of the effecient frontiere based on resolving the
        optimisation probleme of markovitz : minimize risk for a given return.
        """
        covariance_compo = np.array(cov_returns) * self.numb_days
        daily_average_compo = np.array(average_returns) * self.numb_days
        inver_covariance_compo = np.linalg.inv(covariance_compo)
        identity = np.ones(len(self.daily_returns_compo.columns))
        eq1 = float(
            np.dot(
                np.dot(np.transpose(identity), inver_covariance_compo),
                daily_average_compo,
            )
        )
        eq2 = float(
            np.dot(
                np.dot(np.transpose(daily_average_compo), inver_covariance_compo),
                daily_average_compo,
            )
        )
        eq3 = float(
            np.dot(np.dot(np.transpose(identity), inver_covariance_compo), identity)
        )
        eq4 = eq2 * eq3 - eq1**2
        columns = ["Return", "Std"]
        columns.extend(
            (compo + "_weight" for compo in self.daily_returns_compo.columns)
        )
        markovitz_ef = np.zeros((1000, len(columns)))
        for i in np.arange(1, 1001):
            markovitz_ef[i - 1][2:] = (
                (
                    eq2 * np.dot(inver_covariance_compo, identity)
                    - eq1 * np.dot(inver_covariance_compo, daily_average_compo)
                )
                + (i / 100001)
                * (
                    eq3 * np.dot(inver_covariance_compo, daily_average_compo)
                    - eq1 * np.dot(inver_covariance_compo, identity)
                )
            ) / eq4
            markovitz_ef[i - 1][1] = np.sqrt(
                (eq3 * (i / 1001) ** 2 - 2 * eq1 * (i / 1001) + eq2) / eq4
            )
            markovitz_ef[i - 1][0] = i / 1001
        return markovitz_ef, columns

    def get_markovitz_efficient_frontier(self):
        """
        This method will serve to compute effecient frontiere
        weights based on forcasted data or historical.
        """
        if self.dist_prev == "normal":
            markovitz_ef, columns = self.calculate_ef_weights(
                self.normal_prev_returns.mean(), self.normal_prev_returns.cov()
            )
        elif self.dist_prev == "t_stud":
            markovitz_ef, columns = self.calculate_ef_weights(
                self.tstud_prev_returns.mean(), self.tstud_prev_returns.cov()
            )
        else:
            markovitz_ef, columns = self.calculate_ef_weights(
                self.daily_average_compo, self.daily_covariance_compo
            )
        markovitz_ef = pd.DataFrame(markovitz_ef, index=range(1000), columns=columns)
        markovitz_ef = markovitz_ef[
            markovitz_ef["Return"]
            >= markovitz_ef[markovitz_ef["Std"] == markovitz_ef.Std.min()]["Std"].iloc[
                0
            ]
        ]
        markovitz_ef = markovitz_ef.sort_values(
            by="Return", ascending=True, ignore_index=True
        )
        markovitz_ef = markovitz_ef.drop_duplicates()
        markovitz_ef = markovitz_ef.reset_index(drop=True)
        self.markovitz_ef = markovitz_ef.dropna()

    def calculate_semi_cov(self, data):
        """
        This method will serve to compute semi-covariance matrix.
        """
        cols = data.columns
        semi_cov_matrix = pd.DataFrame(columns=cols, index=cols)
        sample_numb = len(data.index)
        for i in range(len(cols)):
            semi_cov_matrix.iloc[i, i] = np.dot(
                np.minimum(data.iloc[:, i] - self.interest_rate, 0),
                np.minimum(data.iloc[:, i] - self.interest_rate, 0),
            ) / (sample_numb - 1)
            for k in range(len(cols) - 1, i, -1):
                semi_cov_matrix.iloc[i, k] = np.dot(
                    np.transpose(data.iloc[:, i] - self.interest_rate),
                    (data.iloc[:, k] - self.interest_rate),
                ) / (sample_numb - 1)
                semi_cov_matrix.iloc[k, i] = np.dot(
                    np.transpose(data.iloc[:, i] - self.interest_rate),
                    (data.iloc[:, k] - self.interest_rate),
                ) / (sample_numb - 1)
        return semi_cov_matrix

    @property
    def semi_cov_matrix(self):
        """
        Based on distribution selected this method will compute the
        semi-covariance matrix.
        """
        if self.dist_prev == "normal":
            self._semi_cov_matrix = self.calculate_semi_cov(self.normal_prev_returns)
        elif self.dist_prev == "t_stud":
            self._semi_cov_matrix = self.calculate_semi_cov(self.tstud_prev_returns)
        else:
            self._semi_cov_matrix = self.calculate_semi_cov(self.daily_returns_compo)
        return self._semi_cov_matrix

    def get_data_to_study(self):
        """
        This method will serve to get the eligible data to study.
        """
        if self.dist_prev == "normal":
            self.norm_dist_prev_mcs(int(len(self.portfolio_comp_data) / 2))
        elif self.dist_prev == "t_stud":
            self.tstud_dist_prev_mcs(int(len(self.portfolio_comp_data) / 2))

    def compute(self):
        """
        This method will compute all the necessary components
        for running the process and provide the eligible data.
        """
        super().compute()
        if self.dist_prev:
            self.get_data_to_study()
        self.get_markovitz_portfolios()
        self.get_markovitz_efficient_frontier()
        self.downside_std()
        self.maxdrawdown_calmar()
        self.beta()
        self.portfolio_perf()

    def downside_std(self):
        """
        This method will serve to compute the downside
        standard deviation based on the interest rate given.
        """
        self.markovitz_portfolios["Downside_Std"] = np.sqrt(
            (
                np.dot(
                    np.dot(self.markovitz_portfolios.iloc[:, 2:], self.semi_cov_matrix),
                    np.transpose(self.markovitz_portfolios.iloc[:, 2:]),
                )
                * self.numb_days
            ).astype("float")
        ).diagonal()

    def maxdrawdown_calmar(self):
        """
        This method will serve to compute the maximum drawdown
        being the max spread of the price for the given period
        and then compute the Calmar ratio.
        """
        matrix1 = np.dot(
            self.portfolio_comp_data,
            self.markovitz_portfolios.iloc[
                :, 2 : len(self.portfolio_comp_data.columns) + 2
            ].T,
        )
        matrix1 = pd.DataFrame(matrix1.transpose())
        self.maxdrawdown_markovitz = abs(
            (matrix1.max(axis=1) - matrix1.min(axis=1)) / matrix1.max(axis=1)
        )
        returns = np.dot(
            self.markovitz_portfolios.iloc[
                :, 2 : len(self.portfolio_comp_data.columns) + 2
            ],
            np.array(self.daily_returns_compo.mean(axis=0)) * self.numb_days
            - self.interest_rate,
        )
        self.calmar_mcs = returns / self.maxdrawdown_markovitz

    def beta(self):
        """
        This method will serve to compute the beta
        of the components to the given benchmark.
        """
        self.benchmark_data = PortfolioConstruction(
            self.benchmark, self.start_date, self.end_date, self.dist_prev
        )
        self.benchmark_data.compute()
        all_data = self.daily_returns_compo.copy()
        all_data["SP500"] = np.array(self.benchmark_data.daily_returns_compo)
        all_data = all_data.dropna()
        cov_matrix = all_data.cov()
        variance_benchmark = float(all_data["SP500"].std()) ** 2
        self.beta_compo = pd.DataFrame(
            np.array(cov_matrix.iloc[:, -1] / variance_benchmark),
            columns=["Beta"],
            index=all_data.columns,
        ).drop("SP500")

    def portfolio_perf(self):
        """
        This method will serve to compute the portfolios
        performances and get the optimal portfolios based
        on each ratio.
        """
        self.markovitz_portfolios["Beta"] = np.dot(
            self.markovitz_portfolios.iloc[:, 2:-1], self.beta_compo
        ).flatten()
        self.markovitz_portfolios["Sharpe"] = (
            self.markovitz_portfolios.Return - self.interest_rate
        ) / self.markovitz_portfolios.Std
        self.markovitz_portfolios["Sortino"] = (
            self.markovitz_portfolios.Return - self.interest_rate
        ) / self.markovitz_portfolios.Downside_Std
        self.markovitz_portfolios["Treynor"] = (
            self.markovitz_portfolios.Return - self.interest_rate
        ) / self.markovitz_portfolios.Beta
        self.markovitz_portfolios["Max_drawdown"] = self.maxdrawdown_markovitz
        self.markovitz_portfolios["Calmar"] = self.calmar_mcs

        self.optim_calmar = self.markovitz_portfolios[
            self.markovitz_portfolios["Calmar"]
            == self.markovitz_portfolios.Calmar.max()
        ]
        self.optim_maxdrawdown = self.markovitz_portfolios[
            self.markovitz_portfolios["Max_drawdown"]
            == self.markovitz_portfolios.Max_drawdown.min()
        ]
        self.optim_treynor = self.markovitz_portfolios[
            self.markovitz_portfolios["Treynor"]
            == self.markovitz_portfolios.Treynor.max()
        ]
        self.optim_sortino = self.markovitz_portfolios[
            self.markovitz_portfolios["Sortino"]
            == self.markovitz_portfolios.Sortino.max()
        ]
        self.optim_minvar = self.markovitz_portfolios[
            self.markovitz_portfolios["Std"] == self.markovitz_portfolios.Std.min()
        ]
        self.optim_sharpe = self.markovitz_portfolios[
            self.markovitz_portfolios["Sharpe"]
            == self.markovitz_portfolios.Sharpe.max()
        ]

        return (
            self.optim_calmar,
            self.optim_maxdrawdown,
            self.optim_treynor,
            self.optim_sortino,
            self.optim_minvar,
            self.optim_sharpe,
        )

    def performance_plot(self):
        """
        This method will serve to plot all the portfolios
        and map the optimal portfolios in the plot.
        """
        plt.scatter(
            self.markovitz_portfolios.Std,
            self.markovitz_portfolios.Return,
            s=10,
            alpha=0.1,
        )
        plt.plot(self.markovitz_ef.Std, self.markovitz_ef.Return)

        plt.scatter(
            self.optim_minvar.Std, self.optim_minvar.Return, s=40, label="min_variance"
        )
        plt.scatter(
            self.optim_sharpe.Std,
            self.optim_sharpe.Return,
            s=35,
            label="Optimal Sharpe",
        )
        plt.scatter(
            self.optim_sortino.Std,
            self.optim_sortino.Return,
            s=30,
            label="Optimal Sortino",
        )
        plt.scatter(
            self.optim_treynor.Std,
            self.optim_treynor.Return,
            s=25,
            label="Optimal Treynor",
        )
        plt.scatter(
            self.optim_calmar.Std,
            self.optim_calmar.Return,
            s=20,
            label="Optimal Calmar",
        )

        plt.xlim(
            [
                min(self.markovitz_portfolios.Std) - 0.1,
                max(self.markovitz_portfolios.Std) + 0.2,
            ]
        )
        plt.title("Monte Carlo Simulation for the Portfolio")
        plt.xticks(rotation=45)
        plt.ylabel("Return")
        plt.xlabel("Volatility")
        plt.ylim(
            [
                min(self.markovitz_portfolios.Return) - 0.1,
                max(self.markovitz_portfolios.Return) + 0.2,
            ]
        )
        plt.legend()
        plt.show()


# p = PortfolioManagement(
#     ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "META"],
#     "2023-01-01",
#     "2024-07-11",
#     1000,
#     "t_stud",
#     0,
#     "^GSPC",
# )
# p.compute()
# print(p.optim_sharpe)
#
p = PortfolioManagement(
    ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "META"],
    "2023-01-01",
    "2024-07-11",
    1000,
    "markovitz",
    0,
    "^GSPC",
)
p.compute()
print(p.optim_sharpe)
p.performance_plot()
# p = PortfolioManagement(
#     ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "META"],
#     "2020-01-01",
#     "2024-07-11",
#     1000,
#     "normal",
#     0,
#     "^GSPC",
# )
# p.compute()
# print(p.optim_sharpe)