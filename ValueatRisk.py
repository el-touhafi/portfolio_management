from scipy.stats import norm, t
import numpy as np
import pandas as pd
from PortfolioManagement import PortfolioManagement


class ValueatRisk(PortfolioManagement):
    """
    This class will serve on measuring portfolio risk
    using Value at Risk (VaR) and the Expected Shortfall.
    """

    def __init__(
        self,
        components_tics,
        start_date,
        end_date,
        numb_sim,
        dist_prev,
        interest_rate,
        benchmark,
        alpha,
        selection: str,
    ):
        super().__init__(
            components_tics,
            start_date,
            end_date,
            numb_sim,
            dist_prev,
            interest_rate,
            benchmark,
        )
        self.alpha = alpha
        self.selection = selection
        self._value_at_risk_data = pd.DataFrame()
        self.optimal_weight = None

    def get_optimal_weights(self):
        """
        This method will give the optimal weights of the chosen portfolio
        based on investor's preferences.
        """
        if self.selection == "sharpe":
            self.optimal_weight = self.optim_sharpe.iloc[
                0, 2 : len(self.portfolio_comp_data.columns) + 2
            ]
        elif self.selection == "sortino":
            self.optimal_weight = self.optim_sortino.iloc[
                0, 2 : len(self.portfolio_comp_data.columns) + 2
            ]
        else:
            self.optimal_weight = self.optim_minvar.iloc[
                0, 2 : len(self.portfolio_comp_data.columns) + 2
            ]

    def compute(self):
        """
        This method will compute all the necessary components
        for running the process and get the VaR and the Expected Shortfall.
        """
        super().compute()
        self.get_optimal_weights()
        print(self.value_at_risk_data)

    def normal_dist_var(self, mu, sigma):
        """
        This method will serve in computing normal Variance-Covariance or parametric value_at_risk
        """
        value_at_risk = -np.dot(self.optimal_weight, mu) - norm.ppf(
            self.alpha
        ) * np.sqrt(np.dot(np.dot(self.optimal_weight, sigma), self.optimal_weight.T))
        return float(value_at_risk)

    def normal_dist_es(self, mu, sigma):
        """
        This method will serve in computing normal
        Variance-Covariance or parametric expected_shortfall
        """
        expected_shortfall = -np.dot(self.optimal_weight, mu) + norm.pdf(
            norm.ppf(self.alpha)
        ) / self.alpha * np.sqrt(
            np.dot(np.dot(self.optimal_weight, sigma), self.optimal_weight.T)
        )
        return float(expected_shortfall)

    def t_stud_dist_var(self, daily_returns):
        """
        This method will serve in computing t-student
        Variance-Covariance or parametric value_at_risk
        """
        degr_of_free = t.fit(np.dot(daily_returns, self.optimal_weight.transpose()))[0]
        value_at_risk = -(
            np.dot(daily_returns, self.optimal_weight.transpose())
        ).mean() + np.sqrt((degr_of_free - 2) / degr_of_free) * t.ppf(
            1 - self.alpha, degr_of_free
        ) * np.sqrt(
            np.dot(
                np.dot(self.optimal_weight, daily_returns.cov()), self.optimal_weight.T
            )
        )
        return float(value_at_risk)

    def t_stud_dist_es(self, daily_returns):
        """
        This method will serve in computing t-student
        Variance-Covariance or parametric expected_shortfall
        """
        degr_of_free = t.fit(np.dot(daily_returns, self.optimal_weight.transpose()))[0]
        alpha_val = t.ppf(self.alpha, degr_of_free)
        expected_shortfall = -(
            np.dot(daily_returns, self.optimal_weight.transpose())
        ).mean() - 1 / self.alpha * 1 / (1 - degr_of_free) * (
            degr_of_free - 2 + alpha_val**2
        ) * t.pdf(
            alpha_val, degr_of_free
        ) * np.sqrt(
            np.dot(
                np.dot(self.optimal_weight, daily_returns.cov()), self.optimal_weight.T
            )
        )
        return float(expected_shortfall)

    def historical_var(self, daily_returns):
        """
        This method will serve in computing historical value_at_risk
        """
        portfolio_returns = pd.DataFrame(
            np.dot(daily_returns, self.optimal_weight),
            columns=["Portfolio returns"],
            index=daily_returns.index,
        )
        portfolio_returns = portfolio_returns.sort_values(by="Portfolio returns")
        n = len(daily_returns.index) * self.alpha
        if not isinstance(n, int):
            value_at_risk = -(
                portfolio_returns.iloc[int(n), 0]
                + (n * self.alpha - int(n))
                * (
                    portfolio_returns.iloc[int(n) + 1, 0]
                    - portfolio_returns.iloc[int(n), 0]
                )
            )
        else:
            value_at_risk = -portfolio_returns.iloc[int(n), 0]
        return value_at_risk

    def historical_es(self, daily_returns):
        """
        This method will serve in computing historical expected shortfall
        """
        portfolio_returns = pd.DataFrame(
            np.dot(daily_returns, self.optimal_weight),
            columns=["Portfolio returns"],
            index=daily_returns.index,
        )
        portfolio_returns = portfolio_returns.sort_values(by="Portfolio returns")
        n = len(daily_returns.index) * self.alpha
        if not isinstance(n, int):
            value_at_risk = -(
                portfolio_returns.iloc[int(n), 0]
                + (n * self.alpha - int(n))
                * (
                    portfolio_returns.iloc[int(n) + 1, 0]
                    - portfolio_returns.iloc[int(n), 0]
                )
            )
            df = portfolio_returns[portfolio_returns <= -value_at_risk].dropna()
            df = df.values
            df = np.append(df, -value_at_risk)
            expected_shortfall = -df.mean()
        else:
            value_at_risk = portfolio_returns.iloc[int(n), 0]
            df = portfolio_returns[portfolio_returns <= -value_at_risk].dropna()
            df = df.values
            df = np.append(df, -value_at_risk)
            expected_shortfall = -df.mean()
        return expected_shortfall

    def normal_mcs_var(self, daily_returns, n):
        """
        This method will serve in computing Monte Carlo simulation
        using normal distribution to get value_at_risk
        """
        average_returns = daily_returns.mean(axis=0)
        corr_compo = daily_returns.corr()
        std_compo = daily_returns.std(axis=0)
        cholesky = pd.DataFrame(
            np.linalg.cholesky(corr_compo),
            columns=daily_returns.columns,
            index=daily_returns.columns,
        )
        ind_random_var = pd.DataFrame(
            np.random.normal(0, 1, (n + 1, len(daily_returns.columns))),
            columns=daily_returns.columns,
            index=np.array(
                pd.date_range(
                    start=daily_returns.index[-1],
                    end=(
                        pd.to_datetime(daily_returns.index[-1])
                        + pd.to_timedelta(n, unit="D")
                    ),
                    freq="D",
                )
            ),
        )
        depend_random_var = pd.DataFrame(
            np.dot(cholesky, ind_random_var.transpose()).transpose(),
            columns=daily_returns.columns,
            index=ind_random_var.index,
        )
        mcs_returns = pd.DataFrame(
            np.dot(
                average_returns + depend_random_var * std_compo, self.optimal_weight
            ),
            columns=["P"],
            index=depend_random_var.index,
        )
        mcs_returns = mcs_returns.sort_values(by="P")
        n = len(ind_random_var.index) * self.alpha
        if not isinstance(n, int):
            value_at_risk = -(
                mcs_returns.iloc[int(n), 0]
                + (n * self.alpha - int(n))
                * (mcs_returns.iloc[int(n) + 1, 0] - mcs_returns.iloc[int(n), 0])
            )
        else:
            value_at_risk = mcs_returns.iloc[int(n), 0]
        return value_at_risk

    def normal_mcs_es(self, daily_returns, n):
        """
        This method will serve in computing Monte Carlo simulation
        using normal distribution to get expected shortfall
        """
        average_returns = daily_returns.mean(axis=0)
        corr_compo = daily_returns.corr()
        std_compo = daily_returns.std(axis=0)
        cholesky = pd.DataFrame(
            np.linalg.cholesky(corr_compo),
            columns=daily_returns.columns,
            index=daily_returns.columns,
        )
        ind_random_var = pd.DataFrame(
            np.random.normal(0, 1, (n + 1, len(daily_returns.columns))),
            columns=daily_returns.columns,
            index=np.array(
                pd.date_range(
                    start=daily_returns.index[-1],
                    end=(
                        pd.to_datetime(daily_returns.index[-1])
                        + pd.to_timedelta(n, unit="D")
                    ),
                    freq="D",
                )
            ),
        )
        depend_random_var = pd.DataFrame(
            np.dot(cholesky, ind_random_var.transpose()).transpose(),
            columns=daily_returns.columns,
            index=ind_random_var.index,
        )
        portfolio_returns = pd.DataFrame(
            np.dot(
                average_returns + depend_random_var * std_compo, self.optimal_weight
            ),
            columns=["Portfolio returns"],
            index=depend_random_var.index,
        )
        portfolio_returns = portfolio_returns.sort_values(by="Portfolio returns")
        n = len(ind_random_var.index) * self.alpha
        if not isinstance(n, int):
            value_at_risk = -(
                portfolio_returns.iloc[int(n), 0]
                + (n * self.alpha - int(n))
                * (
                    portfolio_returns.iloc[int(n) + 1, 0]
                    - portfolio_returns.iloc[int(n), 0]
                )
            )
            df = portfolio_returns[portfolio_returns <= -value_at_risk].dropna()
            df = df.values
            df = np.append(df, -value_at_risk)
            expected_shortfall = -df.mean()
        else:
            value_at_risk = portfolio_returns.iloc[int(n), 0]
            df = portfolio_returns[portfolio_returns <= -value_at_risk].dropna()
            df = df.values
            df = np.append(df, -value_at_risk)
            expected_shortfall = -df.mean()
        return expected_shortfall

    def t_stud_mcs_var(self, daily_returns, n):
        """
        This method will serve in computing Monte Carlo simulation
        using t-student distribution to get value_at_risk
        """
        degr_of_free = t.fit(np.dot(daily_returns, self.optimal_weight.transpose()))[0]
        average_returns = daily_returns.mean(axis=0)
        std_compo = daily_returns.std(axis=0)
        cholesky = pd.DataFrame(
            np.linalg.cholesky(daily_returns.corr()),
            columns=daily_returns.columns,
            index=daily_returns.columns,
        )
        ind_random_var = pd.DataFrame(
            np.random.standard_t(degr_of_free, (n + 1, len(daily_returns.columns))),
            columns=daily_returns.columns,
            index=np.array(
                pd.date_range(
                    start=daily_returns.index[-1],
                    end=(
                        pd.to_datetime(daily_returns.index[-1])
                        + pd.to_timedelta(n, unit="D")
                    ),
                    freq="D",
                )
            ),
        )
        depend_random_var = pd.DataFrame(
            np.dot(cholesky, ind_random_var.transpose()).transpose(),
            columns=daily_returns.columns,
            index=ind_random_var.index,
        )
        portfolio_returns = pd.DataFrame(
            np.dot(
                average_returns + depend_random_var * std_compo, self.optimal_weight
            ),
            columns=["Portfolio returns"],
            index=depend_random_var.index,
        )
        portfolio_returns = portfolio_returns.sort_values(by="Portfolio returns")
        n = len(ind_random_var.index) * self.alpha
        if not isinstance(n, int):
            value_at_risk = -(
                portfolio_returns.iloc[int(n), 0]
                + (n * self.alpha - int(n))
                * (
                    portfolio_returns.iloc[int(n) + 1, 0]
                    - portfolio_returns.iloc[int(n), 0]
                )
            )
        else:
            value_at_risk = portfolio_returns.iloc[int(n), 0]
        return value_at_risk

    def t_stud_mcs_es(self, daily_returns, n):
        """
        This method will serve in computing Monte Carlo simulation
        using t-student distribution to get expected shortfall
        """
        degr_of_free = t.fit(np.dot(daily_returns, self.optimal_weight.transpose()))[0]
        average_returns = daily_returns.mean(axis=0)
        std_compo = daily_returns.std(axis=0)
        cholesky = pd.DataFrame(
            np.linalg.cholesky(daily_returns.corr()),
            columns=daily_returns.columns,
            index=daily_returns.columns,
        )
        ind_random_var = pd.DataFrame(
            np.random.standard_t(degr_of_free, (n + 1, len(daily_returns.columns))),
            columns=daily_returns.columns,
            index=np.array(
                pd.date_range(
                    start=daily_returns.index[-1],
                    end=(
                        pd.to_datetime(daily_returns.index[-1])
                        + pd.to_timedelta(n, unit="D")
                    ),
                    freq="D",
                )
            ),
        )
        depend_random_var = pd.DataFrame(
            np.dot(cholesky, ind_random_var.transpose()).transpose(),
            columns=daily_returns.columns,
            index=ind_random_var.index,
        )
        portfolio_returns = pd.DataFrame(
            np.dot(
                average_returns + depend_random_var * std_compo, self.optimal_weight
            ),
            columns=["Portfolio returns"],
            index=depend_random_var.index,
        )
        portfolio_returns = portfolio_returns.sort_values(by="Portfolio returns")
        n = len(ind_random_var.index) * self.alpha
        if not isinstance(n, int):
            value_at_risk = -(
                portfolio_returns.iloc[int(n), 0]
                + (n * self.alpha - int(n))
                * (
                    portfolio_returns.iloc[int(n) + 1, 0]
                    - portfolio_returns.iloc[int(n), 0]
                )
            )
            df = portfolio_returns[portfolio_returns <= -value_at_risk].dropna()
            df = df.values
            df = np.append(df, -value_at_risk)
            expected_shortfall = -df.mean()
        else:
            value_at_risk = portfolio_returns.iloc[int(n), 0]
            df = portfolio_returns[portfolio_returns <= -value_at_risk].dropna()
            df = df.values
            df = np.append(df, -value_at_risk)
            expected_shortfall = -df.mean()
        return expected_shortfall

    @property
    def value_at_risk_data(self):
        """
        This method will serve in measuring risk using the value at risk and expected shortfall
        """
        cols = [
            "Normal parametric Value at Risk",
            "Normal parametric Expected Shortfall",
            "T_student parametric Value at Risk",
            "T_student parametric Expected Shortfall",
            "Normal monte carlo sim Value at Risk",
            "Normal monte carlo sim Expected Shortfall",
            "T_student monte carlo sim Value at Risk",
            "T_student monte carlo sim Expected Shortfall",
            "Historical Value at Risk",
            "Historical Expected Shortfall",
        ]
        if self.dist_prev == "normal":
            mu = self.normal_prev_returns.mean(axis=0)
            sigma = self.normal_prev_returns.cov()
            self._value_at_risk_data = pd.DataFrame(
                np.array(
                    self.compute_value_at_risk(
                        self.normal_prev_returns, mu, sigma, 1000
                    )
                ).reshape((1, 10)),
                columns=cols,
            )
        elif self.dist_prev == "t_stud":
            mu = self.tstud_prev_returns.mean(axis=0)
            sigma = self.tstud_prev_returns.cov()
            self._value_at_risk_data = pd.DataFrame(
                np.array(
                    self.compute_value_at_risk(self.tstud_prev_returns, mu, sigma, 1000)
                ).reshape((1, 10)),
                columns=cols,
            )
        else:
            self._value_at_risk_data = pd.DataFrame(
                np.array(
                    self.compute_value_at_risk(
                        self.daily_returns_compo,
                        self.daily_average_compo,
                        self.daily_covariance_compo,
                        1000,
                    )
                ).reshape((1, 10)),
                columns=cols,
            )
        return self._value_at_risk_data.transpose().rename(columns={0: "daily values"})

    def compute_value_at_risk(self, daily_returns, mu, sigma, n):
        """
        This method will serve in gathering the computed
        methods for value at risk and expected shortfall
        """
        p = []
        p.append(self.normal_dist_var(mu, sigma))
        p.append(self.normal_dist_es(mu, sigma))
        p.append(self.t_stud_dist_var(daily_returns))
        p.append(self.t_stud_dist_es(daily_returns))
        p.append(self.normal_mcs_var(daily_returns, n))
        p.append(self.normal_mcs_es(daily_returns, n))
        p.append(self.t_stud_mcs_var(daily_returns, n))
        p.append(self.t_stud_mcs_es(daily_returns, n))
        p.append(self.historical_var(daily_returns))
        p.append(self.historical_es(daily_returns))
        return p


# p = ValueatRisk(
#     ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "META"],
#     "2024-01-01",
#     "2024-07-11",
#     1000,
#     'nomal',
#     0,
#     "^GSPC",
#     0.05,
#     "sharpe",
# )
# p.compute()
"""
P = value_at_risk(['AAPL','GOOG','TSLA','MSFT','AMZN','A','AA','META'],'2023-01-01','2023-11-11',0.05,[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.3])
#print(P.N_VC_value_at_risk())
print(P.N_VC_ES())
#print(P.H_value_at_risk())
print(P.H_ES())
#print(P.N_MCS_value_at_risk(1000))
print(P.N_MCS_ES(1000))
#print(P.t_VC_value_at_risk())
print(P.t_VC_ES())
#print(P.t_MCS_value_at_risk(1000))
print(P.t_MCS_ES(1000))

"""
