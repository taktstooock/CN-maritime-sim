import base
import linear_pred
from base import Agent, Env, Simulation, DISCOUNT_RATE
import random

class CustomAgent(linear_pred.CustomAgent):
    def __init__(self, i):
        super().__init__(i)
        self.n_green = 0
        self.investment_cutoff = 2050  # 2040年まで研究投資
        self.investment_type = random.choice(['Type1', 'Type2', 'Type3'])
        self.investment_rate = {'Type1': 0.20, 'Type2': 0.10, 'Type3': 0.01}[self.investment_type]
        self.green_available_year = {'Type1': 2026, 'Type2': 2028, 'Type3': 2030}[self.investment_type]

    def trade(self, env, other_agents, year):
        """
        - `green_available_year` 以前は `diff_green = 0`
        - `green_available_year` 以降は、通常の `trade` 処理
        - `Type1` は **2026年～2030年** の間、`pv_green` を 0.7倍にしてコスト計算
        """
        avg_oil = sum(agent.n_oil for agent in other_agents) / len(other_agents)
        avg_green = sum(agent.n_green for agent in other_agents) / len(other_agents) if year >= self.green_available_year else 0
        n_green_old = self.n_green
        n_oil_old = self.n_oil

        past_years = random.randint(1, 3)
        future_years = random.randint(1, 5)

        predict_n_oils, predict_n_greens = self.predict_n_future(env, other_agents, past_years, future_years)
        self.history_predict_n_oil.append(predict_n_oils)
        self.history_predict_n_green.append(predict_n_greens)

        max_benefit = -1e9
        best_diff_oil = 0
        best_diff_green = 0

        test_case = 100  # 購入数の変動幅

        for diff_oil in range(-test_case, test_case + 1, 2):
            diff_green = 0 if year < self.green_available_year else random.choice(range(-test_case, test_case + 1, 2))

            n_oil = max(0, self.n_oil + diff_oil)
            n_green = max(0, self.n_green + diff_green)

            predict_penalties, predict_rebates = self.predict_feebate_future(env, predict_n_oils, predict_n_greens, n_oil, n_green, future_years)

            predict_benefits = 0
            for i in range(future_years):
                discount_factor = (1 - DISCOUNT_RATE) ** (i + 1)

                # Type1 のみ 2026年～2030年の間は pv_green を 0.7倍する
                adjusted_pv_green = env.pv_green * 0.5 if self.investment_type == 'Type1' and 2027 <= year <= 2035 else env.pv_green

                predict_costs = (
                    (n_oil + diff_oil * i) * (1 / 20 * env.pv_oil + env.p_oil) +
                    (n_green + diff_green * i) * (1 / 20 * adjusted_pv_green + env.p_green)
                )

                predict_fare = env.fare * 1.025 ** (i + 1) * env.total_n / (predict_n_oils[i] + predict_n_greens[i])
                predict_sales = (n_oil + diff_oil * i + n_green + diff_green * i) * predict_fare
                predict_benefits += (predict_sales - predict_costs - predict_penalties[i] + predict_rebates[i]) * discount_factor

            if predict_benefits > max_benefit:
                max_benefit = predict_benefits
                best_diff_oil = diff_oil
                best_diff_green = diff_green

        self.n_green += best_diff_green
        self.n_oil += best_diff_oil

        self.n_oil = max(0, self.n_oil)
        self.n_green = max(0, self.n_green)
        self.history_oil.append(self.n_oil - n_oil_old)
        self.history_green.append(self.n_green - n_green_old)

class CustomEnv(Env):
    def __init__(self, agents, initial_p_green, initial_p_oil, initial_pv_green, initial_pv_oil, initial_fare, initial_feebate_rate, feebate_change_rate):
        self.research_fund = 0  # 研究資金の初期化
        self.feebate_change_rate = feebate_change_rate  # フィーベイト率の変更
        super().__init__(agents, initial_p_green, initial_p_oil, initial_pv_green, initial_pv_oil, initial_fare, initial_feebate_rate)

    def apply_research_effects(self):
        reduction_factor = max(0.7, 1 - (self.research_fund / 1e8))
        self.pv_green *= reduction_factor
        self.p_green *= reduction_factor

    def update(self):
        self.apply_research_effects()
        super().update()

class CustomSimulation(Simulation):
    def __init__(self):
        """元の Simulation クラスを拡張し、CustomAgent を使用"""
        self.FEEBATE_CHANGE_RATE = 0.1  # フィーベイト率の変更
        super().__init__(CustomAgent, lambda agents, p_green, p_oil, pv_green, pv_oil, fare, feebate_rate:
                         CustomEnv(agents, p_green, p_oil, pv_green, pv_oil, fare, feebate_rate, self.FEEBATE_CHANGE_RATE))

    def run(self):
        # シミュレーションの実行
        for year in range(2020, 2020 + self.time):
            self.years.append(year)

            for agent in self.agents:
                agent.trade(self.env, self.agents, year)

            self.env.update()

            for agent in self.agents:
                agent.renew(self.env, year)

            self.total_n_green_history.append(self.env.total_n_green)
            self.total_n_oil_history.append(self.env.total_n_oil)
            self.total_n_history.append(self.env.total_n)
            self.p_green_history.append(self.env.p_green)
            self.p_oil_history.append(self.env.p_oil)
            self.pv_green_history.append(self.env.pv_green)
            self.pv_oil_history.append(self.env.pv_oil)
            self.fare_history.append(self.env.fare)
            self.demand_history.append(self.env.demand)
            self.agent_avg_oil.append(sum(agent.n_oil for agent in self.agents) / len(self.agents))
            self.agent_avg_green.append(sum(agent.n_green for agent in self.agents) / len(self.agents))
            self.feebate_rate_history.append(self.env.feebate_rate)

            # エージェントごとの利益を記録
            for i, agent in enumerate(self.agents):
                self.agent_benefit_history[i].append(agent.benefit)

            print("\r" + f"Year: {year}, Total Green Ships: {self.env.total_n_green}, Total Oil Ships: {self.env.total_n_oil}, Demand: {self.env.demand}, Fare: {self.env.fare}", end="")
        print("\n")

    def plot(self):
       super().plot()
       #投資のtypeを表示
       print("\nAgent Investment Types:")
       for agent in self.agents:
        print(f"Agent {agent.ind}: {agent.investment_type}")

    def validate(self):
        """
        シミュレーションの結果を検証する
        1. 2050年までにoil船の数が0になること
        2. 2050年までにoil線の数が初期値の7割以下になること
        """
        index_2050 = self.years.index(2050)
        total_n_oil_2050 = self.total_n_oil_history[index_2050]
        print("2050年までにoil船の数が0になること:", total_n_oil_2050 == 0)
        print("2050年までにoil線の数が初期値の7割以下になること:", total_n_oil_2050 < 0.7 * self.initial_pv_oil)
if __name__ == '__main__':
    sim = CustomSimulation()
    sim.run()
    sim.plot()

# if __name__ == '__main__':
#     sim = CustomSimulation()
#     sim.run()
#     sim.plot()
