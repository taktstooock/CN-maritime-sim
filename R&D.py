class CustomAgent(Agent):
    def __init__(self, i):
       super().__init__(i)
       self.n_green = 0
       self.investment_cutoff = 2040  # 追加!! 2040年まで研究投資
       # 投資タイプの割り当て　＃追加
       self.investment_type = random.choice(['Type1', 'Type2', 'Type3'])
       self.investment_rate = {'Type1': 0.10, 'Type2': 0.05, 'Type3': 0.0}[self.investment_type]
       self.green_available_year = {'Type1': 2026, 'Type2': 2028, 'Type3': 2030}[self.investment_type]

    def renew(self, env, year):  # 決算 #yearを変数に追加
       # 研究投資の処理　＃追加
       if year <= self.investment_cutoff:
            research_investment = self.benefit * self.investment_rate
            self.benefit -= research_investment
            env.research_fund += research_investment
    def trade(self, env, other_agents, year):
        if year < self.green_available_year:
            can_buy_green = False
        else:
            can_buy_green = True

        best_diff_oil, best_diff_green, max_benefit = 0, 0, -1e9
        test_case = 25

        for diff_oil in range(-test_case, test_case + 1):
            for diff_green in range(-test_case, test_case + 1):
                if not can_buy_green and diff_green > 0:
                    continue

                n_oil = max(0, self.n_oil + diff_oil)
                n_green = max(0, self.n_green + diff_green)
                green_price = env.p_green * 0.5 if self.investment_type == 'Type1' else env.p_green

                cost = n_oil * (1/20 * env.pv_oil + env.p_oil) + n_green * (1/20 * env.pv_green + green_price)
                sales = (n_oil + n_green) * env.fare
                benefit = sales - cost

                if benefit > max_benefit:
                    max_benefit, best_diff_oil, best_diff_green = benefit, diff_oil, diff_green

        self.n_green += best_diff_green
        self.n_oil += best_diff_oil
        self.n_oil = max(0, self.n_oil)
        self.n_green = max(0, self.n_green)
        self.history_oil.append(best_diff_oil)
        self.history_green.append(best_diff_green)

class CustomEnv(Env):
    def __init__(self, agents, initial_p_green, initial_p_oil, initial_pv_green, initial_pv_oil, initial_fare, initial_feebate_rate, feebate_change_rate):
        self.research_fund = 0  # Initialize research_fund here
        super().__init__(agents, initial_p_green, initial_p_oil, initial_pv_green, initial_pv_oil, initial_fare, initial_feebate_rate, feebate_change_rate)
       

       #関数を追加
    def apply_research_effects(self):
        reduction_factor = max(0.8, 1 - (self.research_fund / 5e9))
        self.pv_green *= reduction_factor
        self.p_green *= reduction_factor
    
    def update(self):
        # Call apply_research_effects before calling super().update()
        # self.apply_research_effects() #研究効果を先に適用 # Move this line after super().update()
        super().update()
      #追加
        self.apply_research_effects()  # 研究効果を適用
        self.fare *= 1.025  # 毎年2.5%増加
        self.feebate_rate *= 1 - self.feebate_change_rate # Remove this line - already in super().update()


class CustomSimulation(Simulation):
    def __init__(self):
        """元の Simulation クラスを拡張し、CustomAgent を使用"""
        super().__init__(CustomAgent, CustomEnv)

    def run(self):
        # super().run()  # 親クラスの run メソッドを直接呼び出さない
        for year in range(2020, 2020 + self.time):
            self.years.append(year)

            for agent in self.agents:
                agent.trade(self.env, self.agents, year)

            self.env.update()

            for agent in self.agents:
                agent.renew(self.env, year)

            # Add the following lines to update history lists:
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

            # エージェントごとの利益を記録
            for i, agent in enumerate(self.agents):
                self.agent_benefit_history[i].append(agent.benefit)

        print(f"Year: {year}, Research Fund: {self.env.research_fund:.2f}")

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
