import matplotlib.pyplot as plt
import random
import base

DISCOUNT_RATE = 0.05

class CustomAgent(base.Agent):
    def __init__(self, i):
        super().__init__(i)
        # GFS用
        self.initial_n_oil = self.n_oil  # 初期のoil船の数
        self.failed_years = 0  # GFSの基準を満たせなかった年数を保存
        random.seed(i + 100)

    def trade(self, env, other_agents):  # 売買
        avg_green = sum(agent.n_green for agent in other_agents) / len(other_agents)
        avg_oil = sum(agent.n_oil for agent in other_agents) / len(other_agents)
        n_green_old = self.n_green
        n_oil_old = self.n_oil

        past_years = random.randint(1, 5)
        future_years = random.randint(1, 5)
        max_benefit = 0
        best_diff_oil = 0
        best_diff_green = 0
        # GFSを守るかどうか
        skip_gfs = self.skip_gfs()

        test_case = 25
        for diff_oil in range(-test_case,test_case+1):
            for diff_green in range(-test_case,test_case+1):
                if not skip_gfs:
                    # GFS の基準をクリアしているかどうか
                    if not self.clear_gfs(diff_oil):
                        continue
                n_oil = max(0, self.n_oil + diff_oil)
                n_green = max(0, self.n_green + diff_green)
                predict_n_oils, predict_n_greens = self.predict_n_future(env, other_agents, past_years, future_years)
                predict_penalties, predict_rebates = self.predict_feebate_future(env, predict_n_oils, predict_n_greens, n_oil, n_green, future_years)

                predict_costs = 0
                predict_sales = 0
                predict_benefits = 0
                for i in range(future_years):
                    predict_costs = (n_oil + diff_oil*i) * (1/20*env.pv_oil + env.p_oil) + (n_green + diff_green*i) * (1/20*env.pv_green + env.p_green)
                    predict_sales = (n_oil + diff_oil*i + n_green + diff_green*i) * env.fare
                    predict_benefits += (predict_sales - predict_costs - predict_penalties[i] + predict_rebates[i]) * (1 - DISCOUNT_RATE) ** (i + 1)
                if predict_benefits > max_benefit:
                    max_benefit = predict_benefits
                    best_diff_oil = diff_oil
                    best_diff_green = diff_green

        self.n_green += best_diff_green
        self.n_oil += best_diff_oil
        # クリアしているかどうか判断していなかったら1年プラス
        if self.clear_gfs(best_diff_oil):
            self.failed_years = 0
        else:
            self.failed_years += 1

        self.n_oil = max(0, self.n_oil)
        self.n_green = max(0, self.n_green)
        self.history_oil.append(self.n_oil - n_oil_old)
        self.history_green.append(self.n_green - n_green_old)

    def skip_gfs(self):
        # 1年クリアしていなかったらskipしない
        if self.failed_years:
            skip_gfs = 0
        else:
            skip_gfs = random.randint(0,1)
        return skip_gfs

    def clear_gfs(self, diff_oil):
        gfs_value = int(-0.7*self.initial_n_oil*(self.failed_years + 1)/(2050 - 2020))
        return diff_oil < gfs_value

class CustomEnv(base.Env):
    pass

if __name__ == '__main__':
    sim = base.Simulation(CustomAgent, CustomEnv)
    sim.run()
    sim.plot()
    sim.validate()