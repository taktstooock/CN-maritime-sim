import random
import matplotlib.pyplot as plt
import numpy as np

import base

DISCOUNT_RATE = 0.05  # 割引率


class CustomAgent(base.Agent):
    def trade(self, env, other_agents):  # 売買
        avg_green = sum(agent.n_green for agent in other_agents) / len(other_agents)
        avg_oil = sum(agent.n_oil for agent in other_agents) / len(other_agents)
        n_green_old = self.n_green
        n_oil_old = self.n_oil

        past_years = random.randint(2, 3)
        future_years = random.randint(1, 5)

        predict_n_oils, predict_n_greens = self.predict_n_future(env, other_agents, past_years, future_years) # 未来の船の数を予測(他のエージェントの合計)
        self.history_predict_n_oil.append(predict_n_oils)
        self.history_predict_n_green.append(predict_n_greens)
        # print(f"Agent {self.ind}: Predicted {predict_n_oils} oil ships and {predict_n_greens} green ships in {future_years} years.")

        max_benefit = -1e9 # 負の無限大（最大値を求めるため）
        best_diff_oil = 0
        best_diff_green = 0

        test_case = 100 # 重油船、グリーン船の購入数の変動の幅
        for diff_oil in range(-test_case,test_case+1, 2):
            for diff_green in range(-test_case,test_case+1, 2):
                n_oil = max(0, self.n_oil + diff_oil)
                n_green = max(0, self.n_green + diff_green)
                # predict_n_oils += n_oil
                # predict_n_greens += n_green
                predict_penalties, predict_rebates = self.predict_feebate_future(env, predict_n_oils, predict_n_greens, n_oil, n_green, future_years)

                predict_costs = 0
                predict_sales = 0
                predict_benefits = 0
                for i in range(future_years):
                    predict_costs = (n_oil + diff_oil*i) * (1/20*env.pv_oil + env.p_oil) + (n_green + diff_green*i) * (1/20*env.pv_green + env.p_green)
                    predict_fare = env.fare * 1.025 ** (i + 1) * env.total_n / (predict_n_oils[i] + predict_n_greens[i]) # 運賃の予測（需要が年率2.5%増加）
                    predict_sales = (n_oil + diff_oil*i + n_green + diff_green*i) * predict_fare
                    predict_benefits += (predict_sales - predict_costs - predict_penalties[i] + predict_rebates[i]) * (1 - DISCOUNT_RATE) ** (i + 1) # 割引現在価値
                if predict_benefits > max_benefit:
                    max_benefit = predict_benefits
                    best_diff_oil = diff_oil
                    best_diff_green = diff_green

        self.n_green += best_diff_green
        self.n_oil += best_diff_oil

        if best_diff_oil == test_case or best_diff_oil == -test_case or best_diff_green == test_case or best_diff_green == -test_case:
            # 購入数が最大値または最小値に達した場合、警告を表示
            # print(f"Agent {self.ind}: The number of ships has reached the maximum or minimum value. Bought {best_diff_oil} oil ships and {best_diff_green} green ships.")
            pass

        self.n_oil = max(0, self.n_oil)
        self.n_green = max(0, self.n_green)
        self.history_oil.append(self.n_oil - n_oil_old)
        self.history_green.append(self.n_green - n_green_old)



    def predict_n_future(self, env, agents, past_years, future_years):
        """
        過去past_years年間の特徴量を用いて、未来future_years年後の船の数を予測する
        """
        # 自分以外のAgentが買う重油船の数を予測
        past_sum_n_oils = np.zeros(past_years)
        past_sum_n_greens = np.zeros(past_years)
        years = np.zeros(past_years)
        lack_of_history = 0
        for i in range(past_years):
            years[i] = i
            for agent in agents:
                if len(agent.history_oil) == 0 or agent.history_oil == 1:
                    lack_of_history = 1
                    continue
                elif agent.ind == self.ind:
                    continue
                elif len(agent.history_oil) < past_years-i:
                    continue
                # 過去の重油船とグリーン船の合計を集計
                past_sum_n_oils[i] += agent.history_oil[-(past_years-i)]
                past_sum_n_greens[i] += agent.history_green[-(past_years-i)]

        # 線形回帰
        a_oil, b_oil = np.polyfit(years,past_sum_n_oils,1)
        a_green, b_green = np.polyfit(years, past_sum_n_greens,1)

        pred_n_oil = np.zeros(future_years)
        pred_n_green = np.zeros(future_years)

        for j in range(future_years):
            if lack_of_history:
                if j == 0:
                    pred_n_oil[j] = random.randint(0,5)*len(agents)+env.total_n_oil
                    pred_n_green[j] = random.randint(0,5)*len(agents)+env.total_n_green
                else:
                    pred_n_oil[j] = pred_n_oil[0]
                    pred_n_green[j] = pred_n_green[0]
            else:
                pred_n_oil[j] = a_oil*(j+past_years)+b_oil+env.total_n_oil
                pred_n_green[j] = a_green*(j+past_years)+b_green+env.total_n_green
                if pred_n_oil[j] < 0:
                    pred_n_oil[j] = 0
                if pred_n_green[j] < 0:
                    pred_n_green[j] = 0

        return pred_n_oil, pred_n_green

class CustomEnv(base.Env):
    pass

if __name__ == '__main__':
    sim = base.Simulation(CustomAgent, CustomEnv)
    sim.run()
    sim.plot()
    sim.validate()
