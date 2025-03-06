import random
import matplotlib.pyplot as plt
import numpy as np

import base

DISCOUNT_RATE = 0.05  # 割引率


class CustomAgent(base.Agent):
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
                if len(agent.history_oil) == 0 or len(agent.history_oil) == 1:
                    lack_of_history = 1
                    continue
                elif agent.ind == self.ind:
                    continue
                elif len(agent.history_oil) < past_years-i:
                    continue
                # 過去の重油船とグリーン船の差分の合計を集計
                past_sum_n_oils[i] += agent.history_oil[-(past_years-i)]
                past_sum_n_greens[i] += agent.history_green[-(past_years-i)]
            if not i:
                continue
                # 差分を一番過去から足し合わせる
            for j in range(i):
                past_sum_n_oils[i] += past_sum_n_oils[j]
                past_sum_n_greens[i] += past_sum_n_greens[j]

        # 線形回帰
        a_oil, _ = np.polyfit(years,past_sum_n_oils,1)
        a_green, _ = np.polyfit(years, past_sum_n_greens,1)

        pred_n_oil = np.zeros(future_years)
        pred_n_green = np.zeros(future_years)

        for j in range(future_years):
            for agent in agents:
                if agent.ind == self.ind:
                    continue
                pred_n_oil[j] += agent.n_oil
                pred_n_green[j] += agent.n_green

            if lack_of_history:
                if j == 0:
                    pred_n_oil[j] = random.randint(-5,5)*len(agents)
                    pred_n_green[j] = random.randint(-5,5)*len(agents)
                else:
                    pred_n_oil[j] = pred_n_oil[0]
                    pred_n_green[j] = pred_n_green[0]
            else:
                pred_n_oil[j] += a_oil*(j+1)
                pred_n_green[j] += a_green*(j+1)

        pred_n_oil = np.maximum(pred_n_oil,0)
        pred_n_green = np.maximum(pred_n_green,0)

        return pred_n_oil, pred_n_green

if __name__ == '__main__':
    sim = base.Simulation(CustomAgent, base.Env)
    sim.run()
    sim.plot()
    sim.validate()
