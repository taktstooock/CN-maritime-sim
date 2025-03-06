import matplotlib.pyplot as plt
import random
import decreasing_feebaterate as dec
import linear_pred
import base

DISCOUNT_RATE = 0.05

class CustomAgent(dec.CustomAgent):
    def __init__(self, i):
        super().__init__(i)
        # GFS用
        self.initial_n_oil = self.n_oil  # 初期のoil船の数
        self.failed_years = 0  # GFSの基準を満たせなかった年数を保存
        random.seed(i + 100)


    def trade(self, env, other_agents): # 売買
        avg_green = sum(agent.n_green for agent in other_agents) / len(other_agents)
        avg_oil = sum(agent.n_oil for agent in other_agents) / len(other_agents)
        n_green_old = self.n_green
        n_oil_old = self.n_oil

        past_years = random.randint(3, 5)
        future_years = random.randint(1, 3)

        predict_n_oils, predict_n_greens = self.predict_n_future(env, other_agents, past_years, future_years) # 未来の船の数を予測(他のエージェントの合計)
        # print(f"Agent {self.ind}: Predicted {predict_n_oils} oil ships and {predict_n_greens} green ships in {future_years} years.")

        max_benefit = -1e9 # 負の無限大（最大値を求めるため）
        best_diff_oil = 0
        best_diff_green = 0
        # GFSを守るかどうか
        skip_gfs = self.skip_gfs()

        test_case = 500 # 重油船、グリーン船の購入数の変動の幅
        for diff_oil in range(-test_case,test_case+1, 10):
            for diff_green in range(-test_case,test_case+1, 10):
                if not skip_gfs:
                    # GFS の基準をクリアしているかどうか
                    if diff_oil > self.clear_gfs():
                        continue
                n_oil = max(0, self.n_oil + diff_oil)
                n_green = max(0, self.n_green + diff_green)
                predict_total_n_oils = predict_n_oils + n_oil # 自分の重油船の数を加える
                predict_total_n_greens = predict_n_greens + n_green # 自分のグリーン船の数を加える
                predict_penalties, predict_rebates = self.predict_feebate_future(env, predict_total_n_oils, predict_total_n_greens, n_oil, n_green, future_years)

                predict_costs = 0
                predict_sales = 0
                predict_benefits = 0
                for i in range(future_years):
                    predict_costs = (n_oil + diff_oil*i) * (1/20*env.pv_oil + env.p_oil) + (n_green + diff_green*i) * (1/20*env.pv_green + env.p_green)
                    predict_fare = env.fare * 1.025 ** (i + 1) * env.total_n / (predict_total_n_oils[i] + predict_total_n_greens[i]) # 運賃の予測（需要が年率2.5%増加）
                    predict_sales = (n_oil + diff_oil*i + n_green + diff_green*i) * predict_fare
                    predict_benefits += (predict_sales - predict_costs - predict_penalties[i] + predict_rebates[i]) * (1 - DISCOUNT_RATE) ** (i + 1) # 割引現在価値
                if predict_benefits > max_benefit:
                    max_benefit = predict_benefits
                    best_diff_oil = diff_oil
                    best_diff_green = diff_green
                    best_predict_total_n_oils = predict_total_n_oils
                    best_predict_total_n_greens = predict_total_n_greens


        self.history_predict_n_oil.append(best_predict_total_n_oils)
        self.history_predict_n_green.append(best_predict_total_n_greens)
        self.n_green += best_diff_green
        self.n_oil += best_diff_oil
        # クリアしているかどうか判断していなかったら1年プラス
        if best_diff_oil < self.clear_gfs():
            self.failed_years = 0
        else:
            self.failed_years += 1

        if best_diff_oil == test_case or best_diff_oil == -test_case or best_diff_green == test_case or best_diff_green == -test_case:
            # 購入数が最大値または最小値に達した場合、警告を表示
            # print(f"Agent {self.ind}: The number of ships has reached the maximum or minimum value. Bought {best_diff_oil} oil ships and {best_diff_green} green ships.")
            pass

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

    def clear_gfs(self):
        gfs_calc_value = int(-0.7*self.initial_n_oil*(self.failed_years + 1)/(2050 - 2020))
        # GFS の減らす基準よりoil船の数が少なかったらoil船を0にする
        if self.n_oil > 0:
            if self.n_oil + gfs_calc_value < 0:
                gfs_value = -self.n_oil
            else:
                gfs_value = gfs_calc_value
        # oil船の数が0だったら0を継続
        else:
            gfs_value = 0
        return gfs_value

if __name__ == '__main__':
    INITIAL_FEEBATE_RATE = 0.3
    # FEEBATE_CHANGE_RATE = -0.2
    sim = base.Simulation(CustomAgent, dec.CustomEnv)
    sim.initial_feebate_rate = INITIAL_FEEBATE_RATE
    sim.run()
    sim.plot()
    sim.validate()