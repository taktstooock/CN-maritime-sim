import random
import matplotlib.pyplot as plt
import numpy as np
import base
from base import Agent, Env, Simulation

DISCOUNT_RATE = 0.05  # 割引率

class Agent:
    def __init__(self, i):
        self.n_green = random.randint(0, 1)  # green船の数
        self.n_oil = random.randint(150, 250)  # oil船の数
        self.ind = i  # 識別子
        self.benefit = 0  # 初期利益
        self.last_benefit = 0  # 前年の利益
        self.history_oil = [] #重油のログ
        self.history_green = [] #グリーンのログ
        self.history_predict_n_oil = [] #重油船の総数の予測ログ
        self.history_predict_n_green = [] #グリーン船の総数の予測ログ

    def renew(self, env, year):  # 決算
        self.cost = self.n_green * (1/20*env.pv_green + env.p_green) + self.n_oil * (1/20*env.pv_oil + env.p_oil)  # 輸送コスト
        self.sales = (self.n_green + self.n_oil) * env.fare  # 売上
        self.last_benefit = self.benefit  # 前年の利益を更新
        self.benefit = self.sales - self.cost  # 利益

    def trade(self, env, other_agents):  # 売買
        avg_green = sum(agent.n_green for agent in other_agents) / len(other_agents)
        avg_oil = sum(agent.n_oil for agent in other_agents) / len(other_agents)
        n_green_old = self.n_green
        n_oil_old = self.n_oil

        past_years = random.randint(3, 7)
        future_years = random.randint(1, 3)

        predict_n_oils, predict_n_greens = self.predict_n_future(env, other_agents, past_years, future_years) # 未来の船の数を予測(他のエージェントの合計)
        # print(f"Agent {self.ind}: Predicted {predict_n_oils} oil ships and {predict_n_greens} green ships in {future_years} years.")

        max_benefit = -1e9 # 負の無限大（最大値を求めるため）
        best_diff_oil = 0
        best_diff_green = 0

        test_case = 500 # 重油船、グリーン船の購入数の変動の幅
        for diff_oil in range(-test_case,test_case+1, 10):
            for diff_green in range(-test_case,test_case+1, 10):
                n_oil = max(0, self.n_oil + diff_oil)
                n_green = max(0, self.n_green + diff_green)
                predict_total_n_oils = predict_n_oils + n_oil # 自分の重油船の数を加える
                predict_total_n_greens = predict_n_greens + n_green # 自分のグリーン船の数を加える
                predict_costs = 0
                predict_sales = 0
                predict_benefits = 0
                for i in range(future_years):
                    predict_costs = (n_oil + diff_oil*i) * (1/20*env.pv_oil + env.p_oil) + (n_green + diff_green*i) * (1/20*env.pv_green + env.p_green)
                    predict_fare = env.fare * 1.025 ** (i + 1) * env.total_n / (predict_total_n_oils[i] + predict_total_n_greens[i]) if predict_total_n_oils[i] + predict_total_n_greens[i] != 0 else 0 # 運賃の予測（需要が年率2.5%増加）
                    predict_sales = (n_oil + diff_oil*i + n_green + diff_green*i) * predict_fare
                    predict_benefits += (predict_sales - predict_costs) * (1 - DISCOUNT_RATE) ** (i + 1) # 割引現在価値
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

        if best_diff_oil == test_case or best_diff_oil == -test_case or best_diff_green == test_case or best_diff_green == -test_case:
            # 購入数が最大値または最小値に達した場合、警告を表示
            # print(f"Agent {self.ind}: The number of ships has reached the maximum or minimum value. Bought {best_diff_oil} oil ships and {best_diff_green} green ships.")
            pass

        self.n_oil = max(0, self.n_oil)
        self.n_green = max(0, self.n_green)
        self.history_oil.append(self.n_oil - n_oil_old)
        self.history_green.append(self.n_green - n_green_old)

    def predict_n(self, env, agents, year):
        predict_sum_n_oil = 0; predict_sum_n_green = 0

        # 自分以外のAgentが買う重油船の数を予測
        for agent in agents:

            # もし自分であればスキップ
            if agent.ind == self.ind:
              continue

            if len(agent.history_green) == 0:
                new_buy_oil = random.randint(0, 5)
                new_buy_green = random.randint(0, 5)
            elif len(agent.history_green) < year:
                new_buy_oil = sum(agent.history_oil)/ len(agent.history_oil)
                new_buy_green = sum(agent.history_green)/ len(agent.history_green)
            else:
            #total
                new_buy_oil = sum(agent.history_oil[-year:]) / year
                new_buy_green = sum(agent.history_green[-year:]) / year

            predict_sum_n_oil += new_buy_oil
            predict_sum_n_green += new_buy_green

        #整数化
        return env.total_n_oil + int(predict_sum_n_oil), env.total_n_green + int(predict_sum_n_green)

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

class Env:
    def __init__(self, agents, initial_p_green, initial_p_oil, initial_pv_green, initial_pv_oil, initial_fare):
        self.agents = agents
        self.p_green = initial_p_green
        self.p_oil = initial_p_oil
        self.pv_green = initial_pv_green
        self.pv_oil = initial_pv_oil
        self.fare = initial_fare
        self.total_n_green = self.cal_total_n_green()
        self.total_n_oil = self.cal_total_n_oil()
        self.total_n = self.total_n_green + self.total_n_oil
        self.demand = self.fare * self.total_n
        self.update()

    def cal_total_n_oil(self):
        return sum(agent.n_oil for agent in self.agents)

    def cal_total_n_green(self):
        return sum(agent.n_green for agent in self.agents)

    def market(self):
        self.p_oil *= 1 + random.uniform(-0.10, 0.10)
        self.pv_oil *= 1 + random.uniform(-0.10, 0.10)

        self.p_green = max(self.p_green, self.p_oil/2) *(1+random.uniform(-0.10, 0))
        self.pv_green = max(min(180 ,180*200*4/(self.total_n_green+1)),70) *(1+random.uniform(-0.05, 0.05)) # 200 = 25年くらいで下がるように

    def update(self):
        self.total_n_green = self.cal_total_n_green()
        self.total_n_oil = self.cal_total_n_oil()
        self.total_n = self.total_n_green + self.total_n_oil
        self.demand *= 1.025  # 毎年2.5%増加
        self.fare = self.demand / self.total_n
        self.market()

class Simulation:
    N = 4  # エージェントの数

    def __init__(self, Agent, Env, time=50, initial_p_green=83.55, initial_p_oil=13.64, initial_pv_green=180, initial_pv_oil=70, initial_fare=144.8):
        # 初期値
        random.seed(1001)
        self.time = time  # シミュレーション期間
        self.initial_p_green = initial_p_green # 最初のgreen燃料価格
        self.initial_p_oil = initial_p_oil  # 最初のoil燃料価格
        self.initial_pv_green = initial_pv_green  # 最初のgreen船の価格
        self.initial_pv_oil = initial_pv_oil  # 最初のoil船の価格
        self.initial_fare = initial_fare  # 最初の運賃

        # N個のエージェントを作成
        self.agents = [Agent(i) for i in range(self.N)]

        # 環境の作成
        self.env = Env(self.agents, self.initial_p_green, self.initial_p_oil, self.initial_pv_green,
                       self.initial_pv_oil, self.initial_fare)

        # 結果を保存するリスト
        self.years = []
        self.total_n_green_history = []
        self.total_n_oil_history = []
        self.total_n_history = []
        self.p_green_history = []
        self.p_oil_history = []
        self.pv_green_history = []
        self.pv_oil_history = []
        self.fare_history = []
        self.demand_history = []
        self.benefit_history = []
        self.agent_benefit_history = [[] for _ in range(self.N)]
        self.agent_avg_oil = []
        self.agent_avg_green = []

    def run(self):
        # シミュレーションの実行
        for year in range(2020, 2020 + self.time):
            self.years.append(year)

            for agent in self.agents:
                agent.trade(self.env, self.agents)

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

            # エージェントごとの利益を記録
            for i, agent in enumerate(self.agents):
                self.agent_benefit_history[i].append(agent.benefit)

            print("\r" + f"Year: {year}, Total Green Ships: {self.env.total_n_green}, Total Oil Ships: {self.env.total_n_oil}, Demand: {self.env.demand}, Fare: {self.env.fare}", end="")
        print("\n")

    def plot(self):
        # 結果の表示
        print("Year\tShips\tDemand\tFare")
        for year, total_n, demand, fare in zip(self.years, self.total_n_history, self.demand_history, self.fare_history):
            print(f"{year}\t{total_n}\t{demand:.2f}\t{fare:.2f}")

        # グラフのプロット
        plt.figure(figsize=(20, 15))
        agent_colors = ['red', 'blue', 'green', 'orange']  # エージェントごとの色

        # 1. 総green船とoil船の数の変化と各エージェントの予測
        plt.subplot(3, 2, 1)
        # 実際の推移をプロット
        plt.plot(self.years, self.total_n_green_history, label='Actual Green Ships', color='green', linewidth=2)
        plt.plot(self.years, self.total_n_oil_history, label='Actual Oil Ships', color='blue', linewidth=2)

        # 各エージェントの予測をプロット

        for i, agent in enumerate(self.agents):
            color_green = agent_colors[i % len(agent_colors)]
            color_oil = agent_colors[i % len(agent_colors)]

            # 各時点での予測をプロット
            for t, year in enumerate(self.years):
                if t < len(agent.history_predict_n_oil) and t < len(agent.history_predict_n_green):
                    # 予測の長さ（future_years）を取得
                    future_years = len(agent.history_predict_n_oil[t])
                    if future_years > 0:
                        # 予測データの時間軸を作成
                        future_x = [year + j for j in range(1, future_years + 1)]

                        # 油船の予測をプロット
                        plt.plot(future_x, agent.history_predict_n_oil[t],
                            color=color_oil, linestyle='--', alpha=0.3, linewidth=1)

                        # グリーン船の予測をプロット
                        plt.plot(future_x, agent.history_predict_n_green[t],
                            color=color_green, linestyle=':', alpha=0.3, linewidth=1)

        plt.title('Total Ships Over Time with Agent Predictions')
        plt.xlabel('Year')
        plt.ylabel('Number of Ships')
        plt.legend(['Actual Green Ships', 'Actual Oil Ships'])
        # plt.ylim(0, 10000)

        # 2. greenとoil燃料の価格の変化
        plt.subplot(3, 2, 2)
        plt.plot(self.years, self.p_green_history, label='Price of Green Fuel', color='green')
        plt.plot(self.years, self.p_oil_history, label='Price of Oil Fuel', color='blue')
        plt.title('Fuel Prices Over Time')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.legend()
        # plt.ylim(0, 100)

        # 3. greenとoil船の価格の変化
        plt.subplot(3, 2, 3)
        plt.plot(self.years, self.pv_green_history, label='Price of Green Ships', color='green')
        plt.plot(self.years, self.pv_oil_history, label='Price of Oil Ships', color='blue')
        plt.title('Ship Prices Over Time')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.legend()
        # plt.ylim(0, 190)

        # 4. 運賃と需要の変化 - 左右の軸を使用
        ax1 = plt.subplot(3, 2, 4)
        color = 'orange'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Fare', color=color)
        ax1.plot(self.years, self.fare_history, color=color, label='Fare')
        ax1.tick_params(axis='y', labelcolor=color)
        # ax1.set_ylim(0, 120)

        ax2 = ax1.twinx()  # 右側のy軸を作成
        color = 'red'
        ax2.set_ylabel('Demand', color=color)
        ax2.plot(self.years, self.demand_history, color=color, label='Demand')
        ax2.tick_params(axis='y', labelcolor=color)

        # 凡例の作成
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.title('Fare and Demand Over Time')

        # 5. 利益をエージェントごとに色分けして描画
        plt.subplot(3, 2, 5)

        for i, agent in enumerate(self.agents):
            plt.plot(self.years, self.agent_benefit_history[agent.ind], label=f'Agent {agent.ind}',
                    color=agent_colors[i], alpha=0.5)
        plt.title('Agent Benefits Over Time')
        plt.xlabel('Year')
        plt.ylabel('Benefit')
        plt.legend()

        # 6. 通常と特殊の平均船数の比較
        plt.subplot(3, 2, 6)
        plt.plot(self.years, self.agent_avg_oil, label='Average Oil Ships')
        plt.plot(self.years, self.agent_avg_green, label='Average Green Ships')
        plt.xlabel('Year')
        plt.ylabel('Average Number of Ships')
        plt.title('Average Number of Ships Over Time')
        plt.legend()
        # plt.ylim(0, 1200)

        # レイアウトの調整
        plt.tight_layout()
        
        # 画像として保存
        plt.savefig('results/simulation_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 結果の表示
        print("Final Total Green Ships:", self.env.total_n_green)
        print("Final Total Oil Ships:", self.env.total_n_oil)

if __name__ == '__main__':
    sim = Simulation(Agent, Env)
    sim.run()
    sim.plot()
    sim.validate()
