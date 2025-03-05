import random
import matplotlib.pyplot as plt
import numpy as np

class Agent:
    def __init__(self, i):
        self.n_green = random.randint(0, 5)  # green船の数
        self.n_oil = random.randint(80, 120)  # oil船の数
        self.ind = i  # 識別子
        self.benefit = 0  # 初期利益
        self.last_benefit = 0  # 前年の利益
        self.history_oil = [] #重油のログ
        self.history_green = [] #グリーンのログ

    def renew(self, env):  # 決算
        self.cost = self.n_green * (1/20*env.pv_green + env.p_green) + self.n_oil * (1/20*env.pv_oil + env.p_oil)  # 輸送コスト
        self.sales = (self.n_green + self.n_oil) * env.fare  # 売上
        self.last_benefit = self.benefit  # 前年の利益を更新
        self.benefit = self.sales - self.cost  # 利益

    def trade(self, env, other_agents):  # 売買
        avg_green = sum(agent.n_green for agent in other_agents) / len(other_agents)
        avg_oil = sum(agent.n_oil for agent in other_agents) / len(other_agents)
        n_green_old = self.n_green
        n_oil_old = self.n_oil
        
        # Rule 3: 燃料価格に基づくエコ戦略
        if env.p_green <= 80 and env.p_oil >= 40:
            self.n_green += random.randint(1, 10)
            self.n_oil = max(0, self.n_oil - random.randint(1, 5))

        if env.p_green <= 80 and env.p_oil < 40:
            self.n_green += random.randint(1, 10)
            self.n_oil += random.randint(1, 10)

        if env.p_green > 80 and env.p_oil < 40:
            self.n_green = max(0, self.n_green - random.randint(1, 5))
            self.n_oil += random.randint(1, 10)

        if env.p_green > 80 and env.p_oil >= 40:
            self.n_green = max(0, self.n_green - random.randint(1, 5))
            self.n_oil = max(0, self.n_oil - random.randint(1, 5))

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

            #totl_hist = （あるエージェントが直近1,3,5年で買った船の数の合計）
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
    
    def predict_n_feature(self, env, agents, past_years, feature_years):
        """
        過去past_years年間の特徴量を用いて、未来feature_years年後の船の数を予測する
        """
        predict_sum_n_oils = np.zeros(feature_years)
        predict_sum_n_greens = np.zeros(feature_years)

        # 自分以外のAgentが買う重油船の数を予測
        for agent in agents:
            # もし自分であればスキップ
            if agent.ind == self.ind:
                continue
            
            new_buy_oils = np.zeros(feature_years)
            new_buy_greens = np.zeros(feature_years)
            for i in range(feature_years):
                # 過去past_years年間の特徴量を取得
                year = len(agent.history_green) + i
                if len(agent.history_green) == 0:
                    if i == 0:
                        new_buy_oil = random.randint(0, 5)
                        new_buy_green = random.randint(0, 5)
                    else:
                        pass
                else:
                    new_buy_oil = (sum(agent.history_oil[-(past_years-i):]) + sum(new_buy_oils)) / past_years
                    new_buy_green = (sum(agent.history_green[-(past_years-i):]) + sum(new_buy_greens)) / past_years
                new_buy_oils[i] = new_buy_oil
                new_buy_greens[i] = new_buy_green
            
            predict_sum_n_oils += new_buy_oils
            predict_sum_n_greens += new_buy_greens
        
        return env.total_n_oil + predict_sum_n_oils, env.total_n_green + predict_sum_n_greens

    def predict_feebate(self, env, n_oil, n_green, self_n_oil, self_n_green, feebate_rate=None):
        feebate_rate = env.feebate_rate if feebate_rate is None else feebate_rate
        if n_oil == 0:
            return 0, self_n_green * (env.p_green - env.p_oil * feebate_rate)
        if n_oil != 0:
               penalty = self_n_oil * (env.p_green - env.p_oil * feebate_rate) * n_green / n_oil
               rebate = self_n_green * (env.p_green - env.p_oil * feebate_rate)
               return penalty ,rebate
    
    def predict_feebate_future(self, env, n_oils :np.ndarray, n_greens :np.ndarray, self_n_oil, self_n_green, future_years):
        penalties = np.zeros(future_years)
        rebates = np.zeros(future_years)
        for i in range(future_years):
            penalty, rebate = self.predict_feebate(env, n_oils[i], n_greens[i], self_n_oil, self_n_green)
            penalties[i] = penalty
            rebates[i] = rebate
        return penalties, rebates


class Env:
    def __init__(self, agents, initial_p_green, initial_p_oil, initial_pv_green, initial_pv_oil, initial_fare, initial_feebate_rate, fee_change_rate=0.1):
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
        self.feebate_rate = initial_feebate_rate
        self.feebate_change_rate = fee_change_rate
        self.update()

    def cal_total_n_oil(self):
        return sum(agent.n_oil for agent in self.agents)

    def cal_total_n_green(self):
        return sum(agent.n_green for agent in self.agents)

    def market(self):
        self.p_green = self.p_green * (1 + random.uniform(-0.05, 0))
        self.p_oil = self.p_oil * (1 + random.uniform(0, 0.05))
        self.pv_oil = self.p_oil * 10
        self.pv_green = self.p_green * 8

    def feebate(self):
        feebate_rate = self.feebate_rate

        sum_n_oil = self.cal_total_n_oil()
        sum_n_green = self.cal_total_n_green()
        
        penalty_rate = (self.p_green - self.p_oil * feebate_rate) * sum_n_green / sum_n_oil
        rebate_rate = self.p_green - self.p_oil * feebate_rate

        if penalty_rate < 0 or rebate_rate < 0:
            pass

        for agent in self.agents:
            penalty = agent.n_oil * penalty_rate
            agent.benefit -= penalty

        for agent in self.agents:
            rebate = agent.n_green * rebate_rate
            agent.benefit += rebate

    def update(self):
        self.total_n_green = self.cal_total_n_green()
        self.total_n_oil = self.cal_total_n_oil()
        self.total_n = self.total_n_green + self.total_n_oil
        self.demand *= 1.025  # 毎年2.5%増加
        self.fare = self.demand / self.total_n
        self.feebate_rate *= 1 - self.feebate_change_rate
        self.market()
        self.feebate()


class Simulation:
    N = 4  # エージェントの数

    def __init__(self, Agent, Env):
        # 初期値
        random.seed(100)
        self.time = 50  # シミュレーション期間
        self.initial_p_green = 100  # 最初のgreen燃料価格
        self.initial_p_oil = 20  # 最初のoil燃料価格
        self.initial_pv_green = 300  # 最初のgreen船の価格
        self.initial_pv_oil = 200  # 最初のoil船の価格
        self.initial_fare = 100  # 最初の運賃
        self.initial_feebate_rate = 1  # フィーベイト率

        self.FEEBATE_CHANGE_RATE = 0.1  # フィーベイト率の変化率
        self.DISCOUNT_RATE = 0  # 割引率

        # N人のエージェントを作成
        self.agents = [Agent(i) for i in range(self.N)]

        # 環境の作成
        self.env = Env(self.agents, self.initial_p_green, self.initial_p_oil, self.initial_pv_green,
                       self.initial_pv_oil, self.initial_fare, self.initial_feebate_rate)

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
                agent.renew(self.env)

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

        # 1. 総green船とoil船の数の変化
        plt.subplot(3, 2, 1)
        plt.plot(self.years, self.total_n_green_history, label='Total Green Ships', color='green')
        plt.plot(self.years, self.total_n_oil_history, label='Total Oil Ships', color='blue')
        plt.title('Total Ships Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Ships')
        plt.legend()

        # 2. greenとoil燃料の価格の変化
        plt.subplot(3, 2, 2)
        plt.plot(self.years, self.p_green_history, label='Price of Green Fuel', color='green')
        plt.plot(self.years, self.p_oil_history, label='Price of Oil Fuel', color='blue')
        plt.title('Fuel Prices Over Time')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.legend()

        # 3. greenとoil船の価格の変化
        plt.subplot(3, 2, 3)
        plt.plot(self.years, self.pv_green_history, label='Price of Green Ships', color='green')
        plt.plot(self.years, self.pv_oil_history, label='Price of Oil Ships', color='blue')
        plt.title('Ship Prices Over Time')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.legend()

        # 4. 運賃と需要の変化 - 左右の軸を使用
        ax1 = plt.subplot(3, 2, 4)
        color = 'orange'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Fare', color=color)
        ax1.plot(self.years, self.fare_history, color=color, label='Fare')
        ax1.tick_params(axis='y', labelcolor=color)

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
        rule_colors = ['red', 'blue', 'green', 'orange']

        for i, agent in enumerate(self.agents):
            plt.plot(self.years, self.agent_benefit_history[agent.ind], label=f'Agent {agent.ind}',
                    color=rule_colors[i], alpha=0.5)
        plt.title('Agent Benefits Over Time')
        plt.xlabel('Year')
        plt.ylabel('Benefit')
        plt.legend()

        # 6. 通常と特殊の平均船数の比較
        plt.subplot(3, 2, 6)
        plt.plot(self.years, self.agent_avg_oil, label='Average Oil Ships (Normal)')
        plt.plot(self.years, self.agent_avg_green, label='Average Green Ships (Normal)')
        plt.xlabel('Year')
        plt.ylabel('Average Number of Ships')
        plt.title('Average Number of Ships Over Time')
        plt.legend()

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