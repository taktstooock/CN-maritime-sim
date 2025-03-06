import base

class CustomAgent(base.Agent):
    pass


class CustomEnv(base.Env):
    def update(self):
        super().update()
        self.feebate_rate *= 1 - 0.5


if __name__ == '__main__':
    sim = base.Simulation(CustomAgent, CustomEnv)
    sim.run()
    sim.plot()
    sim.validate()