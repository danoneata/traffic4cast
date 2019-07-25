from train import train


class Worker():
    def compute(self, config, budget):
        return train(model_type, config, max_epochs=budget)
