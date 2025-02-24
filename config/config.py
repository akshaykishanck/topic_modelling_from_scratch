import json

class Config:
    def __init__(self):
        with open("config/config.json", 'r') as f:
            config_data = json.load(f)
        self.main_data_path = config_data['main_data_path']
        self.plots_path = config_data['plots_path']
        self.runs = config_data['runs']
        self.number_of_iterations = config_data['number_of_iterations']
        self.dataset = config_data['dataset']
        self.hyper_parameters = config_data['hyper_parameters']
        self.number_of_topics, self.alpha, self.beta = self.get_hyper_parameters()

    def get_hyper_parameters(self):
        if self.dataset in self.hyper_parameters:
            params = self.hyper_parameters[self.dataset]
        else:
            params = self.hyper_parameters['default']
        return params['number_of_topics'], params['alpha'], params['beta']
