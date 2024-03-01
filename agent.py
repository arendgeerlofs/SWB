import mesa
import numpy as np

class Individual(mesa.Agent):
    "An individual with traits and characteristics with a SWB score"

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Traits
        self.extraversion = np.random.randint(1, 6)
        self.neuroticism = np.random.randint(1, 6)
        self.openness = np.random.randint(1, 6)
        self.conscientiesness = np.random.randint(1, 6)
        self.agreeableness = np.random.randint(1, 6)

        # Adaptation properties
        self.habituation = np.random.randint(1, 6)
        self.sensitisation = np.random.randint(1, 6)
        self.desensitisation = np.random.randint(1, 6)

        # Characteristics
        self.financial = np.random.randint(1, 11)
        self.health = np.random.randint(1, 11)
        self.social = np.random.randint(1, 11)
        self.religion = np.random.randint(1, 11)
        
        # SWB
        self.SWB = min(max(np.random.normal(6.5, 1.5), 0), 10)

    def step(self):
        print(f"SWB: {str(self.SWB)}.")

        # Change social circle based on traits and probability


        # Change grid location based on location and characteristics


        # Let some event occur based on some probability


        # Calculate SWB level based on stimuli difference and adaptation


