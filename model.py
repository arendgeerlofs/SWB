import mesa
from agent import Individual

class SWB_model(mesa.Model):
    "A model on SWB"

    def __init__(self, N):
        super().__init__()
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        
        # Create agents
        for i in range(self.num_agents):
            agent = Individual(i, self)
            self.schedule.add(agent)

    def step(self):
        # Next step in model
        self.schedule.step()

        # Add intervention method to individuals


        # Add intervention method to grid/place