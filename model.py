from pathlib import Path

import numpy as np

import mesa
from agents import SugarAgent
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer


class SugarScapeModel(mesa.Model):
    def calc_gini(self):
        agent_sugars = [a.sugar for a in self.agents]
        if not agent_sugars or sum(agent_sugars) == 0:
            return 0
        sorted_sugars = sorted(agent_sugars)
        n = len(sorted_sugars)
        x = sum(el * (n - ind) for ind, el in enumerate(sorted_sugars)) / (n * sum(sorted_sugars))
        return 1 + (1 / n) - 2 * x

    # modification: reporters for the new repression dynamics
    def calc_mean_repression(self):
        return float(np.mean(self.grid.repression.data))

    def count_protesters(self):
        return sum(1 for a in self.agents if a.is_protesting)

    def count_population(self):
        return len(self.agents)

    def __init__(
        self,
        width=50,
        height=50,
        initial_population=200,
        endowment_min=25,
        endowment_max=50,
        metabolism_min=1,
        metabolism_max=5,
        vision_min=1,
        vision_max=5,
        # new parameters for repression + protest dynamics
        lethality=4.0,
        routineness=0.3,
        repression_increment=0.5,
        repression_decay=0.05,
        protest_cost=0.5,
        visibility_threshold=30,
        protest_threshold_min=15,
        protest_threshold_max=35,
        repression_tolerance_min=0.5,
        repression_tolerance_max=2.5,
        seed=None,
    ):
        if seed is not None:
            seed = int(seed)
        super().__init__(rng=seed)

        self.width = width
        self.height = height
        self.running = True

        # modification: store repression-regime parameters
        self.lethality = lethality
        self.routineness = routineness
        self.repression_increment = repression_increment
        self.repression_decay = repression_decay
        self.protest_cost = protest_cost
        self.visibility_threshold = visibility_threshold

        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Gini": self.calc_gini,
                "MeanRepression": self.calc_mean_repression,
                "Protesters": self.count_protesters,
                "Population": self.count_population,
            },
        )

        # Sugar landscape
        self.sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        self.grid.add_property_layer(
            PropertyLayer.from_data("sugar", self.sugar_distribution)
        )

        # repression landscape as second property layer
        self.grid.add_property_layer(
            PropertyLayer.from_data(
                "repression", np.zeros_like(self.sugar_distribution)
            )
        )

        # modification: heterogeneous protest thresholds and tolerances
        protest_thresholds = self.rng.uniform(
            protest_threshold_min, protest_threshold_max, initial_population
        )
        repression_tolerances = self.rng.uniform(
            repression_tolerance_min, repression_tolerance_max, initial_population
        )

        SugarAgent.create_agents(
            self,
            initial_population,
            self.random.choices(self.grid.all_cells.cells, k=initial_population),
            sugar=self.rng.integers(
                endowment_min, endowment_max, (initial_population,), endpoint=True
            ),
            metabolism=self.rng.integers(
                metabolism_min, metabolism_max, (initial_population,), endpoint=True
            ),
            vision=self.rng.integers(
                vision_min, vision_max, (initial_population,), endpoint=True
            ),
            # MODIFICATION: pass heterogeneous protest attributes
            protest_threshold=protest_thresholds,
            repression_tolerance=repression_tolerances,
        )
        self.datacollector.collect(self)

    # applies repression to the landscape
    def apply_repression(self):
        """
        Two components of state repression, mixed by `routineness`:
        - Targeted: pressure on cells where agents are protesting now.
        - Routine: pressure on cells where wealthy agents exist (visibility).
        Then apply exponential decay.
        """
        pressure_map = np.zeros_like(self.grid.repression.data)
        targeted_weight = 1.0 - self.routineness
        routine_weight = self.routineness

        for agent in self.agents:
            x, y = agent.cell.coordinate
            # Targeted: response to visible protest
            if agent.is_protesting:
                pressure_map[x, y] += self.repression_increment * targeted_weight
            # Routine: preemptive surveillance of wealthy agents
            if agent.sugar > self.visibility_threshold:
                pressure_map[x, y] += self.repression_increment * routine_weight

        # apply decay, then add new pressure
        self.grid.repression.data = (
            self.grid.repression.data * (1 - self.repression_decay) + pressure_map
        )

    def step(self):
        # sugar regrowth +1 (constant 1 per step, capped at capacity)
        self.grid.sugar.data = np.minimum(
            self.grid.sugar.data + 1, self.sugar_distribution
        )

        # agents move, harvest, decide whether to protest, check survival
        self.agents.shuffle_do("move")
        self.agents.shuffle_do("gather_and_eat")
        self.agents.shuffle_do("decide_protest")  # protest decision
        self.agents.shuffle_do("see_if_die")

        # state acts after agents
        self.apply_repression()

        self.datacollector.collect(self)