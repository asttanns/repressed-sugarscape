import math
from mesa.discrete_space import CellAgent


def get_distance(cell_1, cell_2):
    x1, y1 = cell_1.coordinate
    x2, y2 = cell_2.coordinate
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx**2 + dy**2)


class SugarAgent(CellAgent):
    def __init__(
        self,
        model,
        cell,
        sugar=0,
        metabolism=0,
        vision=0,
        # modification: new agent attributes for protest behavior
        protest_threshold=20,
        repression_tolerance=1.0,
    ):
        super().__init__(model)
        self.cell = cell
        self.sugar = sugar
        self.metabolism = metabolism
        self.vision = vision
        # protest-related attributes
        self.protest_threshold = protest_threshold
        self.repression_tolerance = repression_tolerance
        self.is_protesting = False

    def move(self):
        # agent's movement rule now considers repression
        # agents rank cells by sugar / (1 + repression), so they avoid
        # high-repression cells even when sugar is abundant.
        possibles = [
            cell
            for cell in self.cell.get_neighborhood(self.vision, include_center=True)
            if cell.is_empty
        ]

        # compute discounted score instead of raw sugar
        scores = [
            cell.sugar / (1.0 + self._cell_repression(cell))
            for cell in possibles
        ]
        max_score = max(scores)
        candidates_index = [
            i for i in range(len(scores)) if math.isclose(scores[i], max_score)
        ]
        candidates = [possibles[i] for i in candidates_index]

        min_dist = min(get_distance(self.cell, cell) for cell in candidates)
        final_candidates = [
            cell
            for cell in candidates
            if math.isclose(get_distance(self.cell, cell), min_dist, rel_tol=1e-02)
        ]
        self.cell = self.random.choice(final_candidates)

    # modification: helper method to read repression at a cell
    def _cell_repression(self, cell):
        x, y = cell.coordinate
        return self.model.grid.repression.data[x, y]

    def gather_and_eat(self):
        self.sugar += self.cell.sugar
        self.cell.sugar = 0
        self.sugar -= self.metabolism

    # modification: new method — decide whether to protest this step
    def decide_protest(self):
        """
        Protest if wealthy enough AND local repression is below personal tolerance.
        Protesting costs a small sugar amount (activist labor).
        """
        local_rep = self._cell_repression(self.cell)
        if self.sugar > self.protest_threshold and local_rep < self.repression_tolerance:
            self.is_protesting = True
            self.sugar -= self.model.protest_cost  # pay activist cost
        else:
            self.is_protesting = False

    def see_if_die(self):
        # modification: two modes of death.
        # 1. Starvation - based on original sugarsacape
        # 2. Lethal repression - if local repression exceeds lethality threshold
        if self.sugar <= 0:
            self.remove()
            return
        if self._cell_repression(self.cell) > self.model.lethality:
            self.remove()