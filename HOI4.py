import numpy as np
import matplotlib.pyplot as plt

# Global Variables
CIV_COST = 10800
MIL_COST = 7200
INFRA_COST = 6000
MAX_CIV = 20
MAX_MIL = 20
MAX_INFRA = 10
MAX_SLOTS = 25
CIV_LIMIT = 15
CIV_SPEED = 5
TIME_LIMIT = 365 * 4


class State:
    def __init__(self, name: str, civs: int, mils: int, infra: int):
        self.name = name
        self.civs = civs
        self.mils = mils
        self.infra = infra
        # Progress towards to constructing another item
        self.civ_prog = 0
        self.mil_prog = 0
        self.infra_prog = 0

    def addProg(self, type: str, assigned_civs: int):
        if self.civs + self.mils < MAX_SLOTS:
            if type == "civ" and self.civs <= MAX_CIV:
                self.civ_prog += 5 * assigned_civs * (1 + self.infra / MAX_INFRA)
                if self.civ_prog >= CIV_COST:
                    self.civ_prog = 0
                    self.civs += 1
            elif type == "mil" and self.mils <= MAX_MIL:
                self.mil_prog += 5 * assigned_civs * (1 + self.infra / MAX_INFRA)
                if self.mil_prog > MIL_COST:
                    self.mil_cost = 0
                    self.mils += 1
        if type == "infra" and self.infra < MAX_INFRA:
            self.infra_prog += 5 * assigned_civs * (1 + self.infra / MAX_INFRA)
            if self.infra_prog > INFRA_COST:
                self.infra_prog = 0
                self.infra += 1


class Country(State):
    def __init__(self, name: str, civs: int, mils: int, infra: int):
        super().__init__(name, civs, mils, infra)
        self.name = name
        self.states = []
        self.civs = sum([st.civs for st in self.states])
        self.mils = sum([st.mils for st in self.states])
        self.civ_dist = []
        self.max_projects = len(self.civ_dist)
        # Keys: State, Entries: Type
        self.que = {st: [] for st in self.states}

    def addState(self, name: str, civs: int, mils: int, infra: int):
        self.states.append(State(name, civs, mils, infra))

    # Things to run at the beginning of each day
    def newDay(self):
        self.civs = sum([st.civs for st in self.states])
        if self.civs % CIV_LIMIT == 0:
            self.civ_dist = [CIV_LIMIT] * self.civs // CIV_LIMIT
        else:
            self.civ_dist = [CIV_LIMIT] * self.civs // CIV_LIMIT + [self.civs % CIV_LIMIT]
