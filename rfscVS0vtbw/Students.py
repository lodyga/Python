class Students:
    def __init__(self, name, major, dps):
        self.name = name
        self.major = major
        self.dps = dps

    def is_high(self):
        return self.dps >= 3.5