class Atom():
    def __init__(name):
        self.name = name

class And():
    def __init__(atoms):
        self.atoms = atoms

class Or():
    def __init__(atoms):
        self.atoms = atoms


def build_cnf(network):
    
