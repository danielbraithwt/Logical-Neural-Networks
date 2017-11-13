import numpy as np

class OR:
    def __init__(self, weights):
        self.weights = weights

    def set_children(self, children):
        self.children = children
    
    def p(self):
        return np.product([c.p() for c in self.children])


class AND:
    def __init__(self, weights):
        self.weights = weights

    def set_children(self, children):
        self.children = children

class Out:
    def __init__(self, uid):
        self.id = uid

class Feature:
    def __init__(self, uid):
        self.id = uid

    def set_children(self, children):
        self.children = children

def intepret_model(model, activations):

    pgm_layers = []

    prev_layer = []
    for i in range(len(model[0])):
        prev_layer.append(Feature(i))


    inputs = prev_layer
    pgm_layers.append(prev_layer)
                   
    ## Build PGM ##
    for l in range(len(model)):
        probabilities = np.transpose(np.array(model[l]))
        print(activations[l])

        layer = []

        
        for n in range(len(probabilities)-1):
            
            if activations[l] == "AND":
                layer.append(AND(probabilities[n]))
            else:
                layer.append(OR(probabilities[n]))

        for node in prev_layer:
            node.set_children(layer)
        
        prev_layer = layer
        pgm_layers.append(prev_layer)

    
    outputs = []
    for i in range(len(model[len(model)-1])):
        outputs.append(Out(i))

    for node in prev_layer:
            node.set_children(outputs)

    pgm_layers.append(outputs)

    pgm_layers = np.array(pgm_layers)
    pgm_layers = np.flip(pgm_layers, 0)
    ## Finish Building PGM ##
