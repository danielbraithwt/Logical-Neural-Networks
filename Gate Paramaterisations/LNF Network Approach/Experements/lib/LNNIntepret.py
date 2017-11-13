import numpy as np

def intepret_model(model, activations):
    influence_by_layer = []

    num_inputs = int(len(model[0][0])/2)
    print(num_inputs)
    
    first_hidden_layer = model[0]
    influences = []

    print(first_hidden_layer)
    print(len(first_hidden_layer))

    # Weights directly corospond to influence the input has
    for weights in first_hidden_layer:
        influences.append(weights[0:int(len(weights)/2)])

    influence_by_layer.append(influences)
    print(influence_by_layer)

    for l in range(1, len(model)):
        layer = model[l]
        layer_influences = []
        #weights = layer[0]

        # For each neuron in the layer
        for n in range(len(layer)):
            neuron = layer[n]
            print(len(neuron))
            #print(neuron)
            
            influence = np.zeros(num_inputs)
            for j in range(len(influence)):
                prod = 1
                for m in range(int(len(neuron)/2)):
                    if activations[l] == "OR":
                        prod = prod * np.power(neuron[m], influences[m][j])
                    elif activations[l] == "AND":
                        prod = prod * np.power(neuron[m], 1 - influences[m][j])

                if activations[l] == "OR":
                    influence[j] = 1 - prod
                else:
                    influence[j] = prod
            layer_influences.append(influence)


        influences = layer_influences
        influence_by_layer.append(influences)



    return influence_by_layer
