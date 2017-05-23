import numpy as np

def relevance_pruning(network, threshold):
    hidden = network[0]
    output = network[1]

    pruned_hidden = []
    pruned_out = []

    pruned_out.append(output[0])

    for i in range(1, len(output)):
        hidden_node = hidden[i-1]
        node = np.zeros(len(hidden_node))
        if output[i] < threshold:
            pruned_hidden.append(node)
            pruned_out.append(0.0)
        else:
            node[0] = hidden_node[0]
            for j in range(1, len(hidden_node)):
                if hidden_node[j] >= threshold:
                    node[j] = hidden_node[j]
                
            pruned_hidden.append(node)
            pruned_out.append(output[i])

    return (pruned_hidden, pruned_out)


