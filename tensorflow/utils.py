import numpy as np

def ctrl_bin(state, level):

        state_bin = ''
        i = state
        while i//2 != 0:
            if(i>3):
                state_bin = str(i%2)+state_bin
                i = i//2
            else:
                state_bin = str(i//2)+str(i%2)+state_bin
                i = i//2

        #if level > len(state_bin):
        i = level - len(state_bin) - 1

        if state//2 == 0 and level > len(state_bin):
             state_bin = str(state%2)

        for _ in range(level-len(state_bin)):
            state_bin = '0'+state_bin

        return state_bin

def initializer(vetor):
    num_qu = int(np.log2(len(vetor)))
    circuit = qiskit.QuantumCircuit(num_qu)

    norms = lambda v: np.sqrt(np.absolute(v[0::2])**2 + np.absolute(v[1::2])**2)
    select_alpha = lambda v,p,i: 2*np.arcsin(v[2*i + 1]/p[i]) if v[2*i]>0 else 2*np.pi - 2*np.arcsin(v[2*i + 1]/p[i])

    alphas = []
    parents = norms(vetor)
    alphas = np.append(alphas, np.array([ select_alpha(vetor,parents,i) for i in range(vetor.shape[0]//2)]))[::-1]

    for _ in range(int(np.log2(len(vetor)))-1):
        new_parents = norms(parents)
        alphas = np.append(alphas, np.array([ select_alpha(parents,new_parents,i) for i in range(parents.shape[0]//2)]))[::-1]
        parents = new_parents

    circuit.ry(alphas[0],[0])
    circuit.ry((alphas[1]+alphas[2])/2,[1])
    circuit.cnot(0, 1)
    circuit.ry((alphas[1]-alphas[2])/2,[1])
    circuit.cnot(0, 1)

    return circuit

def split_data(X, Y, val, k_index):
    train_data = np.delete(X, range(k_index*val,(k_index+1)*val), axis=0)
    train_target = np.delete(Y, range(k_index*val,(k_index+1)*val))
    val_data = X[k_index*val:(k_index+1)*val,:]
    val_target = Y[k_index*val:(k_index+1)*val]

    return train_data, train_target, val_data, val_target

def shuffle_data(X,Y):
    shuf = np.array(range(Y.shape[0]))
    np.random.shuffle(shuf)

    return X[shuf], Y[shuf]