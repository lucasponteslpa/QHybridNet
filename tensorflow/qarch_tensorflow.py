import cirq
import sympy
from tqdm import tqdm
import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.model_selection import train_test_split
import numpy as np
from utils import ctrl_bin, accuracy

class QInputCircuit():
    def __init__(self, vector, qubits):
        self.alphas = []
        self.qu = qubits
        self.v_state = vector
        self.n_qu = int(np.log2(len(self.v_state)))
        self.circuit = cirq.Circuit()

    def make_circuit(self):
        alphas = self.get_alphas()
        gate_op = cirq.ry(alphas[0][0])
        self.circuit.append([gate_op.on(self.qu[0])])
        for i, angles in enumerate(alphas[1:],1):
            self._load_multiplexer_angles(angles, 0, len(angles), False)
            self._total_multiplexer(angles, list(range(i+1)))
            self.circuit.append([cirq.CNOT(self.qu[0],self.qu[i])])

        return self.circuit

    def get_alphas(self):
        norms = lambda v: np.sqrt(np.absolute(v[0::2])**2 + np.absolute(v[1::2])**2)
        def select_alpha(v,p,i):
            if p[i] != 0.0:
                return 2*np.arcsin(v[2*i + 1]/p[i]) if v[2*i]>0 else 2*np.pi - 2*np.arcsin(v[2*i + 1]/p[i])
            else:
                return 0.0

        alphas = []
        parents = norms(self.v_state)
        alphas.append(np.array([ select_alpha(self.v_state,parents,i) for i in range(self.v_state.shape[0]//2)]))

        for _ in range(self.n_qu-1):
            new_parents = norms(parents)
            alphas.append( np.array([ select_alpha(parents,new_parents,i) for i in range(parents.shape[0]//2)]))
            parents = new_parents

        return alphas[::-1]


    def _load_multiplexer_angles(self, angles, start_index, end_index, reversed_dec):
        interval_len_half = (end_index - start_index) // 2
        for i in range(start_index, start_index + interval_len_half):
            if not reversed_dec:
                angles[i], angles[i + interval_len_half] = \
                    self._update_angles(
                        angles[i], angles[i + interval_len_half])
            else:
                angles[i + interval_len_half], angles[i] = \
                    self._update_angles(
                        angles[i], angles[i + interval_len_half])
        if interval_len_half <= 1:
            return
        else:
            self._load_multiplexer_angles(angles, start_index, start_index + interval_len_half,
                                             False)
            self._load_multiplexer_angles(angles, start_index + interval_len_half, end_index,
                                             True)

    def _update_angles(self, angle1, angle2):
        """Calculate the new rotation angles according to Shende's decomposition."""
        return (angle1 + angle2) / 2.0, (angle1 - angle2) / 2.0

    def _minimum_multiplexer(self, angles, control_index, rot_index):
        gate_ry0 = cirq.ry(angles[0])
        gate_ry1 = cirq.ry(angles[1])
        self.circuit.append([gate_ry0.on(self.qu[rot_index]),
                             cirq.CNOT(self.qu[control_index], self.qu[rot_index]),
                             gate_ry1.on(self.qu[rot_index])])

    def _total_multiplexer(self, angles, qu_indexes):
        if len(angles) == 2:
            self._minimum_multiplexer(angles, qu_indexes[0], qu_indexes[1])
        else:
            child_indexes = qu_indexes[1:] if len(qu_indexes) > 2 else qu_indexes
            left_angles = angles[:len(angles)//2]
            right_angles = angles[len(angles)//2:]

            self._total_multiplexer(left_angles, child_indexes)
            self.circuit.append([cirq.CNOT(self.qu[qu_indexes[0]], self.qu[qu_indexes[-1]])])
            self._total_multiplexer(right_angles, child_indexes)



class CircuitClassModelBuilder():
    def __init__(self, qubits, layer_type=0):
        self.qubits = qubits
        self.layer_type = layer_type
        self.circuit = cirq.Circuit()

    def add_layer(self, prefix):
        if self.layer_type == 0:
            for i in range(0, len(self.qubits)-1, 2):
                self.circuit.append([cirq.CNOT(self.qubits[i], self.qubits[i+1])])
            for i in range(1, len(self.qubits)-1, 2):
                self.circuit.append([cirq.CNOT(self.qubits[i], self.qubits[i+1])])
            for i, qubit in enumerate(self.qubits):
                symbol = sympy.Symbol(prefix + '_' + str(i))
                gate_ry = cirq.ry(symbol)
                self.circuit.append([gate_ry.on(qubit)])
        else:
            for i, qubit in enumerate(self.qubits):
                symbol_y = sympy.Symbol(prefix +'_eRy' + str(i))
                symbol_z = sympy.Symbol(prefix +'_eRz' + str(i))
                gate_ry = cirq.ry(symbol_y)
                gate_rz = cirq.rz(symbol_z)
                self.circuit.append([gate_ry.on(qubit), gate_rz.on(qubit)])
            for i in range(0, len(self.qubits)-1, 2):
                self.circuit.append([cirq.CNOT(self.qubits[i], self.qubits[i+1])])
            for i in range(1, len(self.qubits)-1, 2):
                symbol_y0 = sympy.Symbol(prefix +'_oRy' + str(i))
                symbol_z0 = sympy.Symbol(prefix +'_oRz' + str(i))
                symbol_y1 = sympy.Symbol(prefix +'_oRy' + str(i+1))
                symbol_z1 = sympy.Symbol(prefix +'_oRz' + str(i+1))
                gate_ry0 = cirq.ry(symbol_y0)
                gate_rz0 = cirq.rz(symbol_z0)
                gate_ry1 = cirq.ry(symbol_y1)
                gate_rz1 = cirq.rz(symbol_z1)
                self.circuit.append([gate_ry0.on(self.qubits[i]),gate_rz0.on(self.qubits[i]),
                                     gate_ry1.on(self.qubits[i+1]),gate_rz1.on(self.qubits[i+1]),
                                     cirq.CNOT(self.qubits[i], self.qubits[i+1])])

    def ctrl_gate(self, symbol, qubit, ctrl_qubit, gate='rz'):
        if gate == 'rz':
            gate_p = cirq.rz(symbol/2)
            gate_n = cirq.rz(-symbol/2)
            self.circuit.append([gate_p.on(qubit), cirq.CNOT(ctrl_qubit, qubit),
                                 gate_n.on(qubit), cirq.CNOT(ctrl_qubit, qubit)])






class QMCModel(CircuitClassModelBuilder):
    def __init__(self, qubits, n_layers, layer_type=0):
        super().__init__(qubits, layer_type=layer_type)
        self.n_layers = n_layers

        for l in range(self.n_layers):
            prefix = 'layer'+str(l)
            self.add_layer(prefix)

class HermitianLabels():
    def __init__(self, classes, n_qubits) -> None:
        self.classes = classes
        self.n_qubits = n_qubits
        self.base_hermitians = {'0':np.array([[1.0,0.0],
                                              [0.0,0.0]]),
                                '1':np.array([[0.0,0.0],
                                              [0.0,1.0]])}
        self.labels_encoding = self.load_label_encoding()
        self.label_hermitian, self.Ys = self.load_label_state(self.base_hermitians, self.labels_encoding)

    def load_label_state(self, hermitians, labels_encoding):
        l_states = {}
        Ys_coefs = {}
        for i, dict in enumerate(labels_encoding.items()):
            l, e = dict
            b = hermitians[e[0]]
            Ys = np.ones(len(labels_encoding))*1/4
            Ys[i] = 1.0
            for j in e[1:]: # e is a binary string
                b = np.kron(b,hermitians[j])
            Ys_coefs[l] = Ys
            l_states[l] = b

        return l_states, Ys_coefs

    def load_label_encoding(self):
        labels_encoding = {}
        for i,c in enumerate(self.classes):
            s = "{0:b}".format(i)
            for _ in range(self.n_qubits - len(s)):
                s = '0' + s
            labels_encoding[c] = s

        return labels_encoding



class QuantumInput(HermitianLabels):
    def __init__(self, train, val, classes, n_qubits, n_layers):
        super().__init__(classes, n_qubits)
        self.n_classes = len(classes)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        n = len(classes) * (n_qubits+1)

        self.qubits = [cirq.GridQubit(0,j) for j in range(n_qubits)]

        self.measurement = [cirq.Z(self.qubits[j]) for j in range(self.n_qubits)]


        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)
        self.X_train, self.X_test, self.y_train, self.y_test = train[0], val[0], train[1], val[1]

        quantum_input, quantum_input_labels = self.build_data_circuits(self.X_train, self.y_train,v=False)
        val_input, val_labels = self.build_data_circuits(self.X_test, self.y_test, len(classes))

        self.quantum_input_labels = tf.convert_to_tensor(np.array(quantum_input_labels))
        self.quantum_input_tensor = tfq.convert_to_tensor(quantum_input)

        self.quantum_val_labels = tf.convert_to_tensor(np.array(val_labels))
        self.quantum_val_tensor = tfq.convert_to_tensor(val_input)

        self.quantum_model_circuit = QMCModel(self.qubits, n_layers=self.n_layers, layer_type=1).circuit
        differentiator = tfq.differentiators.ParameterShift()
        self.expectation_layer = tfq.layers.PQC(self.quantum_model_circuit,
                                           operators=self.measurement,
                                           repetitions=5000,
                                           differentiator=differentiator)
        del quantum_input
        del val_input

    def build_data_circuits(self, X, y,v=True):
        quantum_input = []
        quantum_input_labels = []
        with tqdm(total=y.shape[0]) as t:
            for i, sample in enumerate(X):
                quantum_input_circuit = QInputCircuit(sample, self.qubits)
                quantum_circuit = quantum_input_circuit.make_circuit()
                quantum_input = quantum_input + [quantum_circuit]
                one_hot_label = np.zeros(self.n_classes)
                one_hot_label[int(y[i])] = 1.0
                quantum_input_labels.append(one_hot_label)
                t.update()
        # import pdb;pdb.set_trace()
        return quantum_input, quantum_input_labels


    def training(self, batch_size = 4, epochs=100):

        # TFQ differentiator
        #differentiator = tfq.differentiators.ParameterShift()

        # Quantum data input for the keras model
        q_data_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='circuits_input')

        #  Get expectation outut from Parametric Quantum Circuit layer
        # self.expectation_layer = tfq.layers.PQC(self.quantum_model_circuit,
        #                                    operators=self.measurement,
        #                                    repetitions=5000,
        #                                    differentiator=differentiator)

        self.model = tf.keras.Sequential([
            # The input is the data-circuit, encoded as a tf.string
            q_data_input,
            # The PQC layer returns the expected value of the readout gate, range [-1,1].
            self.expectation_layer,
            tf.keras.layers.Dense(10, activation="sigmoid")
        ])

        # Optimizer for update parameters of the 'quantum model'
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # Loss of the keras model
        # loss = tf.keras.losses.MeanSquaredError()

        # Configuring the model
        self.model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Training the model
        self.train = self.model.fit(x=self.quantum_input_tensor,
                                    y=self.quantum_input_labels,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    verbose=1,
                                    validation_data=(self.quantum_val_tensor,
                                                     self.quantum_val_labels))

    def validate_model(self):
        symbols_dic = {}
        alpha_params = self.model.weights[0].numpy()
        for k in range(self.n_classes):
            for i in range(self.n_qubits+1):
                symbols_dic['theta' + '_' + str(i)+str(k)] = alpha_params[k*(self.n_qubits+1)+i]
        results = []
        get_z_results = lambda v: np.array([v_i[0] if v_i[0] == 1 else -1 for v_i in v])
        #for i in range(0,len(self.quantum_input), self.n_classes):
        z_meas = [cirq.measure(zq, key='m'+str(i)) for i,zq in enumerate(self.readout)]
        simulator = cirq.Simulator()
        circuit_val = cirq.Circuit()
        for i in range(self.n_classes):
            circuit_val = circuit_val + self.quantum_input[i]
        resolver = cirq.ParamResolver(symbols_dic)
        circuit_val = circuit_val + self.quantum_model_circuit
        circuit_val.append(z_meas)
        result = simulator.run(circuit_val, resolver, repetitions=5000)
        res = {key:get_z_results(r) for key,r in result.measurements.items()}
        res = [m.sum()/m.shape[0] for _,m in res.items()]
        return res



