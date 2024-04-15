from qiskit import*
from qiskit_ibm_runtime import*
import numpy as np
import matplotlib.pyplot as plt
import ast
import qiskit_aer as Aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import pyscf as HF
import scipy.integrate as sc
from pyscf import gto, scf, ao2mo
from pyscf.tools import cubegen
from qiskit_aer.primitives import Sampler as ASampler
from qiskit.circuit import QuantumCircuit, Qubit, Clbit
import copy
import scipy.interpolate as interp

class MyVQE:
    def __init__(self, atoms, coordinates, spherical=False, basis='cc-pVDZ', spin=0):
        self.m = 1 # iteration number
        self.molecule = MyMolecule(atoms, coordinates, spherical=spherical, basis=basis, spin=spin)
        self.initial_ansatz = MyAnsatz(self.molecule)
        self.ansatz = MyAnsatz(self.molecule)
        self.currentEnergy = self.ansatz.calculateEnergy()
        self.param_excites = [] # List of tuples with [0] a parameter value and [1] an excitation ([p, q, r, s]) -- These are the excitations to construct the currently optimized ansatz

        single_excitations = []
        double_excitations = []
        for i in range(self.molecule.noOfMOs):
            for j in range(i+1, self.molecule.noOfMOs):
                single_excitations.append([i, j])
                for k in range(j+1, self.molecule.noOfMOs):
                    for l in range(k+1, self.molecule.noOfMOs):
                        if i == j and i == k and i == l:
                            break;
                        double_excitations.append([i, j, k, l])
                        double_excitations.append([i, k, j, l])
                        double_excitations.append([i, l, j, k])
                        
        self.excitation_pool = (single_excitations, double_excitations)
        print(f"excitation pool: {self.excitation_pool}")

    def run(self, threshold, max_iterations):
        print("Welcome to MyVQE!")
        while True:
            best_candidate, newAnsatz = self.iterate(dtheta=0.01, n=3)
            if best_candidate == None:
                print("VQE has finished! There are no relevant excitations to attempt to optimize")
                break;
            energy_reduction = best_candidate[0]
            new_param_excites = best_candidate[1]
            if energy_reduction < threshold:
                print("MyVQE has finished! Desired accuracy level was achieved")
                break;
            elif self.m >= max_iterations:
                print("Maximum number of VQE iterations were reached. Stopping...")
                break;
            else:
                self.ansatz = newAnsatz
                self.param_excites = new_param_excites
                self.currentEnergy -= energy_reduction
                print("VQE starting new iteration...")



    def iterate(self, dtheta, n): # dtheta difference quotient interval size, n is size of the "largest gradients list".
        grad_threshold = -dtheta
        n_gradients = [(0, [0, 0, 0, 0])] #(0, [0, 0, 0, 0]) These are all gradients and their corresponding excitations #We initialize it as one zero element, since we are looking for negative gradients anyway, and the list needs to have a non-zero initial length
        noOfCalculatedSingleGradients = 0
        noOfCalculatedDoubleGradients = 0
        for excite in self.excitation_pool[0]:
            noOfCalculatedSingleGradients += 1
            print(f"Calculating energy differentials for single electron excitations.. Currently at {noOfCalculatedSingleGradients} out of {len(self.excitation_pool[0])} single electrone excitations in the excitation pool.")
            newGrad = (self.ansatz.calculateDifferential(0, excite)[0], excite) # We look for gradients at theta=0
            for i in range(len(n_gradients)):
                if newGrad[0] < grad_threshold and (newGrad[0] <= n_gradients[i][0] or len(n_gradients) < n):
                    if len(n_gradients) >= n:
                        n_gradients[i] = newGrad
                        break;
                    else:
                        n_gradients.append(newGrad)
                        break;
        for excite in self.excitation_pool[1]:
            noOfCalculatedDoubleGradients += 1
            print(f"Calculating energy differentials for double electron excitations.. Currently at {noOfCalculatedDoubleGradients} out of {len(self.excitation_pool[1])} double electron excitations in the excitation pool.")
            newGrad = (self.ansatz.calculateDifferential(0, excite)[0], excite)
            for i in range(len(n_gradients)):
                if newGrad[0] < grad_threshold and (newGrad[0] <= n_gradients[i][0] or len(n_gradients) < n):
                    if len(n_gradients) >= n:
                        n_gradients[i] = newGrad
                        break;
                    else:
                        n_gradients.append(newGrad)
                        break;
        if n_gradients[0] == (0, [0, 0, 0, 0]): del n_gradients[0]

        print(f"Excitation pool found!: {n_gradients}")
        n_candidates = [] # This will hold all potential evolutions that lead to the maximum energy reduction. It holds tuple that includes an energy reduction ([0]), and a param_excites ([1]), which is a list of parametrized excitations
        for grad in n_gradients:
            excite = grad[1] # This is the corresponding excitation indexes (ij or ijkl) to the energy gradient
            if len(excite) == 4:
                print(f"Optimizing double electron excitation {excite}...")
                #Double
                new_param_excite = self.param_excites.copy()
                new_param_excite.append((0, excite))
                opt = MyOptimizer(self.initial_ansatz, new_param_excite)
                candidate = opt.run(threshold=dtheta)
                cand_ans = self.ansatz.copyMyAnsatz()
                cand_ans.apply_param_excites(candidate)
                n_candidates.append((self.currentEnergy - cand_ans.calculateEnergy(), candidate))
            elif len(excite) == 2:
                print(f"Optimizing single electron excitation {excite}...")
                #Single
                new_param_excite = self.param_excites.copy()
                new_param_excite.append((0, excite))
                opt = MyOptimizer(self.initial_ansatz, new_param_excite)
                candidate = opt.run(threshold=dtheta)
                cand_ans = self.ansatz.copyMyAnsatz()
                cand_ans.apply_param_excites(candidate)
                n_candidates.append((self.currentEnergy - cand_ans.calculateEnergy(), candidate))
            else:
                print("ISSUES!")
                raise ValueError(f"Current excitation has length {len(excite)}, which is neither 4 nor 2")

        best_candidate = None
        #if len(n_candidates) < n:
            #raise ValueError(f"FEJL! Længden af candidates er for lille ({len(n_candidates)})")
        for cand in n_candidates:
            if cand == None: raise ValueError("FEJL! Aktuelle kandidat er en NONE!");
            if best_candidate == None or cand[0] > best_candidate[0]:
                best_candidate = cand
        
        if best_candidate != None:
            energy_reduction = best_candidate[0]
            newAnsatz = self.ansatz.copyMyAnsatz()
            newAnsatz.apply_param_excites(best_candidate[1])
            self.m += 1
            return best_candidate, newAnsatz
        
        return best_candidate, self.ansatz
        
class MyOptimizer: # This class takes an ansatz and a set of parametrized excitations which are to be optimized
    def __init__(self, ansatz, param_excites):
        self.ansatz = ansatz
        self.param_excites = param_excites


    def run(self, learningRate=0.2, threshold=1e-5):
        converged = False
        while not converged:
            print(f"Current parameters: {self.param_excites}")
            newGradient = self.ansatz.calculateGradient(self.param_excites)
            print(f"Current gradient: {newGradient}")
            new_param_excites = []
            for param_excite, grad in zip(self.param_excites, newGradient):
                param_excite = (param_excite[0]-learningRate*grad, param_excite[1])
                new_param_excites.append(param_excite)
            self.param_excites = new_param_excites 

            gradientSize = 0
            for diff in newGradient:
                gradientSize += np.abs(diff)
            if gradientSize < threshold:
                converged = True
        print(f"Optimization complete! Final set of parameters: {self.param_excites}")
        return self.param_excites


class MyMolecule:
    def __init__(self, atoms, coordinates, spherical=False, basis='cc-pVDZ', spin=0):
        self.nuclearEnergy = self.getNuclearEnergy(atoms=atoms, coordinates=coordinates, spherical=spherical)
        self.mol = gto.Mole()
        self.mol.unit = 'B'
        self.mol.build(atom=MyMolecule.build_atom_string(atoms, coordinates, spherical=spherical), basis=basis, spin=spin)
        self.noOfElectrons = 0
        for atom in atoms:
            self.noOfElectrons += MyMolecule.getAtomNumber(atom)
        self.noOfOccMOs = self.noOfElectrons
        self.mf = scf.UHF(self.mol).run()
        #self.analyze = self.mf.analyze()
        self.fock = self.mf.get_fock()
        self.fockEnergy = self.mf.kernel()
        self.MO_coeff = self.mf.mo_coeff
        self.noOfMOs = len(self.MO_coeff[0])*2
        self.sei = self.mf.get_hcore() # Single electron integrals
        self.dei = self.mol.intor('int2e', aosym=1) # Double Electron Integrals
        self.Gh_pq = [] # Final single electron hamiltonian matrix elements
        self.Gh_pqrs = [] # Final double electron hamiltonian matrix elements
        self.Nh_pqrs = []
        self.Nh_pq = []


        MOsei = np.einsum('pi,pq,qj->ij', self.MO_coeff[0], self.sei, self.MO_coeff[0])

        #We also want to include our unrestricted states, ie. include all spin-orbitals:
        seiIndexes = []
        for p in range(self.noOfMOs):
            qs = []
            for q in range(self.noOfMOs):
                qs.append(0)
            self.Gh_pq.append(qs.copy())
            #self.Nh_pq.append(qs.copy())
        noOfSei = 0
        for i in range(int(self.noOfMOs/2)):
            for j in range(int(self.noOfMOs/2)):
                self.Gh_pq[2*i][2*j] = MOsei[i][j]
                self.Gh_pq[2*i+1][2*j] = MOsei[i][j]
                self.Gh_pq[2*i][2*j+1] = MOsei[i][j]
                self.Gh_pq[2*i+1][2*j+1] = MOsei[i][j]
                if MOsei[i][j] != 0:
                    noOfSei += 1
        for p in range(self.noOfMOs):
            for q in range(self.noOfMOs):
                if p%2 != q %2:
                    self.Gh_pq[p][q] = 0;
                if self.Gh_pq[p][q] != 0:
                    seiIndexes.append([p, q])

        MOdei = ao2mo.incore.full(self.dei, self.MO_coeff[0])
        #MOdei = MOdei.reshape(int(self.noOfMOs/2), int(self.noOfMOs/2), int(self.noOfMOs/2), int(self.noOfMOs/2))

        #Now, to expand to our spin orbitals:
        self.deiIndexes = []
        for p in range(self.noOfMOs):
            qs = []
            for q in range(self.noOfMOs):
                rs = []
                for r in range(self.noOfMOs):
                    ss = []
                    for s in range(self.noOfMOs):
                        ss.append(0)
                    rs.append(ss)
                qs.append(rs)
            self.Gh_pqrs.append(qs.copy())
            #self.Nh_pqrs.append(qs.copy())

        noOfDei = 0
        for p in range(int(self.noOfMOs/2)):
            for q in range(int(self.noOfMOs/2)):
                for r in range(int(self.noOfMOs/2)):
                    for s in range(int(self.noOfMOs/2)):
                        self.Gh_pqrs[2*p][2*q][2*r][2*s] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p][2*q][2*r][2*s+1] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p][2*q][2*r+1][2*s] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p][2*q+1][2*r][2*s] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p+1][2*q][2*r][2*s] = MOdei[p][q][r][s]

                        self.Gh_pqrs[2*p][2*q][2*r+1][2*s+1] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p][2*q+1][2*r+1][2*s] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p+1][2*q+1][2*r][2*s] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p][2*q+1][2*r][2*s+1] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p+1][2*q][2*r+1][2*s] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p+1][2*q][2*r][2*s+1] = MOdei[p][q][r][s]

                        self.Gh_pqrs[2*p][2*q+1][2*r+1][2*s+1] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p+1][2*q][2*r+1][2*s+1] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p+1][2*q+1][2*r][2*s+1] = MOdei[p][q][r][s]
                        self.Gh_pqrs[2*p+1][2*q+1][2*r+1][2*s] = MOdei[p][q][r][s]

                        self.Gh_pqrs[2*p+1][2*q+1][2*r+1][2*s+1] = MOdei[p][q][r][s]

                        if MOdei[p][q][r][s] != 0:
                            noOfDei += 1

        self.Nh_pq = copy.deepcopy(self.Gh_pq)
        self.Nh_pqrs = copy.deepcopy(self.Gh_pqrs)
        #Now we translate from the Chemist notation of PySCF to physicist notation
                            # Essentially, when we write Nh_pqrs[p][q][r][s], we want Gh[p][s][q][r]
        for p in range(self.noOfMOs):
            for q in range(self.noOfMOs):
                for r in range(self.noOfMOs):
                    for s in range(self.noOfMOs):
                        a = self.get_psqr(p, q, r, s)
                        self.Nh_pqrs[p][q][r][s] = a
 


        #ONE IMPORTANT LINE HERE ------- We actually make sure of spin orthogonality in the following loop
        for p in range(self.noOfMOs):
            for q in range(self.noOfMOs):
                for r in range(self.noOfMOs):
                    for s in range(self.noOfMOs):
                        spin_p = p%2
                        spin_q = q%2
                        spin_r = r%2
                        spin_s = s%2
                        if spin_p != spin_s or spin_q != spin_r:
                            self.Nh_pqrs[p][q][r][s] = 0;
                        #if self.Gh_pqrs[p][q][r][s] != 0:
                        #    self.deiIndexes.append([p, q, r, s])
    
    def get_psqr(self, p, q, r, s):
        return self.Gh_pqrs[p][s][q][r];

    def build_atom_string(atoms, coordinates, spherical):
        str = ""
        for atom, coor, i in zip(atoms, coordinates, range(len(atoms))):
            c = coor.split();
            if spherical == True:
                cartesian = MyMath.convert_to_cartesian([float(c[0]), float(c[1]), float(c[2])])
                c[0] = cartesian[0]
                c[1] = cartesian[1]
                c[2] = cartesian[2]
            if i != len(atoms)-1:
                str += f"{atom} {c[0]} {c[1]} {c[2]}; "
            else:
                str += f"{atom} {c[0]} {c[1]} {c[2]}"
        return str
    
    def getAtomNumber(atom):
        periodicTableDict = {"H" : 1, "He" : 2, "Li" : 3, "Be" : 4, "B" : 5, "C" : 6, "N" : 7, "O" : 8, "F" : 9, "Ne" : 10, "Na" : 11, "Mg" : 12, "Al" : 13, "Si" : 14, "P" : 15, "S" : 16, "Cl" : 17, "Ar" : 18}
        return periodicTableDict[atom]
    
    def getNuclearEnergy(self, atoms, coordinates, spherical):
        E = 0;
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                a_i = atoms[i]
                a_j = atoms[j]
                coor_i = coordinates[i].split();
                point_i = [float(coor_i[0]), float(coor_i[1]), float(coor_i[2])]
                coor_j = coordinates[j].split();
                point_j = [float(coor_j[0]), float(coor_j[1]), float(coor_j[2])]
                E += MyMolecule.getAtomNumber(a_i)*MyMolecule.getAtomNumber(a_j)/(MyMath.find_distance(point_i, point_j, spherical=spherical))
        return E


class MyCircuit:
    def __init__(self, N, qiskit_circuit=None, isMeasured=False):
        self.N = N
        if qiskit_circuit == None:
            self.qc = QuantumCircuit(N)
        else:
            self.qc = qiskit_circuit.copy()
        self.isMeasured = isMeasured
        
    def getCircuit(self):
        return self.qc

    def drawCircuit(self):
        self.qc.draw()
    
    def copyMyCircuit(self):
        return MyCircuit(self.N, self.qc.copy(), isMeasured=self.isMeasured)
    
    def hasMeasurement(self):
        return self.isMeasured
    
    def rotateQubits(self, qubits, basis):
          if (basis == "Y"):
               self.qc.sdg(qubits)
               self.qc.h(qubits)
          elif (basis == "X"):
               self.qc.h(qubits);
          elif (basis != "Z"):
               print(f"Invalid basis... Got {basis}, but expected 'Y' or 'X'. No rotation has been performed")

    def measureQubits(self, qubits):
        cbits = []
        if type(qubits) is int:
            c = Clbit()
            cbits.append(c)
        elif isinstance(qubits, list):
            for i in range(len(qubits)):
                c = Clbit()
                cbits.append(c)
        reg = ClassicalRegister(bits=cbits)
        try:
            self.qc.add_register(reg)
            self.qc.measure(qubits, cbits)
            self.isMeasured = True
        except:
            print(f"Exception when measuring qubits: {qubits} onto classical bits {cbits}")
            print("Debug information:")
            print(len(qubits))
            print(isinstance(qubits, list))
            print("Debug information over")


class MyAnsatz(MyCircuit):
    def __init__(self, molecule):
        self.molecule = molecule
        super().__init__(molecule.noOfMOs)
        self.qc.ry(np.pi, range(molecule.noOfOccMOs))
        self.hamiltonian = None
     
    def getHamiltonian(self):
        if self.hamiltonian == None:
            self.hamiltonian = MyHamiltonian(self, self.molecule.Nh_pq, self.molecule.Nh_pqrs)
        return self.hamiltonian

    def copyMyAnsatz(self):
        newAnsatz = MyAnsatz(self.molecule)
        newAnsatz.qc = self.qc.copy()
        newAnsatz.N = self.N
        newAnsatz.isMeasured = self.isMeasured
        return newAnsatz;

    def singleQubitExcitation(self, theta, p, q):
        a = p; b = q
        p = b; q = a
        self.qc.rz(np.pi/2, q)
        self.qc.rx(np.pi/2, [q, p])
        self.qc.cx(q, p)
        self.qc.rx(theta, q)
        self.qc.rz(theta, p)
        self.qc.cx(q, p)
        self.qc.rx(-np.pi/2, [q, p])
        self.qc.rz(-np.pi/2, q)

    def doubleQubitExcitation(self, theta, p, q, r, s):
        a = p; b = q; c = r; d = s
        p = d; q = c; r = b; s = a
        self.qc.cx(s, r)
        self.qc.cx(q, p)
        self.qc.x([p, r])
        self.qc.cx(s, q)
        self.qc.ry(theta/4, s)
        self.qc.h(r)
        self.qc.cx(s, r)
        self.qc.ry(-theta/4, s)
        self.qc.h(p)
        self.qc.cx(s, p)
        self.qc.ry(theta/4, s)
        self.qc.cx(s, r)
        self.qc.ry(-theta/4, s)
        self.qc.h(q)
        self.qc.cx(s, q)
        self.qc.ry(theta/4, s)
        self.qc.cx(s, r)
        self.qc.ry(-theta/4, s)
        self.qc.cx(s, p)
        self.qc.ry(theta/4, s)
        self.qc.h(p)
        self.qc.cx(s, r)
        self.qc.ry(-theta/4, s)
        self.qc.h(r)
        self.qc.h(q)
        self.qc.ry(np.pi/2, q)
        self.qc.rz(np.pi/2, q) # THIS SIGN COULD BE WRONG -> IT WAS MINUS ORIGINALLY -> This sign seems to kill the wrong offset in energy, and to give negative gradient, but energy rises (mismatch there)
        self.qc.cx(s, q)
        self.qc.rz(np.pi/2, s) #THIS SIGN IS ORIGINALLY PLUS
        self.qc.rz(-np.pi/2, q) # THIS SIGN COULD BE WRONG -> IT WAS MINUS ORIGINALLY -> Same as above
        self.qc.ry(-np.pi/2, q) # THIS SIGN COULD BE WRONG -> IT WAS MINUS ORIGINALLY -> Fixes all by itself, but mismatch remains
        self.qc.x(r)
        self.qc.x(p)
        self.qc.cx(s, r)
        self.qc.cx(q, p)

    def apply_param_excites(self, param_excites):
        for pa_ex in param_excites: # Construct ansatz with chosen parameterized excitations
            theta = pa_ex[0]
            excitation = pa_ex[1]
            if len(excitation) == 2:
                if theta != 0: self.singleQubitExcitation(theta, excitation[0], excitation[1])
            elif len(excitation) == 4:
                if theta != 0: self.doubleQubitExcitation(theta, excitation[0], excitation[1], excitation[2], excitation[3])

    def calculateEnergy(self):
        circuits = []
        front_factors = []
        finalEnergy = 0
        h = self.getHamiltonian()
        if len(h.pauli_strings) >= 1000000:
            raise ValueError(f"Too many hamiltonian terms!: {len(h.circuits)}. Must be below 1000000")
        for pau in h.pauli_strings:
            circ = self.copyMyCircuit()
            pure_identity = True #Is to be set to True if all operators in a pauli string are identity operators. Is initiated as True, and set as false as soon as a non-identity operator is found in the pauli string
            for p in pau.paulis:
                op = p.operator
                ind = p.index
                if op != "I":
                    circ.rotateQubits(ind, op)
                    circ.measureQubits(ind)
                    pure_identity = False
            if pure_identity:
                finalEnergy += pau.front_factor
            else:
                circuits.append(circ.qc)
                front_factors.append(pau.front_factor)

        sampler = ASampler()
        #For real quantum backend:
            #pm = generate_preset_pass_manager(optimization_level=1, target=backend.target)
            #isa_circuits = pm.run(h.circuits)
        myResult = sampler.run(circuits, shots=None).result().quasi_dists
        for i in range(len(myResult)):
            expectation = 0;
            for key, prob in myResult[i].binary_probabilities().items():
                eigenval = 1;
                for char in key:
                    if int(char) == 1:
                        eigenval *= -1;

                expectation += eigenval*prob
            finalEnergy += front_factors[i]*expectation;

        return finalEnergy + self.molecule.nuclearEnergy

    def calculateGradient(self, param_excites): # One param_excite is on the form (theta, excitation), i.e. (theta, [p, q, r, s]) or (theta, [p, q])
        final_gradient = [] # Should end up as a list of same length as param_excites, which holds all differential quotients for the given excitations at their given parameter values
        ans = self.copyMyAnsatz()
        for pa_ex in param_excites: # The first loop is to construct the ansatz with the current parameters, such that the gradient is calculated at those parameter values
            theta = pa_ex[0]
            excitation = pa_ex[1]
            if len(excitation) == 2:
                if theta != 0: ans.singleQubitExcitation(theta, excitation[0], excitation[1])
            elif len(excitation) == 4:
                if theta != 0: ans.doubleQubitExcitation(theta, excitation[0], excitation[1], excitation[2], excitation[3])

        for pa_ex in param_excites: # The second loop is to get the differential quotient for each excitation in the current ansatz, such that all ansatz parameters can be optimized simultaniously
            circuits = []
            front_factors = []
            complexes = []
            finalEnergy = 0
            finalComplexEnergy = 0
            theta = pa_ex[0]
            excitation = pa_ex[1]
            if len(excitation) == 2:
                T = (((MyPauliString(f"0.5 X{excitation[0]}") - MyPauliString(f"0.5 i Y{excitation[0]}")) * (MyPauliString(f"0.5 X{excitation[1]}") + MyPauliString(f"0.5 i Y{excitation[1]}"))) - ((MyPauliString(f"0.5 X{excitation[1]}") - MyPauliString(f"0.5 i Y{excitation[1]}")) * (MyPauliString(f"0.5 X{excitation[0]}") + MyPauliString(f"0.5 i Y{excitation[0]}")))).reduce()
            elif len(excitation) == 4:
                T = (((MyPauliString(f"0.5 X{excitation[0]}") - MyPauliString(f"0.5 i Y{excitation[0]}")) * (MyPauliString(f"0.5 X{excitation[1]}") - MyPauliString(f"0.5 i Y{excitation[1]}")) * (MyPauliString(f"0.5 X{excitation[2]}") + MyPauliString(f"0.5 i Y{excitation[2]}")) * (MyPauliString(f"0.5 X{excitation[3]}") + MyPauliString(f"0.5 i Y{excitation[3]}"))) - ((MyPauliString(f"0.5 X{excitation[3]}") - MyPauliString(f"0.5 i Y{excitation[3]}")) * (MyPauliString(f"0.5 X{excitation[2]}") - MyPauliString(f"0.5 i Y{excitation[2]}")) * (MyPauliString(f"0.5 X{excitation[1]}") + MyPauliString(f"0.5 i Y{excitation[1]}")) * (MyPauliString(f"0.5 X{excitation[0]}") + MyPauliString(f"0.5 i Y{excitation[0]}")))).reduce()

            H = MyPauliStringList(ans.getHamiltonian().pauli_strings)
            try:
                commutator = ((H*T).reduce() - (T*H).reduce()).reduce().pauli_strings
            except:
                print("Some error happened when reducing this pauli: ")
                (H*T - T*H).print()
            print(f"Measuring commutator, number of pauli strings: {len(commutator)}")
            for pau in commutator:
                complexes.append(pau.complex)
                circ = ans.copyMyCircuit()
                pure_identity = True
                for p in pau.paulis:
                    op = p.operator
                    ind = p.index
                    if op != "I":
                        pure_identity = False
                        circ.rotateQubits(ind, op)
                        circ.measureQubits(ind)
                if pure_identity:
                    if pau.complex == False:
                        finalEnergy += pau.front_factor
                    else:
                        finalComplexEnergy += pau.front_factor
                else:
                    circuits.append(circ.qc)
                    front_factors.append(pau.front_factor)
                
            sampler = ASampler()
            #For real quantum backend:
                #pm = generate_preset_pass_manager(optimization_level=1, target=backend.target)
                #isa_circuits = pm.run(h.circuits)
            if len(commutator) > 0:
                myResult = sampler.run(circuits, shots=None).result().quasi_dists
                for i in range(len(myResult)):
                    expectation = 0;
                    complexExpectation = 0;
                    for key, prob in myResult[i].binary_probabilities().items():
                        eigenval = 1;
                        for char in key:
                            if int(char) == 1:
                                eigenval *= -1;
                        if complexes[i] == False:
                            expectation += eigenval*prob
                        elif complexes[i] == True:
                            complexExpectation += eigenval*prob
                        else:
                            print("Failed to check if this circuit has a complex front factor or not!")

                    finalEnergy += front_factors[i]*expectation;
                    finalComplexEnergy += front_factors[i]*complexExpectation;
                    if np.abs(finalComplexEnergy) > 1e-10:
                        raise ValueError("There is a finite complex expectation value calculating this energy gradient. Something is wrong...")
            final_gradient.append(finalEnergy)

        return final_gradient


    def calculateDifferential(self, theta, excitation): # Excitation on the form [p, q, r, s]
        circuits = []
        front_factors = []
        complexes = []
        finalEnergy = 0
        finalComplexEnergy = 0
        ans = self.copyMyAnsatz()
        #Find the pauli strings that are to be measured, in order to measure the commutator [H, T]
        if len(excitation) == 2:
            T = (((MyPauliString(f"0.5 X{excitation[0]}") - MyPauliString(f"0.5 i Y{excitation[0]}")) * (MyPauliString(f"0.5 X{excitation[1]}") + MyPauliString(f"0.5 i Y{excitation[1]}"))) - ((MyPauliString(f"0.5 X{excitation[1]}") - MyPauliString(f"0.5 i Y{excitation[1]}")) * (MyPauliString(f"0.5 X{excitation[0]}") + MyPauliString(f"0.5 i Y{excitation[0]}")))).reduce()
            if theta != 0: ans.singleQubitExcitation(theta, excitation[0], excitation[1])
        elif len(excitation) == 4:
            T = (((MyPauliString(f"0.5 X{excitation[0]}") - MyPauliString(f"0.5 i Y{excitation[0]}")) * (MyPauliString(f"0.5 X{excitation[1]}") - MyPauliString(f"0.5 i Y{excitation[1]}")) * (MyPauliString(f"0.5 X{excitation[2]}") + MyPauliString(f"0.5 i Y{excitation[2]}")) * (MyPauliString(f"0.5 X{excitation[3]}") + MyPauliString(f"0.5 i Y{excitation[3]}"))) - ((MyPauliString(f"0.5 X{excitation[3]}") - MyPauliString(f"0.5 i Y{excitation[3]}")) * (MyPauliString(f"0.5 X{excitation[2]}") - MyPauliString(f"0.5 i Y{excitation[2]}")) * (MyPauliString(f"0.5 X{excitation[1]}") + MyPauliString(f"0.5 i Y{excitation[1]}")) * (MyPauliString(f"0.5 X{excitation[0]}") + MyPauliString(f"0.5 i Y{excitation[0]}")))).reduce()
            if theta != 0: ans.doubleQubitExcitation(theta, excitation[0], excitation[1], excitation[2], excitation[3])
        print("Excitation pauli strings found...")
        H = MyPauliStringList(ans.getHamiltonian().pauli_strings)
        print("Hamiltonian pauli strings found...")
        try:
            commutator = ((H*T) - (T*H)).pauli_strings
        except:
            print("Some error happened when reducing this pauli: ")
            (H*T - T*H).print()
        print(f"Measuring commutator. Number of pauli strings: {len(commutator)}")
        #for com in commutator:
            #com.print()
        for pau in commutator:
            complexes.append(pau.complex)
            circ = ans.copyMyCircuit()
            pure_identity = True
            for p in pau.paulis:
                op = p.operator
                ind = p.index
                if op != "I":
                    pure_identity = False
                    circ.rotateQubits(ind, op)
                    circ.measureQubits(ind)
            if pure_identity:
                if pau.complex == False:
                    finalEnergy += pau.front_factor
                else:
                    finalComplexEnergy += pau.front_factor
            else:
                circuits.append(circ.qc)
                front_factors.append(pau.front_factor)
            
        sampler = ASampler()
        #For real quantum backend:
            #pm = generate_preset_pass_manager(optimization_level=1, target=backend.target)
            #isa_circuits = pm.run(h.circuits)
        if len(commutator) > 0:
            myResult = sampler.run(circuits, shots=None).result().quasi_dists
            for i in range(len(myResult)):
                expectation = 0;
                complexExpectation = 0;
                for key, prob in myResult[i].binary_probabilities().items():
                    eigenval = 1;
                    for char in key:
                        if int(char) == 1:
                            eigenval *= -1;
                    if complexes[i] == False:
                        expectation += eigenval*prob
                    elif complexes[i] == True:
                        complexExpectation += eigenval*prob
                    else:
                        print("Failed to check if this circuit has a complex front factor or not!")

                finalEnergy += front_factors[i]*expectation;
                finalComplexEnergy += front_factors[i]*complexExpectation;

        return finalEnergy, finalComplexEnergy


class MyHamiltonian:

    def __init__(self, MyAnsatzInstance, h_pq, h_pqrs):
        self.h_pq = copy.deepcopy(h_pq);
        self.h_pqrs = copy.deepcopy(h_pqrs);
        self.pauli_strings = [] #List of MyPauliString, so that we can manipulate the pauli strings and construct circuits later, if we wish

        for p in range(len(self.h_pqrs)):
            for q in range(len(self.h_pqrs[0])):
                singleIntegral = self.h_pq[p][q]
                if singleIntegral != 0: # All zero integrals should simply be skipped
                    if p != q:
                        self.h_pq[q][p] = 0; # Set hermitian conjugate integral to zero, since we will include it in this term
                        
                        pau1 = MyPauliString([MyPauli("X", p), MyPauli("X", q)], front_factor=singleIntegral/2)
                        pau2 = MyPauliString([MyPauli("Y", p), MyPauli("Y", q)], front_factor=singleIntegral/2)

                        for i in range(max([p, q])):
                            if i != p and i != q:
                                a = 0;
                                for l in [p, q]:
                                    if i < l:
                                        a += 1;
                                if a%2 != 0:
                                    for pau in [pau1, pau2]:
                                        pau.append(MyPauliString([MyPauli("Z", i)]))

                        pau1.reduce()
                        pau2.reduce()
                        self.pauli_strings.extend([pau1, pau2])

                    if p == q:
                        pau1 = MyPauliString([MyPauli("I", p)], front_factor=singleIntegral/2)
                        pau2 = MyPauliString([MyPauli("Z", p)], front_factor=-singleIntegral/2)
                        pau1.reduce(); pau2.reduce()
                        self.pauli_strings.extend([pau1, pau2])

                for r in range(len(self.h_pqrs[0][0])):
                    for s in range(len(self.h_pqrs[0][0][0])):
                        doubleIntegral = self.h_pqrs[p][q][r][s]
                        integralSign = 1;
                        if (p > q):
                            integralSign *= -1;
                        if (r < s):
                            integralSign *= -1;
                        if doubleIntegral != 0:
                        
                            if [p, q, r, s] != [s, r, q, p]:
                                self.h_pqrs[s][r][q][p] = 0; # First thing is to assure that the hermitian conjugate is set to zero, so we won't double count it

                                #Now we wish to sort for all the different general cases

                                ##----------------FIRST CASE----------------##
                                if p != q and p!= r and p != s and q != r and q != s and r != s:
                                    pau1 = MyPauliString([MyPauli("X", p), MyPauli("X", q), MyPauli("X", r), MyPauli("X", s)], front_factor=doubleIntegral*integralSign/16)
                                    pau2 = MyPauliString([MyPauli("X", p), MyPauli("X", q), MyPauli("Y", r), MyPauli("Y", s)], front_factor=-doubleIntegral*integralSign/16)
                                    pau3 = MyPauliString([MyPauli("X", p), MyPauli("X", r), MyPauli("Y", q), MyPauli("Y", s)], front_factor=doubleIntegral*integralSign/16)
                                    pau4 = MyPauliString([MyPauli("X", p), MyPauli("X", s), MyPauli("Y", q), MyPauli("Y", r)], front_factor=doubleIntegral*integralSign/16)
                                    pau5 = MyPauliString([MyPauli("X", q), MyPauli("X", r), MyPauli("Y", p), MyPauli("Y", s)], front_factor=doubleIntegral*integralSign/16)
                                    pau6 = MyPauliString([MyPauli("X", q), MyPauli("X", s), MyPauli("Y", p), MyPauli("Y", r)], front_factor=doubleIntegral*integralSign/16)
                                    pau7 = MyPauliString([MyPauli("X", r), MyPauli("X", s), MyPauli("Y", p), MyPauli("Y", q)], front_factor=-doubleIntegral*integralSign/16)
                                    pau8 = MyPauliString([MyPauli("Y", p), MyPauli("Y", q), MyPauli("Y", r), MyPauli("Y", s)], front_factor=doubleIntegral*integralSign/16)

                                    #The "Z_zeta"-term is accounted for in the following for-loop:
                                    for i in range(max([p, q, r, s])):
                                        if i != p and i != q and i != r and i != s:
                                            a = 0;
                                            for l in [p, q, r, s]:
                                                if i < l:
                                                    a += 1;
                                            if a%2 != 0:
                                                for pau in [pau1, pau2, pau3, pau4, pau5, pau6, pau7, pau8]:
                                                    pau.append(MyPauliString([MyPauli("Z", i)]))

                                    for pau in [pau1, pau2, pau3, pau4, pau5, pau6, pau7, pau8]:
                                        pau.reduce()
                                        self.pauli_strings.append(pau)

                                ##----------------THIRD CASE----------------##
                                #in this case, the final Pauli string is zero. Thus, we do nothing...

                                ##----------------FOURTH CASE----------------##
                                #if (p == q or r == s):
                                    #Do nothing. This subcase also results in a zero Pauli string...
                                doSomething = False;    
                                a = 0; b = 0; c = 0;
                                if p == r and q != s and p != q and p != s:
                                    a = p; b = q; c = s;
                                    doSomething = True;
                                elif p == s and q!= r and p != q and p != r:
                                    a = p; b = q; c = r;
                                    doSomething = True;
                                elif q == r and p != s and q != p and q != s:
                                    a = q; b = p; c = s;
                                    doSomething = True;
                                elif q == s and p != r and q != p and q != r:
                                    a = q; b = p; c = r;
                                    doSomething = True;
                                
                                if doSomething == True:
                                    pau1 = MyPauliString([MyPauli("X", b), MyPauli("X", c)], front_factor=integralSign*doubleIntegral/8)
                                    pau2 = MyPauliString([MyPauli("X", b), MyPauli("X", c), MyPauli("Z", a)], front_factor=-integralSign*doubleIntegral/8)
                                    pau3 = MyPauliString([MyPauli("Y", b), MyPauli("Y", c)], front_factor=integralSign*doubleIntegral/8)
                                    pau4 = MyPauliString([MyPauli("Y", b), MyPauli("Y", c), MyPauli("Z", a)], front_factor=-integralSign*doubleIntegral/8)

                                    for i in range(max([p, q, r, s])):
                                        if i != p and i != q and i != r and i != s:
                                            a = 0;
                                            for l in [p, q, r, s]:
                                                if i < l:
                                                    a += 1;
                                            if a%2 != 0:
                                                for pau in [pau1, pau2, pau3, pau4]:
                                                    pau.append(MyPauliString([MyPauli("Z", i)]))

                                    for pau in [pau1, pau2, pau3, pau4]:
                                        pau.reduce()
                                        self.pauli_strings.append(pau)

                                ##----------------FIFTH CASE (A)----------------##
                                #if p == q and r == s:
                                    #This subcase leaves no Pauli string :)
                                
                                if p == r and q == s and p != q:
                                    a = p;
                                    b = q;
                                    pau1 = MyPauliString([MyPauli("I", a), MyPauli("I", b)], front_factor=integralSign*doubleIntegral/4)
                                    pau2 = MyPauliString([MyPauli("I", a), MyPauli("Z", b)], front_factor=-integralSign*doubleIntegral/4)
                                    pau3 = MyPauliString([MyPauli("I", b), MyPauli("Z", a)], front_factor=-integralSign*doubleIntegral/4)
                                    pau4 = MyPauliString([MyPauli("Z", a), MyPauli("Z", b)], front_factor=integralSign*doubleIntegral/4)

                                    for i in range(max([p, q, r, s])):
                                        if i != p and i != q and i != r and i != s:
                                            a = 0;
                                            for l in [p, q, r, s]:
                                                if i < l:
                                                    a += 1;
                                            if a%2 != 0:
                                                for pau in [pau1, pau2, pau3, pau4]:
                                                    pau.append(MyPauliString([MyPauli("Z", i)]))

                                    for pau in [pau1, pau2, pau3, pau4]:
                                        pau.reduce()
                                        self.pauli_strings.append(pau)

                            ##-----------------Cases of "diagonal" indexes-----------------##
                            if [p, q, r, s] == [s, r, q, p]:

                                ##----------------SECOND CASE----------------##
                                #if p == q and p == r and p == s:
                                    ##do nothing
                            
                                ##----------------FIFTH CASE (B)----------------##
                                if p == s and q == r and p!=q: #This is the abba-case -> abba is a palindrome :)
                                    a = p;
                                    b = q;

                                    pau1 = MyPauliString([MyPauli("I", a), MyPauli("I", b)], front_factor=integralSign*doubleIntegral/8)
                                    pau2 = MyPauliString([MyPauli("I", a), MyPauli("Z", b)], front_factor=-integralSign*doubleIntegral/8)
                                    pau3 = MyPauliString([MyPauli("I", b), MyPauli("Z", a)], front_factor=-integralSign*doubleIntegral/8)
                                    pau4 = MyPauliString([MyPauli("Z", a), MyPauli("Z", b)], front_factor=integralSign*doubleIntegral/8)

                                    for i in range(max([p, q, r, s])):
                                        if i != p and i != q and i != r and i != s:
                                            a = 0;
                                            for l in [p, q, r, s]:
                                                if i < l:
                                                    a += 1;
                                            if a%2 != 0:
                                                for c, pau in [pau1, pau2, pau3, pau4]:
                                                    pau.append(MyPauliString([MyPauli("Z", i)]))
        
                                    for pau in [pau1, pau2, pau3, pau4]:
                                        pau.reduce()
                                        self.pauli_strings.append(pau)


class MyPauli:
    def __init__(self, operator, index):
        self.operator = operator
        self.index = index

    def toString(self):
        string = f"{self.operator}{self.index}"
        return string

    def print(self):
        print(self.toString())

    def __eq__(self, other):
        if self.operator == other.operator and self.index == other.index:
            return True
        else:
            return False
    
    def __hash__(self):
        return 3*self.operator.__hash__() + 5*self.index.__hash__()


class MyPauliString:
    def __init__(self, paulis=[], complex=False, front_factor=1): # String on the form "float i(optional) X1 Y2 Z3(amount of operators optional)" Example: "10 Y1 X2" or "-39 i X1"
        if isinstance(paulis, str):
            self.paulis = []
            s = paulis
            self.complex = complex
            self.splitString = s.split()
            self.front_factor = float(self.splitString[0])
            self.pauIndex = 1
            if self.splitString[1] == "i":
                self.complex = True
                self.pauIndex = 2
            for i in self.splitString[self.pauIndex:]:
                self.paulis.append(MyPauli(i[0], int(i[1:])))
        else:
            self.paulis = paulis
            self.complex = complex
            #self.sign = sign
            self.front_factor = front_factor
            #self.reduce()

    def __hash__(self):
        return 5*self.paulis.__hash__() + 7*complex.__hash__()

    def __eq__(self, other):
        if self.complex != other.complex:
            return False
        for p, po in zip(self.paulis, other.paulis):
            if p != po:
                return False
        return True


    def toStringNoFront(self): # Returns a string representation only containing the pauli operators and complex
        printString = "";
        #if self.sign == -1:
            #printString += "-"
        printString += f"{1} "
        if self.complex == True:
            printString += "i "
        for p in self.paulis:
            printString += f"{p.toString()} "
        return printString
    
    def toString(self):
        printString = "";
        #if self.sign == -1:
            #printString += "-"
        printString += f"{self.front_factor}*"
        if self.complex == True:
            printString += "i "
        for p in self.paulis:
            printString += f"{p.toString()} "
        return printString

    def print(self):
        print(self.toString())

    def times_i(self): #Multiplies pauli string by imaginary unit
        if self.complex:
            self.complex = False
            self.front_factor *= -1
        else:
            self.complex = True

    def attemptIteration(length, i, ii):
        if ii + 1 < length:
            ii += 1
            return i, ii
        elif i + 2 < length:
            i += 1
            ii = i + 1
            return i, ii
        else:
            return None, None

    def reduce(self):
        resultDict = dict() #Each key is an index for a qubit, and the corresponding value is the current pauli-operator acting on that qubit
        keySet = set() # Set of keys in resultDict

        for pau in self.paulis:
            i = pau.index
            if i in keySet:
                op1 = resultDict[i]
                op2 = pau.operator
                op12 = op1 + op2
                if op1 == op2:
                    resultDict[i] = "I"
                elif op1 == "I":
                    resultDict[i] = op2
                elif op12 == "XY":
                    resultDict[i] = "Z"
                    self.times_i()
                elif op12 == "YX":
                    resultDict[i] = "Z"
                    self.front_factor *= -1
                    self.times_i()
                elif op12 == "ZX":
                    resultDict[i] = "Y"
                    self.times_i()
                elif op12 == "XZ":
                    resultDict[i] = "Y"
                    self.front_factor *= -1
                    self.times_i()
                elif op12 == "ZY":
                    resultDict[i] = "X"
                    self.front_factor *= -1
                    self.times_i()
                elif op12 == "YZ":
                    resultDict[i] = "X"
                    self.times_i()                 
            else:
                resultDict[i] = pau.operator
                keySet.add(i)

        new_pauli_list = []
        pureIdentity = True
        for pauli in resultDict.values():
            if pauli != "I": pureIdentity = False

        for index, pauli in zip(resultDict.keys(), resultDict.values()):
            if pureIdentity or pauli != "I":
                new_pauli_list.append(MyPauli(pauli, index))

        new_pauli_list.sort(key=lambda x: x.index)
        #new_pauli_string = MyPauliString(new_pauli_list, self.complex, self.front_factor)
        self.paulis = new_pauli_list
        return self




        p = 0
        pp = 1
        while True:
        #for p in self.paulis:
            #for pp in self.paulis[self.paulis.index(p)+1:]:
            if p == None or len(self.paulis) <= 1:
                break;
            try:
                sameIndex = self.paulis[p].index == self.paulis[pp].index
            except IndexError:
                print(f"Caught an indexerror, since p = {p} and/or pp = {p} while length of self.paulis is {len(self.paulis)}. Attempting to fix the issue...")
                if p >= len(self.paulis):
                    break;
                elif pp >= len(self.paulis):
                    p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
            finally:
                if p == None:
                    break;
            if sameIndex:
                i = self.paulis[p].index
                j = p
                if self.paulis[p].operator == "I":
                    self.paulis[j] = MyPauli(self.paulis[pp].operator, i)
                    self.paulis.remove(self.paulis[pp])
                    p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                elif self.paulis[p].operator == "X":
                    if self.paulis[pp].operator == "X":
                        self.paulis[j] = MyPauli("I", i)
                        self.paulis.remove(self.paulis[pp])
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                    elif self.paulis[pp].operator == "Y":
                        self.paulis[j] = MyPauli("Z", i)
                        self.times_i()
                        self.paulis.remove(self.paulis[pp])
                        try:
                            p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                        except:
                            print("We have an exception that number of arguments is 4 instead of 3...")
                            print("Argument types:")
                            print(type(p))
                            print(type(pp))
                            print(type(len(self.paulis)))
                            print("Length of list of arguments: ", len([len(self.paulis), p, pp]))
                    elif self.paulis[pp].operator == "Z":
                        self.paulis[j] = MyPauli("Y", i)
                        self.paulis.remove(self.paulis[pp])
                        self.front_factor *= -1
                        self.times_i()
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                    elif self.paulis[pp].operator == "I":
                        self.paulis.remove(self.paulis[pp])
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                elif self.paulis[p].operator == "Y":
                    if self.paulis[pp].operator == "X":
                        self.paulis[j] = MyPauli("Z", i)
                        self.paulis.remove(self.paulis[pp])
                        self.front_factor *= -1
                        self.times_i()
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                    elif self.paulis[pp].operator == "Y":
                        self.paulis[j] = MyPauli("I", i)
                        self.paulis.remove(self.paulis[pp])
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                    elif self.paulis[pp].operator == "Z":
                        self.paulis[j] = MyPauli("X", i)
                        self.paulis.remove(self.paulis[pp])
                        self.times_i()
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                    elif self.paulis[pp].operator == "I":
                        self.paulis.remove(self.paulis[pp])
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                elif self.paulis[p].operator == "Z":
                    if self.paulis[pp].operator == "X":
                        self.paulis[j] = MyPauli("Y", i)
                        self.paulis.remove(self.paulis[pp])
                        self.times_i()
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                    elif self.paulis[pp].operator == "Y":
                        self.paulis[j] = MyPauli("X", i)
                        self.paulis.remove(self.paulis[pp])
                        self.front_factor *= -1
                        self.times_i()
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                    elif self.paulis[pp].operator == "Z":
                        self.paulis[j] = MyPauli("I", i)
                        self.paulis.remove(self.paulis[pp])
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                    elif self.paulis[pp].operator == "I":
                        self.paulis.remove(self.paulis[pp])
                        p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
            else:
                p, pp = MyPauliString.attemptIteration(len(self.paulis), p, pp)
                    
        try:
            self.paulis.sort(key=lambda x: x.index)
        except:
            print("Error in sorting this pauli string: ", end="")
            self.print()
            print(self)
            print(self.complex)
            print(self.front_factor)
            print("Type of self.paulis for this MyPauliString: ", type(self.paulis))
            print("Type of list elements in PauliString (should be MyPauli instances): ", type(self.paulis[0]))
            #return self.reduce()
        return self

    def append(self, other): #With this, one can append a pauli string to another, reduce it, thus getting the resultant pauli string from multiplying the two in that particular ordering
        newPauli = MyPauliString(self.paulis)
        for o in other.paulis:
            newPauli.paulis.append(o)
        newPauli.front_factor = self.front_factor * other.front_factor
        if self.complex == True and other.complex == True:
            newPauli.front_factor *= -1
        elif self.complex == True or other.complex == True:
                newPauli.complex = True
        return newPauli
    
    def __mul__(self, other):
        new_self = copy.deepcopy(self)
        if isinstance(other, MyPauliString):
            return (new_self.append(other)).reduce()
        if isinstance(other, int) or isinstance(other, float):
            new_self.front_factor *= other
            return new_self
        
    def __rmul__(self, other):
        if isinstance(other, MyPauliString):
            return (other.append(self)).reduce()
        if isinstance(other, int) or isinstance(other, float):
            self.front_factor *= other
            return self
    
    def __neg__(self):
        self.front_factor *= -1
        return self;

    def __add__(self, other):
        if isinstance(other, MyPauliStringList):
            return (MyPauliStringList([copy.deepcopt.self]) + other).reduce()
        return MyPauliStringList([copy.deepcopy(self), copy.deepcopy(other)]).reduce()
    
    def __sub__(self, other):
        first = copy.deepcopy(self)
        second = copy.deepcopy(other)
        second.front_factor *= -1
        return MyPauliStringList([first, second]).reduce()



class MyPauliStringList:
    def __init__(self, pauli_strings=[]):
        self.pauli_strings = None
        if isinstance(pauli_strings, list):
            self.pauli_strings = np.array(pauli_strings) # This is a LIST og pauli_strings, each with a certain front_factor (and of course list of pauli operators). This is enough data to describe expressions (multiple terms) og pauli strings
        elif isinstance(pauli_strings, np.ndarray):
            self.pauli_strings = pauli_strings
        self.len = len(self.pauli_strings)

    def __eq__(self, other):
        if self.pauli_strings == other.pauli_strings:
            return True
        return False

    def __add__(self, other): # returns one list with all pauli strings from the two pauli_string_lists
        if isinstance(other, MyPauliString):
            return (self + MyPauliStringList([copy.deepcopy(other)])).reduce()
        self_pauli_strings = copy.deepcopy(self.pauli_strings)
        other_pauli_strings = copy.deepcopy(other.pauli_strings)
        self_pauli_strings = np.append(self_pauli_strings, other_pauli_strings)
        return MyPauliStringList(self_pauli_strings).reduce()
    
    def __sub__(self, other):
        new_other = copy.deepcopy(other)
        for p_string in new_other.pauli_strings:
            p_string.front_factor *= -1
        return self + new_other
    
    def __mul__(self, other):
        new_pauli_strings = []
        for first in self.pauli_strings:
            for second in other.pauli_strings:
                new_pauli_strings = np.append(new_pauli_strings, first*second)
        return MyPauliStringList(new_pauli_strings).reduce()

    def reduce(self): #Needs to be able to pick out terms that have the same pauli_strings, such that complex terms hopefully cancel out in the end
        resultDict = dict() #Each key is a tuple of a string representation of a pauli string, and a complex True or False
        keySet = set() # Set of keys in resultDict
        for pau in self.pauli_strings:
            pau.reduce()
            tup = (pau.toStringNoFront(), pau.complex)
            if tup in keySet:
                resultDict[tup] += pau.front_factor
            else:
                resultDict[tup] = pau.front_factor
                keySet.add(tup)
        new_pauli_strings = []
        for tup, front in zip(resultDict.keys(), resultDict.values()):
            if not MyMath.approx(front, 0):
                pauli = MyPauliString(tup[0])
                pauli.front_factor = front
                pauli.complex = tup[1]
                new_pauli_strings.append(pauli)
        self.pauli_strings = new_pauli_strings
        return self

        p = 0
        pp = 1
        for pauli in self.pauli_strings:
            pauli.reduce()
        while True:
            nothingHappened = True
            if p == None or len(self.pauli_strings) <= 1:
                break;
            new_pauli_strings = []
            if self.pauli_strings[p].paulis == self.pauli_strings[pp].paulis and self.pauli_strings[p].complex == self.pauli_strings[pp].complex:
                new_front_factor = self.pauli_strings[p].front_factor + self.pauli_strings[pp].front_factor
                if MyMath.approx(new_front_factor, 0):
                    self.pauli_strings = np.delete(self.pauli_strings, pp)
                    self.pauli_strings = np.delete(self.pauli_strings, p)
                    self.len -= 2
                    nothingHappened = False
                else:
                    self.pauli_strings[p] = MyPauliString(self.pauli_strings[p].paulis, complex=self.pauli_strings[p].complex, front_factor=new_front_factor)
                    self.pauli_strings = np.delete(self.pauli_strings, pp)
                    self.len -= 1
                    nothingHappened = False
            p, pp = MyPauliStringList.attemptIteration(self.len, p, pp)
        if nothingHappened:
            return self

    def attemptIteration(length, i, ii):
        if ii + 1 < length:
            ii += 1
            return i, ii
        elif i + 2 < length:
            i += 1
            ii = i + 1
            return i, ii
        else:
            return None, None

    def print(self):
        for ps in self.pauli_strings:
            if ps.front_factor > 0:
                print("+", end="")
            ps.print()


class MyMath:
    def convert_to_cartesian(spherical_coordinates):
        r = spherical_coordinates[0]
        theta = spherical_coordinates[1]
        phi = spherical_coordinates[2]
        return [r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)]
    
    def find_distance(point1, point2, spherical=False):
        if spherical == True:
            raise ValueError("I have not learned to find distances in spherical coordinates yet... Please wait for an update before attempting this")
        else:
            x1 = point1[0]
            y1 = point1[1]
            z1 = point1[2]
            x2 = point2[0]
            y2 = point2[1]
            z2 = point2[2]
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    def approx(firstFloat, secondFloat):
        if np.abs(firstFloat) - np.abs(secondFloat) <= 1e-10:
            return True
        else:
            return False




