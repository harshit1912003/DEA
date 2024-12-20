from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
from utils.is_efficient import is_efficient
import pickle
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D


class DEA:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.n, self.m = input_data.shape
        self.s = output_data.shape[1]
        self.results = {}

    def save_results(self):
        root = tk.Tk()
        root.withdraw()  
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl", 
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:  
            with open(file_path, "wb") as file:
                pickle.dump(self.results, file)
            print(f"Results saved to {file_path}")
        else:
            print("Save operation canceled.")

    def ccr_input_p1(self):
        efficiencies = []

        for o in range(self.n):
            model = Model("InputEfficiency")
            model.setParam(GRB.Param.OutputFlag, 0)

            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            theta = model.addVar(lb=0, name="theta")

            model.setObjective(theta, GRB.MINIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= theta * self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r])

            model.optimize()
            efficiencies.append(model.objVal)
        return efficiencies

    def ccr_input_p2(self, input_efficiencies):
        results = []

        for o in range(self.n):
            theta_star = input_efficiencies[o]
            model = Model("Phase2")
            model.setParam(GRB.Param.OutputFlag, 0)

            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]

            model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == theta_star * self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= self.output_data[o, r])

            model.optimize()
            results.append({
                'DMU': o,
                'efficiency': theta_star,
                'slacks_minus': [S_minus[i].x for i in range(self.m)],
                'slacks_plus': [S_plus[r].x for r in range(self.s)],
                'Lambda': [lambdas[j].x for j in range(self.n)]
            })
        return pd.DataFrame(results)

    def ccr_input(self):
        if "ccr_input" in self.results:
            return self.results['ccr_input']
        
        results = pd.DataFrame(self.ccr_input_p2(self.ccr_input_p1()))
        results = is_efficient(results, 'ccr_input')
        self.results['ccr_input'] = results
        return results

    def ccr_output(self):
        if "ccr_output" in self.results:
            return self.results['ccr_output']

        if "ccr_input" in self.results:
            results = self.results['ccr_input']
        else:
            input_efficiencies = self.ccr_input_p1()
            results = pd.DataFrame(self.ccr_input_p2(input_efficiencies))

        for i in range(self.n):
            results.loc[i, 'efficiency'] = 1 / results.loc[i, 'efficiency']
            slacks_minus_list = [results.loc[i, 'slacks_minus'][j] * results.loc[i, 'efficiency'] for j in range(len(results.loc[i, 'slacks_minus']))]
            slacks_plus_list = [results.loc[i, 'slacks_plus'][j] * results.loc[i, 'efficiency'] for j in range(len(results.loc[i, 'slacks_plus']))]
            lambda_list = [results.loc[i, 'Lambda'][j] * results.loc[i, 'efficiency'] for j in range(len(results.loc[i, 'Lambda']))]

            results.at[i, 'slacks_minus'] = slacks_minus_list
            results.at[i, 'slacks_plus'] = slacks_plus_list
            results.at[i, 'Lambda'] = lambda_list

        results.rename(columns={
            'efficiency': 'n', 
            'slacks_minus': 't_minus',
            'slacks_plus':'t_plus',
            'Lambda':'u'}, inplace=True)
        
        results = is_efficient(results, 'ccr_output')

        self.results['ccr_output'] = results
        return results

    def bcc_input_p1(self):

        efficiencies = []

        for o in range(self.n):
            model = Model("InputEfficiency")
            model.setParam(GRB.Param.OutputFlag, 0)
            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            theta = model.addVar(lb=0, name="theta")
            model.setObjective(theta, GRB.MINIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= theta * self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r])

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()
            efficiencies.append(model.objVal)

        return efficiencies

    def bcc_input_p2(self, input_efficiencies):
        results = []

        for o in range(self.n):
            theta_star = input_efficiencies[o]
            model = Model("Phase2")
            model.setParam(GRB.Param.OutputFlag, 0)
            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == theta_star * self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= self.output_data[o, r])

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()

            results.append({
                'DMU': o,
                'efficiency': theta_star,
                'slacks_minus': [S_minus[i].x for i in range(self.m)],
                'slacks_plus': [S_plus[r].x for r in range(self.s)],
                'Lambda': [lambdas[j].x for j in range(self.n)]
            })

        return pd.DataFrame(results)

    def bcc_input(self):
        if "bcc_input" in self.results:
            return self.results['bcc_input']

        results = pd.DataFrame(self.bcc_input_p2(self.bcc_input_p1()))
        results = is_efficient(results, 'bcc_input')
        self.results['bcc_input'] = results
        return results

    def bcc_output_p1(self):
        efficiencies = []

        for o in range(self.n):
            model = Model("OutputEfficiency")
            model.setParam(GRB.Param.OutputFlag, 0)

            lambdas = [model.addVar(lb=0, name=f"lambdas_{j}") for j in range(self.n)]
            eta = model.addVar(lb=1, name="eta")

            model.setObjective(eta, GRB.MAXIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= eta * self.output_data[o, r])
            
            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)

            model.optimize()
            efficiencies.append(model.objVal)

        return efficiencies

    def bcc_output_p2(self, output_efficiencies):
        results = []

        for o in range(self.n):
            eta_star = output_efficiencies[o]
            model = Model("Phase2")
            model.setParam(GRB.Param.OutputFlag, 0)

            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]

            model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= eta_star * self.output_data[o, r])

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()

            results.append({
                'DMU': o,
                'efficiency': eta_star,
                'slacks_minus': [S_minus[i].x for i in range(self.m)],
                'slacks_plus': [S_plus[r].x for r in range(self.s)],
                'Lambda': [lambdas[j].x for j in range(self.n)]
            })

        return pd.DataFrame(results)

    def bcc_output(self):
        if "bcc_output" in self.results:
            return self.results['bcc_output']

        results = pd.DataFrame(self.bcc_output_p2(self.bcc_output_p1()))
        results.rename(columns={
            'efficiency': 'n', 
            'slacks_minus': 't_minus',
            'slacks_plus':'t_plus',
            'Lambda':'u'}, inplace=True)
        results = is_efficient(results, 'bcc_output')

        self.results['bcc_output'] = results
        return results
    
    def add(self):
        
        if "add" in self.results:
            return self.results['add']

        results = []

        for o in range(self.n):
            model = Model("AdditiveModel")
            model.setParam(GRB.Param.OutputFlag, 0)

            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]

            model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] == self.output_data[o, r])

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)

            model.optimize()

            results.append({
                'DMU': o,
                'slacks_minus': [S_minus[i].x for i in range(self.m)],
                'slacks_plus': [S_plus[r].x for r in range(self.s)],
                'Lambda': [lambdas[j].x for j in range(self.n)]
            })
        results = pd.DataFrame(results)
        results = is_efficient(results, 'add')
        self.results['add'] = results
        return results

    def sbm(self):
        if "sbm" in self.results:
            return self.results['sbm']
        results = []
        for o in range(self.n):
            model = Model("OriginalSBM_Model")
            model.setParam(GRB.Param.OutputFlag, 0)

            Lambda = [model.addVar(lb=0, name=f"Lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            t = model.addVar(lb= 0, name="t")

            model.setObjective(t - quicksum(S_minus[i] / self.input_data[o, i] for i in range(self.m)) / self.m, GRB.MINIMIZE)

            model.addConstr(t + quicksum(S_plus[r] / self.output_data[o, r] for r in range(self.s)) / self.s == 1)

            for i in range(self.m):
                model.addConstr(quicksum(Lambda[j] * self.input_data[j, i] for j in range(self.n)) == t * self.input_data[o, i] - S_minus[i])

            for r in range(self.s):
                model.addConstr(quicksum(Lambda[j] * self.output_data[j, r] for j in range(self.n)) == t * self.output_data[o, r] + S_plus[r])

            model.optimize()

            rho_star = model.objVal
            t_star = t.x
            lambda_star = [Lambda[j].x / t_star for j in range(self.n)]
            s_minus_star = [S_minus[i].x / t_star for i in range(self.m)]
            s_plus_star = [S_plus[r].x / t_star for r in range(self.s)]

            results.append({
                'DMU': o,
                'rho': rho_star,
                'lambda': lambda_star,
                's_minus': s_minus_star,
                's_plus': s_plus_star
            })

        results = pd.DataFrame(results)
        results = is_efficient(results, 'sbm')
        self.results['sbm'] = results
        return results

    
    # def modified_sbm(self):
    #     results = []
    #     P_minus = np.array([[self.input_data[o, i] - np.min(self.input_data[:, i])
    #                         for i in range(self.m)] for o in range(self.n)])
    #     print("P-", P_minus)
    #     P_plus = np.array([[np.max(self.output_data[:, r]) - self.output_data[o, r]
    #                         for r in range(self.s)] for o in range(self.n)])
    #     print("P+", P_plus)

    #     for o in range(self.n):
    #         model = Model("ModifiedSBM_Model")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         Lambda = [model.addVar(lb=0, name=f"Lambda_{j}") for j in range(self.n)]
    #         S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
    #         S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
    #         V = [model.addVar(lb=0, name=f"V_{r}") for r in range(self.s)]
    #         W = [model.addVar(lb=0, name=f"W_{i}") for i in range(self.m)]
    #         t = model.addVar(lb=0, name="t")

    #         objective_terms = [(W[i] * S_minus[i]) / P_minus[o, i] for i in range(self.m) if P_minus[o, i] != 0]
    #         model.setObjective(t - quicksum(objective_terms) / self.m, GRB.MINIMIZE)

    #         constraint_terms = [(V[r] * S_plus[r]) / P_plus[o, r] for r in range(self.s) if P_plus[o, r] != 0]
    #         model.addConstr(t + quicksum(constraint_terms) / self.s == 1)

    #         for i in range(self.m):
    #             min_input_i = np.min(self.input_data[:, i])
    #             model.addConstr(quicksum(Lambda[j] * (self.input_data[j, i] - min_input_i) for j in range(self.n))
    #                             == t * (self.input_data[o, i] - min_input_i) - S_minus[i])

    #         for r in range(self.s):
    #             max_output_r = np.max(self.output_data[:, r])
    #             model.addConstr(quicksum(Lambda[j] * (max_output_r - self.output_data[j, r]) for j in range(self.n))
    #                             == t * (max_output_r - self.output_data[o, r]) + S_plus[r])

    #         model.addConstr(quicksum(W[i] for i in range(self.m)) == self.m)
    #         model.addConstr(quicksum(V[r] for r in range(self.s)) == self.s)

    #         model.optimize()

    #         rho_star = model.objVal
    #         t_star = t.x
    #         lambda_star = [Lambda[j].x / t_star for j in range(self.n)]
    #         s_minus_star = [W[i]*S_minus[i].x / t_star for i in range(self.m)]
    #         s_plus_star = [V[r]*S_plus[r].x / t_star for r in range(self.s)]

    #         results.append({
    #             'DMU': o,
    #             'rho': rho_star,
    #             'lambda': lambda_star,
    #             's_minus': s_minus_star,
    #             's_plus': s_plus_star
    #         })

    #     return pd.DataFrame(results)


#####################################################################################
    def modified_sbm(self):
        results = []
        P_minus = np.array([[self.input_data[o, i] - np.min(self.input_data[:, i])
                            for i in range(self.m)] for o in range(self.n)])
        print("P-", P_minus)
        P_plus = np.array([[np.max(self.output_data[:, r]) - self.output_data[o, r]
                            for r in range(self.s)] for o in range(self.n)])
        print("P+", P_plus)

        for o in range(self.n):
            model = Model("ModifiedSBM_Model")
            model.setParam(GRB.Param.OutputFlag, 0)

            Lambda = [model.addVar(lb=0, name=f"Lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            t = model.addVar(lb=1e-6, name="t")

            objective_terms = [S_minus[i]/P_minus[o, i] for i in range(self.m) if P_minus[o, i]!= 0]
            model.setObjective(t - quicksum(objective_terms) / self.m, GRB.MINIMIZE)

            constraint_terms = [S_plus[r]/P_plus[o, r] for r in range(self.s) if P_plus[o, r] != 0]
            model.addConstr((t + quicksum(constraint_terms) / self.s) == 1)

            for i in range(self.m):
                model.addConstr(quicksum(Lambda[j] *P_minus[j, i] for j in range(self.n))
                                == t * (P_minus[o, i]) - S_minus[i])
            for r in range(self.s):
                model.addConstr(quicksum(Lambda[j] * (P_plus[j, r]) for j in range(self.n))
                                == t * (P_plus[o, r]) + S_plus[r])

            model.optimize()

            rho_star = model.objVal
            t_star = t.x
            print("t*", t_star)
            lambda_star = [Lambda[j].x / t_star for j in range(self.n)]
            s_minus_star = [S_minus[i].x / t_star for i in range(self.m)]
            s_plus_star = [S_plus[r].x / t_star for r in range(self.s)]

            results.append({
                'DMU': o,
                'rho': rho_star,
                'lambda': lambda_star,
                's_minus': s_minus_star,
                's_plus': s_plus_star
            })

        return pd.DataFrame(results)
#####################################################################################

    def rdm(self):
        if "rdm" in self.results:
            return self.results['rdm']
        
        results = []

        for o in range(self.n):
            model = Model("RDM_Model")
            model.setParam(GRB.Param.OutputFlag, 0)

            R_minus = [self.input_data[o, i] - min(self.input_data[:, i]) for i in range(self.m)]
            R_plus = [max(self.output_data[:, r]) - self.output_data[o, r] for r in range(self.s)]

            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            beta = model.addVar(lb=0, name="beta")
            
            model.setObjective(1-beta, GRB.MINIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i] - beta*R_minus[i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r] + beta*R_plus[r])

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()

            results.append({
                "DMU": o,
                "efficiency": 0.0 if abs(model.ObjVal) < 1e-9 else model.ObjVal
            })
        results = pd.DataFrame(results)
        results = is_efficient(results, 'rdm')
        self.results['rdm'] = results
        return results

    def plot(self, typ):
        fig, ax = plt.subplots()
        N, M = self.input_data.shape
        N, S = self.output_data.shape

        if M == 1 and S == 1:
            x = self.input_data.flatten()
            y = self.output_data.flatten()
            ax.scatter(x, y, c='blue')
            ax.set_xlabel('Input')
            ax.set_ylabel('Output')
            title = 'Input vs Output'

        elif M == 2 and S == 1:
            x = self.input_data[:, 0] / self.output_data.flatten()
            y = self.input_data[:, 1] / self.output_data.flatten()
            ax.scatter(x, y, c='green')
            ax.set_xlabel('Input1 / Output')
            ax.set_ylabel('Input2 / Output')
            title = 'Input1/Output vs Input2/Output'

        elif M == 1 and S == 2:
            x = self.output_data[:, 0] / self.input_data.flatten()
            y = self.output_data[:, 1] / self.input_data.flatten()
            ax.scatter(x, y, c='red')
            ax.set_xlabel('Output1 / Input')
            ax.set_ylabel('Output2 / Input')
            title = 'Output1/Input vs Output2/Input'

        else:
            print("Unsupported combination of M and S")
            return

        typ2func = {
            'ccr_input': self.ccr_input,  
            'ccr_output': self.ccr_output,
            'sbm': self.sbm,
            'add': self.add,
            'bcc_input': self.bcc_input,
            'bcc_output': self.bcc_output,
        }

        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            return

        if typ not in self.results:
            print(f"Calculating results for {typ}...")
            res = typ2func[typ]()  
        else:
            print(f"Using cached results for {typ}...")
            res = self.results[typ]  

        efficiency_data = res  
        if 'is_efficient' in efficiency_data.columns:
            is_efficient = efficiency_data['is_efficient'].values
        else:
            print(f"Warning: 'is_efficient' column missing in results for {typ}")
            return

        for i, efficient in enumerate(is_efficient):
            if efficient:
                circle = plt.Circle((x[i], y[i]), radius=0.05, fill=False, ec='r')
                ax.add_artist(circle)
                ax.text(x[i], y[i] + 0.02, typ, color='black', fontsize=10, ha='center')

        ax.set_title(title)
        plt.show()

    def plot3d(self, typ):
        fig = plt.figure()

        N, M = self.input_data.shape
        N, S = self.output_data.shape

        if M == 1 and S == 1:
            raise ValueError("Exactally 3 axes required")

        elif M == 2 and S == 1:
            ax = fig.add_subplot(111, projection='3d')
            x = self.input_data[:, 0]
            y = self.input_data[:, 1]
            z = self.output_data.flatten()
            ax.scatter(x, y, z, c='green')
            ax.set_xlabel('Input1')
            ax.set_ylabel('Input2')
            ax.set_zlabel('Output')
            title = 'Input1 vs Input2 vs Output'

        elif M == 1 and S == 2:
            ax = fig.add_subplot(111, projection='3d')
            x = self.input_data.flatten()
            y = self.output_data[:, 0]
            z = self.output_data[:, 1]
            ax.scatter(x, y, z, c='red')
            ax.set_xlabel('Input')
            ax.set_ylabel('Output1')
            ax.set_zlabel('Output2')
            title = 'Input vs Output1 vs Output2'

        else:
            raise ValueError("Unsupported combination of M and S")

        typ2func = {
            'ccr_input': self.ccr_input,  
            'ccr_output': self.ccr_output,
            'sbm': self.sbm,
            'add': self.add,
            'bcc_input': self.bcc_input,
            'bcc_output': self.bcc_output,
        }

        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            return

        if typ not in self.results:
            print(f"Calculating results for {typ}...")
            res = typ2func[typ]()  
            self.results[typ] = res  
        else:
            print(f"Using cached results for {typ}...")
            res = self.results[typ]  

        efficiency_data = res  
        if 'is_efficient' in efficiency_data.columns:
            is_efficient = efficiency_data['is_efficient'].values
        else:
            print(f"Warning: 'is_efficient' column missing in results for {typ}")
            return

        for i, efficient in enumerate(is_efficient):
            if efficient:
                ax.scatter(x[i], y[i], z[i], s=200, facecolors='none', edgecolors='r', linewidths=2)
                ax.text(x[i], y[i], z[i] + 0.02, typ, color='black', fontsize=10, ha='center')

        ax.set_title(title)
        ax.grid(True)
        plt.show()

#####################################################################################

    def plot_with_frontier(self, typ):
        if typ in self.results:
            print(f"Results for model '{typ}' are already computed. Skipping plot.")

        if self.input_data.shape[1] != 1 or self.output_data.shape[1] != 1:
            raise ValueError("Unsupported combination of M and S")

        fig, ax = plt.subplots()

        x = self.input_data.flatten()
        y = self.output_data.flatten()
        ax.scatter(x, y, c='blue')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        title = 'Input vs Output'
        fake_origin = (np.min(x), np.max(y))

        max_x, max_y = np.max(x), np.max(y)

        typ2func = {
            'ccr_input': self.ccr_input,  
            'ccr_output': self.ccr_output,
            'sbm': self.sbm,
            'add': self.add,
            'bcc_input': self.bcc_input,
            'bcc_output': self.bcc_output,
        }

        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            return

        if typ not in self.results:
            print(f"Calculating results for {typ}...")
            results = typ2func[typ]()  
        else:
            print(f"Using cached results for {typ}...")
            results = self.results[typ]  

        efficient_points = []
        for i, efficient in enumerate(results['is_efficient']):
            if efficient:
                efficient_points.append((x[i], y[i]))
                circle = plt.Circle((x[i], y[i]), radius=0.05, fill=False, ec='r')
                ax.add_artist(circle)
                ax.text(x[i], y[i] + 0.02, typ, color='black', fontsize=10, ha='center')

        if len(efficient_points) > 2:
            hull = ConvexHull(efficient_points)
            edges = []
            for simplex in hull.simplices:
                edge = (efficient_points[simplex[0]], efficient_points[simplex[1]])
                dist = self.perpendicular_distance(fake_origin, edge[0], edge[1])
                edges.append((edge, dist))

            edges.sort(key=lambda x: x[1])
            selected_edges = edges[:len(efficient_points)-1]

            for edge, _ in selected_edges:
                ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'r-', linewidth=2)

        elif len(efficient_points) == 2:
            ax.plot([efficient_points[0][0], efficient_points[1][0]],
                    [efficient_points[0][1], efficient_points[1][1]], 'r-', linewidth=2)

        elif len(efficient_points) == 1:
            extended_point = self.extend_line((0, 0), efficient_points[0], max_x)
            ax.plot([0, extended_point[0]], [0, extended_point[1]], 'r-', linewidth=2)

        ax.set_xlim(0, max_x * 1.1)
        ax.set_ylim(0, max_y * 1.1)
        ax.set_title(title)
        plt.show()

#####################################################################################

    def plot_with_frontier(self, typ):

        if typ in self.results:
            print(f"Results for model '{typ}' are already computed. Skipping plot.")

        if self.input_data.shape[1] != 1 or self.output_data.shape[1] != 1:
            raise ValueError("Unsupported combination of M and S")

        fig, ax = plt.subplots()

        x = self.input_data.flatten()
        y = self.output_data.flatten()
        ax.scatter(x, y, c='blue')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        title = 'Input vs Output'
        fake_origin = (np.min(x), np.max(y))

        max_x, max_y = np.max(x), np.max(y)

        typ2func = {
            'ccr_input': self.ccr_input,  
            'ccr_output': self.ccr_output,
            'sbm': self.sbm,
            'add': self.add,
            'bcc_input': self.bcc_input,
            'bcc_output': self.bcc_output,
        }

        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            return

        if typ not in self.results:
            print(f"Calculating results for {typ}...")
            results = typ2func[typ]()  
        else:
            print(f"Using cached results for {typ}...")
            results = self.results[typ]  

        efficient_points = []
        for i, efficient in enumerate(results['is_efficient']):
            if efficient:
                efficient_points.append((x[i], y[i]))
                circle = plt.Circle((x[i], y[i]), radius=0.05, fill=False, ec='r')
                ax.add_artist(circle)
                ax.text(x[i], y[i] + 0.02, typ, color='black', fontsize=10, ha='center')

        if len(efficient_points) > 2:
            hull = ConvexHull(efficient_points)
            edges = []
            for simplex in hull.simplices:
                edge = (efficient_points[simplex[0]], efficient_points[simplex[1]])
                dist = self.perpendicular_distance(fake_origin, edge[0], edge[1])
                edges.append((edge, dist))

            edges.sort(key=lambda x: x[1])
            selected_edges = edges[:len(efficient_points)-1]

            for edge, _ in selected_edges:
                ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'r-', linewidth=2)

        elif len(efficient_points) == 2:
            ax.plot([efficient_points[0][0], efficient_points[1][0]],
                    [efficient_points[0][1], efficient_points[1][1]], 'r-', linewidth=2)

        elif len(efficient_points) == 1:
            extended_point = self.extend_line((0, 0), efficient_points[0], max_x)
            ax.plot([0, extended_point[0]], [0, extended_point[1]], 'r-', linewidth=2)

        ax.set_xlim(0, max_x * 1.1)
        ax.set_ylim(0, max_y * 1.1)
        ax.set_title(title)
        plt.show()

#####################################################################################


    def perpendicular_distance(self, point, line_point1, line_point2):
        x0, y0 = point
        x1, y1 = line_point1
        x2, y2 = line_point2
        return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    def extend_line(self, p1, p2, max_x):
        if p2[0] == p1[0]:  
            return (p2[0], max_x * (p2[1] - p1[1]) / (p2[0] - p1[0]) + p1[1])
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            return (max_x, slope * (max_x - p1[0]) + p1[1])