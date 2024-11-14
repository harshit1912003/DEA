from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
from is_efficient import is_efficient
import pickle
import tkinter as tk
from tkinter import filedialog


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

    
    def fdh_input_crs(self):
        if "fdh_input_crs" in self.results:
            return self.results['fdh_input_crs']

        results = []

        base_model = Model("InputEfficiency")
        base_model.setParam(GRB.Param.OutputFlag, 0)

        for o in range(self.n):

            model = base_model.copy()

            z = model.addVars(self.n, vtype=GRB.BINARY, name="z")
            theta = model.addVar(lb=0, name="theta")
            delta = model.addVar(lb=0, name='delta')

            model.setObjective(theta, GRB.MINIMIZE)

            for i in range(self.m):
                model.addConstr(
                    delta * quicksum(z[j] * self.input_data[j, i] for j in range(self.n)) 
                    <= theta * self.input_data[o, i]
                )

            for r in range(self.s):
                model.addConstr(
                    quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) 
                    >= self.output_data[o, r]
                )

            model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

            model.optimize()

            results.append({
                "DMU": o,
                "efficiency": model.objVal
            })

        results = pd.DataFrame(results)
        results = is_efficient(results, 'fdh_input_crs')

        self.results['fdh_input_crs'] = results

        return results

    def fdh_output_crs(self):
        if "fdh_output_crs" in self.results:
            return self.results['fdh_output_crs']
        
        results = []

        for o in range(self.n):
            model = Model("OutputEfficiency")
            model.setParam(GRB.Param.OutputFlag, 0)

            z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
            eta = model.addVar(lb=1, name="eta")
            delta = model.addVar(lb = 0, name='delta')

            model.setObjective(eta, GRB.MAXIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) >= eta * self.output_data[o, r])
            model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

            model.optimize()
            results.append({"DMU": o, 
                            "efficiency": model.objVal})
        results = pd.DataFrame(results)
        results = is_efficient(results, 'fdh_output_crs')
        self.results['fdh_output_crs'] = results
        return results

    def fdh_input_vrs(self):
        if "fdh_input_vrs" in self.results:
            return self.results['fdh_input_vrs']
        results = []
        for o in range(self.n):
            model = Model("InputEfficiency")
            model.setParam(GRB.Param.OutputFlag, 0)

            lambdas = [model.addVar(vtype=GRB.BINARY, name=f"lambda_{j}") for j in range(self.n)]
            theta = model.addVar(lb=0, name="theta")
            model.setObjective(theta, GRB.MINIMIZE)

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= theta * self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r])

            model.optimize()

            results.append({"DMU": o, 
                            "efficiency": model.objVal})

        results = pd.DataFrame(results)
        results = is_efficient(results, 'fdh_input_vrs')
        self.results['fdh_input_vrs'] = results
        return results

    def fdh_output_vrs(self):
        if "fdh_output_vrs" in self.results:
            return self.results['fdh_output_vrs']
        results = []

        for o in range(self.n):
            model = Model("OutputEfficiency")
            model.setParam(GRB.Param.OutputFlag, 0)

            lambdas = [model.addVar(vtype=GRB.BINARY, name=f"lambda_{j}") for j in range(self.n)]
            eta = model.addVar(lb=1, name="eta")

            model.setObjective(eta, GRB.MAXIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= eta * self.output_data[o, r])
            
            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)

            model.optimize()
            results.append({"DMU": o, 
                            "efficiency": model.objVal})
        results = pd.DataFrame(results)
        results = is_efficient(results, 'fdh_output_vrs')
        self.results['fdh_output_vrs'] = results
        return results




    # def modified_sbm(self):

    #     results = []

    #     P_minus = np.array([[self.input_data[o, i] - np.min(self.input_data[:, i])
    #                         for i in range(self.m)] for o in range(self.n)])
    #     P_plus = np.array([[np.max(self.output_data[:, r]) - self.output_data[o, r]
    #                         for r in range(self.s)] for o in range(self.n)])

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

    # def fdh_input_crs(self):
    #     if "fdh_input_crs" in self.results:
    #         return self.results['fdh_input_crs']

    #     results = []

    #     for o in range(self.n):
    #         model = Model("InputEfficiency")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         theta = model.addVar(lb=0, name="theta")
    #         delta = model.addVar(lb=0, name='delta') 

    #         model.setObjective(theta, GRB.MINIMIZE)

    #         for i in range(self.m):
    #             model.addConstr(delta*quicksum(z[j] * self.input_data[j, i] for j in range(self.n)) <= theta * self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta*z[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

    #         model.optimize()

    #         results.append({"DMU": o, 
    #                         "efficiency": model.objVal})
            
    #     print(delta)
    #     results = pd.DataFrame(results)
    #     results = is_efficient(results, 'fdh_input_crs')
    #     self.results['fdh_input_crs'] = results
    #     return results

    # def fdh_input_p2_crs(self, input_efficiencies):
    #     results = []

    #     for o in range(self.n):
    #         theta_star = input_efficiencies[o]
    #         model = Model("Phase2")
    #         model.setParam(GRB.Param.OutputFlag, 0)
    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
    #         S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
    #         model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)
    #         delta = model.addVar(lb=0, name='delta')

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == theta_star * self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)
    #         model.optimize()

    #         results.append({
    #             'DMU': o,
    #             'efficiency': theta_star,
    #             'slacks_minus': [S_minus[i].x for i in range(self.m)],
    #             'slacks_plus': [S_plus[r].x for r in range(self.s)],
    #             'Lambda': [z[j].x for j in range(self.n)]
    #         })


    #     return pd.DataFrame(results)

    # def fdh_input_crs(self):
    #     if "fdh_input_crs" in self.results:
    #         return self.results['fdh_input_crs']

    #     results = pd.DataFrame(self.fdh_input_p2_crs(self.fdh_input_p1_crs()))
    #     results = is_efficient(results, 'fdh_input_crs')
    #     self.results['fdh_input_crs'] = results
    #     return results



    # def fdh_output_p2_crs(self, output_efficiencies):
    #     results = []

    #     for o in range(self.n):
    #         eta_star = output_efficiencies[o]
    #         model = Model("Phase2")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
    #         S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
    #         delta = model.addVar(lb=0, name='delta')

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] == eta_star * self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

    #         model.optimize()

    #         results.append({
    #             'DMU': o,
    #             'efficiency': eta_star,
    #             'slacks_minus': [S_minus[i].X for i in range(self.m)],
    #             'slacks_plus': [S_plus[r].X for r in range(self.s)],
    #             'Lambda': [z[j].X for j in range(self.n)]
    #         })

    #     return pd.DataFrame(results)

    # def fdh_output_crs(self):
    #     if "fdh_output_crs" in self.results:
    #         return self.results['fdh_output_crs']

    #     results = pd.DataFrame(self.fdh_output_p2_crs(self.fdh_output_p1_crs()))
    #     results.rename(columns={
    #         'efficiency': 'n', 
    #         'slacks_minus': 't_minus',
    #         'slacks_plus':'t_plus',
    #         'Lambda':'u'}, inplace=True)
    #     results = is_efficient(results, 'fdh_output_vrs')
    #     self.results['fdh_output_crs'] = results
    #     return results

    # def fdh_input_p2_vrs(self, input_efficiencies):
    #     results = []

    #     for o in range(self.n):
    #         theta_star = input_efficiencies[o]
    #         model = Model("Phase2")
    #         model.setParam(GRB.Param.OutputFlag, 0)
    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
    #         S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
    #         model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)

    #         for i in range(self.m):
    #             model.addConstr(quicksum(z[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == theta_star * self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(z[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)
    #         model.optimize()

    #         results.append({
    #             'DMU': o,
    #             'efficiency': theta_star,
    #             'slacks_minus': [S_minus[i].x for i in range(self.m)],
    #             'slacks_plus': [S_plus[r].x for r in range(self.s)],
    #             'Lambda': [z[j].x for j in range(self.n)]
    #         })


    #     return pd.DataFrame(results)

    # def fdh_input_vrs(self):
    #     if "fdh_input_vrs" in self.results:
    #         return self.results['fdh_input_vrs']
    #     results = pd.DataFrame(self.fdh_input_p2_vrs(self.fdh_input_p1_vrs()))
    #     results = is_efficient(results, 'fdh_input_vrs')
    #     self.results['fdh_input_vrs'] = results
    #     return results

    # def bcc_output_p1(self):
    #     efficiencies = []

    #     for o in range(self.n):
    #         model = Model("OutputEfficiency")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         lambdas = [model.addVar(lb=0, name=f"lambdas_{j}") for j in range(self.n)]
    #         eta = model.addVar(lb=1, name="eta")

    #         model.setObjective(eta, GRB.MAXIMIZE)

    #         for i in range(self.m):
    #             model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= eta * self.output_data[o, r])
            
    #         model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)

    #         model.optimize()
    #         efficiencies.append(model.objVal)

    #     return efficiencies

    # def fdh_output_p2_vrs(self, output_efficiencies):
    #     results = []

    #     for o in range(self.n):
    #         eta_star = output_efficiencies[o]
    #         model = Model("Phase2")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         lambdas = [model.addVar(vtype=GRB.BINARY, name=f"lambda_{j}") for j in range(self.n)]
    #         S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
    #         S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]

    #         model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)

    #         for i in range(self.m):
    #             model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] == eta_star * self.output_data[o, r])

    #         model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)

    #         model.optimize()

    #         results.append({
    #             'DMU': o,
    #             'efficiency': eta_star,
    #             'slacks_minus': [S_minus[i].X for i in range(self.m)],
    #             'slacks_plus': [S_plus[r].X for r in range(self.s)],
    #             'Lambda': [lambdas[j].X for j in range(self.n)]
    #         })

    #     return pd.DataFrame(results)

    # def fdh_output_vrs(self):
    #     if "fdh_output_vrs" in self.results:
    #         return self.results['fdh_output_vrs']

    #     results = pd.DataFrame(self.fdh_output_p2_crs(self.fdh_output_p1_crs()))
    #     results.rename(columns={
    #         'efficiency': 'n', 
    #         'slacks_minus': 't_minus',
    #         'slacks_plus':'t_plus',
    #         'Lambda':'u'}, inplace=True)
    #     results = is_efficient(results, 'fdh_output_vrs')
    #     self.results['fdh_output_vrs'] = results
    #     return results

    # def fdh_input_p1_nirs(self):
    #     efficiencies = []

    #     for o in range(self.n):
    #         model = Model("InputEfficiency")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         theta = model.addVar(lb=0, name="theta")
    #         delta = model.addVar(lb = 0, ub = 1, name='delta') 

    #         model.setObjective(theta, GRB.MINIMIZE)

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) <= theta * self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

    #         model.optimize()

    #         efficiencies.append(model.objVal)

    #     return efficiencies
    
    # def fdh_input_p2_nirs(self, input_efficiencies):
    #     results = []

    #     for o in range(self.n):
    #         theta_star = input_efficiencies[o]
    #         model = Model("Phase2")
    #         model.setParam(GRB.Param.OutputFlag, 0)
    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
    #         S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
    #         model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)
    #         delta = model.addVar(lb=0, ub=1, name='delta')

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == theta_star * self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)
    #         model.optimize()

    #         results.append({
    #             'DMU': o,
    #             'efficiency': theta_star,
    #             'slacks_minus': [S_minus[i].x for i in range(self.m)],
    #             'slacks_plus': [S_plus[r].x for r in range(self.s)],
    #             'Lambda': [z[j].x for j in range(self.n)]
    #         })


    #     return pd.DataFrame(results)

    # def fdh_input_nirs(self):
    #     results = pd.DataFrame(self.fdh_input_p2_nirs(self.fdh_input_p1_nirs()))
    #     return results

    # def fdh_output_p1_nirs(self):
    #     efficiencies = []

    #     for o in range(self.n):
    #         model = Model("OutputEfficiency")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         eta = model.addVar(lb=1, name="eta")
    #         delta = model.addVar(lb = 0, ub=1, name='delta') 

    #         model.setObjective(eta, GRB.MAXIMIZE)

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) >= eta * self.output_data[o, r])
    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

    #         model.optimize()
    #         efficiencies.append(model.objVal)

    #     return efficiencies

    # def fdh_output_p2_nirs(self, output_efficiencies):
    #     results = []

    #     for o in range(self.n):
    #         eta_star = output_efficiencies[o]
    #         model = Model("Phase2")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
    #         S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
    #         delta = model.addVar(lb=0, ub=1, name='delta')

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] == eta_star * self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

    #         model.optimize()

    #         results.append({
    #             'DMU': o,
    #             'efficiency': eta_star,
    #             'slacks_minus': [S_minus[i].X for i in range(self.m)],
    #             'slacks_plus': [S_plus[r].X for r in range(self.s)],
    #             'Lambda': [z[j].X for j in range(self.n)]
    #         })

    #     return pd.DataFrame(results)

    # def fdh_output_nirs(self):
    #     results = pd.DataFrame(self.fdh_output_p2_nirs(self.fdh_output_p1_nirs()))
    #     results.rename(columns={
    #         'efficiency': 'n', 
    #         'slacks_minus': 't_minus',
    #         'slacks_plus':'t_plus',
    #         'Lambda':'u'}, inplace=True)
    #     return results
    
    # def fdh_input_p1_ndrs(self):
    #     efficiencies = []

    #     for o in range(self.n):
    #         model = Model("InputEfficiency")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         theta = model.addVar(lb=0, name="theta")
    #         delta = model.addVar(lb = 1, name='delta') 

    #         model.setObjective(theta, GRB.MINIMIZE)

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) <= theta * self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

    #         model.optimize()

    #         efficiencies.append(model.objVal)

    #     return efficiencies
    
    # def fdh_input_p2_ndrs(self, input_efficiencies):
    #     results = []

    #     for o in range(self.n):
    #         theta_star = input_efficiencies[o]
    #         model = Model("Phase2")
    #         model.setParam(GRB.Param.OutputFlag, 0)
    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
    #         S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
    #         model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)
    #         delta = model.addVar(lb=1, name='delta')

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == theta_star * self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)
    #         model.optimize()

    #         results.append({
    #             'DMU': o,
    #             'efficiency': theta_star,
    #             'slacks_minus': [S_minus[i].x for i in range(self.m)],
    #             'slacks_plus': [S_plus[r].x for r in range(self.s)],
    #             'Lambda': [z[j].x for j in range(self.n)]
    #         })


    #     return pd.DataFrame(results)

    # def fdh_input_ndrs(self):
    #     results = pd.DataFrame(self.fdh_input_p2_ndrs(self.fdh_input_p1_ndrs()))
    #     return results

    # def fdh_output_p1_ndrs(self):
    #     efficiencies = []

    #     for o in range(self.n):
    #         model = Model("OutputEfficiency")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         eta = model.addVar(lb=1, name="eta")
    #         delta = model.addVar(lb = 1, name='delta') 

    #         model.setObjective(eta, GRB.MAXIMIZE)

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) >= eta * self.output_data[o, r])
    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

    #         model.optimize()
    #         efficiencies.append(model.objVal)

    #     return efficiencies

    # def fdh_output_p2_ndrs(self, output_efficiencies):
    #     results = []

    #     for o in range(self.n):
    #         eta_star = output_efficiencies[o]
    #         model = Model("Phase2")
    #         model.setParam(GRB.Param.OutputFlag, 0)

    #         z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
    #         S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
    #         S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
    #         delta = model.addVar(lb=1, name='delta')

    #         for i in range(self.m):
    #             model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == self.input_data[o, i])

    #         for r in range(self.s):
    #             model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] == eta_star * self.output_data[o, r])

    #         model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

    #         model.optimize()

    #         results.append({
    #             'DMU': o,
    #             'efficiency': eta_star,
    #             'slacks_minus': [S_minus[i].X for i in range(self.m)],
    #             'slacks_plus': [S_plus[r].X for r in range(self.s)],
    #             'Lambda': [z[j].X for j in range(self.n)]
    #         })

    #     return pd.DataFrame(results)

    # def fdh_output_ndrs(self):
    #     results = pd.DataFrame(self.fdh_output_p2_ndrs(self.fdh_output_p1_ndrs()))
    #     results.rename(columns={
    #         'efficiency': 'n', 
    #         'slacks_minus': 't_minus',
    #         'slacks_plus':'t_plus',
    #         'Lambda':'u'}, inplace=True)
    #     return results