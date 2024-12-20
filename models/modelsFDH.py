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


class FDH:
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
    
    def plot_fdh(self, typ):  
        if self.input_data.shape[1] != 1 or self.output_data.shape[1] != 1:
            raise ValueError("Unsupported combination of M and S")

        fig, ax = plt.subplots()
        x = self.input_data.flatten()
        y = self.output_data.flatten()
        ax.scatter(x, y, c='blue')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        title = 'Input vs Output'
        max_x, max_y = np.max(x), np.max(y)
        min_x, min_y = np.min(x), np.min(y)

        typ2func = {
            'fdh_input_vrs': self.fdh_input_vrs,
            'fdh_output_vrs': self.fdh_output_vrs,
        }

        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            return

        if typ not in self.results:
            print(f"Calculating results for {typ}...")
            results = typ2func[typ]()  
            self.results[typ] = results
        else:
            print(f"Using cached results for {typ}...")
            results = self.results[typ]

        efficient_points = []
        for i, efficient in enumerate(results['is_efficient']):
            if efficient:
                efficient_points.append((x[i], y[i]))
                circle = plt.Circle((x[i], y[i]), radius=0.05, fill=False, edgecolor='r', linewidth=2)  
                ax.add_artist(circle)

        intersection_points = []
        for i in range(len(efficient_points)):
            for j in range(i+1, len(efficient_points)):
                x1, y1 = efficient_points[i]
                x2, y2 = efficient_points[j]
                intersection = (max(x1, x2), min(y1, y2))
                intersection_points.append((intersection, efficient_points[i], efficient_points[j]))

        unique_intersections = {}
        for point in intersection_points:
            intersection, point1, point2 = point
            x, y = intersection

            if x in unique_intersections:
                if y > unique_intersections[x][0][1]:
                    unique_intersections[x] = (intersection, point1, point2)
            else:
                unique_intersections[x] = (intersection, point1, point2)

        y_dict = {}
        for x, (intersection, point1, point2) in unique_intersections.items():
            y = intersection[1]
            if y in y_dict:
                if x < y_dict[y][0][0]:
                    y_dict[y] = (intersection, point1, point2)
            else:
                y_dict[y] = (intersection, point1, point2)

        for intersection, point1, point2 in y_dict.values():
            ax.plot([intersection[0], point1[0]], [intersection[1], point1[1]], 'g-', linewidth=1)
            ax.plot([intersection[0], point2[0]], [intersection[1], point2[1]], 'g-', linewidth=1)

        rightmost_point = max(efficient_points, key=lambda point: point[0])
        downmost_point = min(efficient_points, key=lambda point: point[1])

        ax.plot([rightmost_point[0], 1.1 * max_x], [rightmost_point[1], rightmost_point[1]], 'g-', linewidth=1)
        ax.plot([downmost_point[0], downmost_point[0]], [downmost_point[1], 0], 'g-', linewidth=1)

        ax.set_xlim(0, max_x * 1.1)
        ax.set_ylim(0, max_y * 1.1)
        ax.set_title(title)
        plt.show()


    def rdm_fdh(self):
        if "rdm_fdh" in self.results:
            return self.results['rdm_fdh']

        results = []

        for o in range(self.n):
            model = Model("RDM_Model")
            model.setParam(GRB.Param.OutputFlag, 0)

            R_minus = [self.input_data[o, i] - min(self.input_data[:, i]) for i in range(self.m)]
            R_plus = [max(self.output_data[:, r]) - self.output_data[o, r] for r in range(self.s)]

            lambdas = [model.addVar(vtype=GRB.BINARY, name=f"lambda_{j}") for j in range(self.n)]
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
        results = is_efficient(results, 'rdm_fdh')
        self.results['rdm_fdh'] = results
        return results
