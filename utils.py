import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import DEA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
import pickle
import tkinter as tk
from tkinter import filedialog

def arr2matrix(x, y):
    input_data = np.array(x).T
    output_data = np.array(y).T
    return input_data, output_data


def load_results():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    if file_path:  
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        print(f"Results loaded from {file_path}")
        return results
    else:
        print("Load operation canceled.")
        return None
    

def xlsx2matrix(file_path, x, y):
    df = pd.read_excel(file_path)
    input_data = np.array(df[x]) 
    output_data = np.array(df[y]) 
    return input_data, output_data


def initializeZ(N, M, S):    
    input_array = np.random.randint(1, 21, size=(N, M))
    output_array = np.random.randint(1, 21, size=(N, S))
    return input_array, output_array

def initializeUnif(N, M, S):    
    input_array = np.random.uniform(1.0, 21.0, size=(N, M))
    output_array = np.random.uniform(1.0, 21.0, size=(N, S))
    return input_array, output_array
#######
def round_elements(val):
    if isinstance(val, float):
        return f"{val:.2f}"
    elif isinstance(val, (list, np.ndarray)):
        return [f"{v:.2f}" if isinstance(v, float) else v for v in val]
    else:
        return val

def handle_commas(text):
    return str(text).replace(',', r'\,')

def df2latex(df):
    df = df.applymap(round_elements)

    latex_code = r"""
\renewcommand{\ttdefault}{pcr} % Set Consolas-like font (use pcr as an example)
\begin{table}[H] % Use [H] to force the table placement (requires \usepackage{float})
    \centering
    \small % Reduce text size to small; you can also use \scriptsize for an even smaller size
    \renewcommand{\arraystretch}{1.2} % Increase row spacing for better readability
    \setlength{\tabcolsep}{8pt} % Adjust column separation for a narrower table
    \resizebox{\textwidth}{!}{ % Resize the table to fit the text width
        \begin{tabular}{|""" + '|'.join(['c'] * len(df.columns)) + r"""|}
            \hline
    """

    latex_code += r"            " + " & ".join([f"\\texttt{{{col}}}" for col in df.columns]) + r" \\" + "\n" + r"            \hline" + "\n"

    for _, row in df.iterrows():
        latex_code += r"            " + " & ".join(
            [f"\\texttt{{{handle_commas(item)}}}" if not isinstance(item, list)
            else "\\texttt{" + ', '.join(map(handle_commas, item)) + "}" for item in row]) + r" \\" + "\n"

    latex_code += r"""            \hline
        \end{tabular}
    }
\end{table}
"""

    print(latex_code)

def plot(input_array, output_array, types, efficiency_func): 
    fig, ax = plt.subplots()

    N, M = input_array.shape
    N, S = output_array.shape

    if M == 1 and S == 1:
        x = input_array.flatten()
        y = output_array.flatten()
        ax.scatter(x, y, c='blue')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        title = 'Input vs Output'

    elif M == 2 and S == 1:
        x = input_array[:, 0] / output_array.flatten()
        y = input_array[:, 1] / output_array.flatten()
        ax.scatter(x, y, c='green')
        ax.set_xlabel('Input1 / Output')
        ax.set_ylabel('Input2 / Output')
        title = 'Input1/Output vs Input2/Output'

    elif M == 1 and S == 2:
        x = output_array[:, 0] / input_array.flatten()
        y = output_array[:, 1] / input_array.flatten()
        ax.scatter(x, y, c='red')
        ax.set_xlabel('Output1 / Input')
        ax.set_ylabel('Output2 / Input')
        title = 'Output1/Input vs Output2/Input'

    else:
        print("Unsupported combination of M and S")
        return

    dea = DEA(input_array, output_array)

    typ2func = {
        'ccr_input': dea.ccr_input,  
        'ccr_output': dea.ccr_output,
        'sbm': dea.sbm,
        'add': dea.add,
        'bcc_input': dea.bcc_input,
        'bcc_output': dea.bcc_output,
    }

    for typ in types:
        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            continue

        res = typ2func[typ]()
        results = efficiency_func(res, typ)

        for i, efficient in enumerate(results['is_efficient']):
            if efficient:
                circle = plt.Circle((x[i], y[i]), radius=0.05, fill=False, ec='r')
                ax.add_artist(circle)
                ax.text(x[i], y[i] + 0.02, typ, color='black', fontsize=10, ha='center')

    ax.set_title(title)
    plt.show()

def plot3d(input_array, output_array, types, efficiency_func):
    fig = plt.figure()

    N, M = input_array.shape
    N, S = output_array.shape

    if M == 1 and S == 1:

        ax = fig.add_subplot(111)
        x = input_array.flatten()
        y = output_array.flatten()
        ax.scatter(x, y, c='blue')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        title = 'Input vs Output'

    elif M == 2 and S == 1:

        ax = fig.add_subplot(111, projection='3d')
        x = input_array[:, 0]
        y = input_array[:, 1]
        z = output_array.flatten()
        ax.scatter(x, y, z, c='green')
        ax.set_xlabel('Input1')
        ax.set_ylabel('Input2')
        ax.set_zlabel('Output')
        title = 'Input1 vs Input2 vs Output'

    elif M == 1 and S == 2:

        ax = fig.add_subplot(111, projection='3d')
        x = input_array.flatten()
        y = output_array[:, 0]
        z = output_array[:, 1]
        ax.scatter(x, y, z, c='red')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output1')
        ax.set_zlabel('Output2')
        title = 'Input vs Output1 vs Output2'

    else:
        raise ValueError("Unsupported combination of M and S")

    dea = DEA(input_array, output_array)

    typ2func = {
        'ccr_input': dea.ccr_input,  
        'ccr_output': dea.ccr_output,
        'sbm': dea.sbm,
        'add': dea.add,
        'bcc_input': dea.bcc_input,
        'bcc_output': dea.bcc_output
    }

    for typ in types:
        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            continue

        res = typ2func[typ]()
        results = efficiency_func(res, typ)

        for i, efficient in enumerate(results['is_efficient']):
            if efficient:

                ax.scatter(x[i], y[i], z[i], s=200, facecolors='none', edgecolors='r', linewidths=2)
                ax.text(x[i], y[i], z[i] + 0.02, typ, color='black', fontsize=10, ha='center')

    ax.set_title(title)
    ax.grid(True)
    plt.show()

def perpendicular_distance(point, line_point1, line_point2):
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

def extend_line(p1, p2, max_x):
    if p2[0] == p1[0]:  
        return (p2[0], max_x * (p2[1] - p1[1]) / (p2[0] - p1[0]) + p1[1])
    else:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        return (max_x, slope * (max_x - p1[0]) + p1[1])

def plotwithfrontier(input_array, output_array, types, efficiency_func):
    if input_array.shape[1] != 1 or output_array.shape[1] != 1 :
        raise ValueError("Unsupported combination of M and S")

    fig, ax = plt.subplots()

    x = input_array.flatten()
    y = output_array.flatten()
    ax.scatter(x, y, c='blue')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    title = 'Input vs Output'
    fake_origin = (np.min(x), np.max(y))

    max_x, max_y = np.max(x), np.max(y)

    dea = DEA(input_array, output_array)
    typ2func = {
        'ccr_input': dea.ccr_input,
        'ccr_output': dea.ccr_output,
        'sbm': dea.sbm,
        'add': dea.add,
        'bcc_input': dea.bcc_input,
        'bcc_output': dea.bcc_output,
    }

    for typ in types:
        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            continue
        res = typ2func[typ]()
        results = efficiency_func(res, typ)

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
                dist = perpendicular_distance(fake_origin, edge[0], edge[1])
                edges.append((edge, dist))

            edges.sort(key=lambda x: x[1])
            selected_edges = edges[:len(efficient_points)-1]

            for edge, _ in selected_edges:
                ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'r-', linewidth=2)

        elif len(efficient_points) == 2:
            ax.plot([efficient_points[0][0], efficient_points[1][0]],
                    [efficient_points[0][1], efficient_points[1][1]], 'r-', linewidth=2)

        elif len(efficient_points) == 1:

            extended_point = extend_line((0, 0), efficient_points[0], max_x)
            ax.plot([0, extended_point[0]], [0, extended_point[1]], 'r-', linewidth=2)

    ax.set_xlim(0, max_x * 1.1)  
    ax.set_ylim(0, max_y * 1.1)  
    ax.set_title(title)
    plt.show()

def perpendicular_distance(point, line_point1, line_point2):
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

def extend_line(p1, p2, max_x):
    if p2[0] == p1[0]:
        return (p2[0], max_x * (p2[1] - p1[1]) / (p2[0] - p1[0]) + p1[1])
    else:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        return (max_x, slope * (max_x - p1[0]) + p1[1])

def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None  

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y)

def plot_fdh(input_array, output_array, types, efficiency_func):  # works only for vrs fdh
    if input_array.shape[1] != 1 or output_array.shape[1] != 1 :
        raise ValueError("Unsupported combination of M and S")


    fig, ax = plt.subplots()
    x = input_array.flatten()
    y = output_array.flatten()
    ax.scatter(x, y, c='blue')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    title = 'Input vs Output'
    max_x, max_y = np.max(x), np.max(y)
    min_x, min_y = np.min(x), np.min(y)
    dea = DEA(input_array, output_array)

    typ2func = {
        'fdh_input_vrs': dea.fdh_input_vrs,
        'fdh_output_vrs': dea.fdh_output_vrs,
    }
    for typ in types:
        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            continue
        res = typ2func[typ]()
        results = efficiency_func(res, typ)
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

