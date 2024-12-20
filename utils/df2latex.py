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

