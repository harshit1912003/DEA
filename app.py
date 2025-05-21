import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QAbstractItemView, QLineEdit, QComboBox, QTableWidget, QTableWidgetItem
)
from typing import List
from models.modelsDEA import DEA
from models.modelsFDH import FDH
from models.modelsNH import Non_Homo
from utils.datainput import xlsx2matrix
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class DataInputApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Input App")
        self.setGeometry(100, 100, 600, 400)

        self.file_path = None
        self.column_headers = []
        self.x = None
        self.y = None

        self.dea = None
        self.fdh = None
        self.nh = None

        self.init_ui()

    def init_ui(self):

        layout = QVBoxLayout()

        self.upload_button = QPushButton("Upload Excel File")
        self.upload_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_button)

        self.file_label = QLabel("No file selected")
        layout.addWidget(self.file_label)

        self.input_label = QLabel("Select Input Columns:")
        self.input_list = QListWidget()
        self.input_list.setSelectionMode(QAbstractItemView.MultiSelection)

        self.output_label = QLabel("Select Output Columns:")
        self.output_list = QListWidget()
        self.output_list.setSelectionMode(QAbstractItemView.MultiSelection)

        layout.addWidget(self.input_label)
        layout.addWidget(self.input_list)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_list)

        self.process_button = QPushButton("Confirm Selection")
        self.process_button.clicked.connect(self.process_selection)
        layout.addWidget(self.process_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Excel File", "", "Excel Files (*.xlsx)", options=options
        )

        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"File Selected: {file_path}")
            self.load_columns()

    def load_columns(self):
        try:
            df = pd.read_excel(self.file_path, nrows=1)
            self.column_headers = df.columns.tolist()

            self.input_list.clear()
            self.output_list.clear()

            self.input_list.addItems(self.column_headers)
            self.output_list.addItems(self.column_headers)
        except Exception as e:
            self.file_label.setText(f"Error loading file: {e}")

    def open_model_selection(self):
        self.model_selection_window = ModelSelectionWindow(self.x, self.y, self.dea, self.fdh, self.nh)
        self.model_selection_window.show()
        self.close()

    def process_selection(self):
        input_columns = [item.text() for item in self.input_list.selectedItems()]
        output_columns = [item.text() for item in self.output_list.selectedItems()]

        if set(input_columns) & set(output_columns):
            self.file_label.setText("Input and Output columns must be mutually exclusive.")
        else:
            try:
                self.x, self.y = xlsx2matrix(self.file_path, input_columns, output_columns)

                self.dea = DEA(self.x, self.y)
                self.fdh = FDH(self.x, self.y)
                self.nh = Non_Homo(self.x, self.y)

                self.open_model_selection()
            except Exception as e:
                self.file_label.setText(f"Error processing file: {e}")

class ModelSelectionWindow(QMainWindow):
    def __init__(self, x, y, dea, fdh, nh):
        super().__init__()
        self.setWindowTitle("Select Model")
        self.setGeometry(100, 100, 600, 600)

        self.x = x
        self.y = y
        self.dea = dea
        self.fdh = fdh
        self.nh = nh

        self.n, self.m = x.shape
        self.s = y.shape[1]

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.model_selector = QComboBox()
        self.model_selector.addItem("DEA")
        self.model_selector.addItem("FDH")
        self.model_selector.addItem("NH")
        self.model_selector.currentTextChanged.connect(self.update_model_list)

        layout.addWidget(self.model_selector)

        self.available_models_label = QLabel("Available Models:")
        layout.addWidget(self.available_models_label)

        self.models_list_widget = QListWidget()
        layout.addWidget(self.models_list_widget)

        self.run_model_button = QPushButton("Run Model")
        self.run_model_button.clicked.connect(self.run_model)
        layout.addWidget(self.run_model_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_model_list()

    def update_model_list(self):
        model_type = self.model_selector.currentText()
        if model_type == "DEA":
            self.available_models = ["ccr_input", "ccr_output" ,"bcc_input", "bcc_output", "sbm", "add", "rdm"]
            self.model = self.dea
        elif model_type == "FDH":
            self.available_models = ["fdh_input_crs", "fdh_input_vrs", "fdh_output_crs", "fdh_output_vrs", "rdm_fdh"]
            self.model = self.fdh
        elif model_type == "NH":
            self.available_models = ["nhmodel1", "nhmodel2"]
            self.model = self.nh

        self.models_list_widget.clear()  
        self.models_list_widget.addItems(self.available_models)  

    def run_model(self):
        model_name = self.models_list_widget.currentItem().text()  
        if model_name:
            try:
                result = getattr(self.model, model_name)()
                if self.m == 1 and self.s == 1 and self.model != "NH":
                    graph = self.model.plot_with_frontier(str(model_name))
                elif (self.m == 1 and self.s == 2) or (self.m == 2 and self.s == 1): #!!!!!!! 
                    graph = self.model.plot3d(str(model_name))
                else:
                    graph = None

                self.results_window = ResultsWindow(result, graph)
                self.results_window.show()
            except Exception as e:
                error_label = QLabel(f"Error running model: {e}")
                self.setCentralWidget(error_label)

# class ResultsWindow(QMainWindow):
#     def __init__(self, dataframe, graph):
#         super().__init__()
#         self.setWindowTitle("Results")
#         self.setGeometry(150, 150, 800, 600)
#         self.graph = graph
#         self.dataframe = dataframe
#         self.init_ui()

#     def init_ui(self):
#         layout = QVBoxLayout()

#         self.table_widget = QTableWidget()
#         self.table_widget.setRowCount(self.dataframe.shape[0])
#         self.table_widget.setColumnCount(self.dataframe.shape[1])
#         self.table_widget.setHorizontalHeaderLabels(self.dataframe.columns)

#         for i in range(self.dataframe.shape[0]):
#             for j in range(self.dataframe.shape[1]):
#                 self.table_widget.setItem(i, j, QTableWidgetItem(str(self.dataframe.iat[i, j])))

#         layout.addWidget(self.table_widget)

#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)


class ResultsWindow(QMainWindow):
    def __init__(self, dataframe, graph=None):
        super().__init__()
        self.setWindowTitle("Results")
        self.setGeometry(150, 150, 800, 600)
        self.graph = graph
        self.dataframe = dataframe
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        if self.graph is not None:
            self.canvas = FigureCanvas(self.graph)
            layout.addWidget(self.canvas)

        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(self.dataframe.shape[0])
        self.table_widget.setColumnCount(self.dataframe.shape[1])
        self.table_widget.setHorizontalHeaderLabels(self.dataframe.columns)

        for i in range(self.dataframe.shape[0]):
            for j in range(self.dataframe.shape[1]):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(self.dataframe.iat[i, j])))

        layout.addWidget(self.table_widget)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = DataInputApp()
    main_window.show()
    sys.exit(app.exec_())
