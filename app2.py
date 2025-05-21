import streamlit as st
import pandas as pd
from models.modelsDEA import DEA
from models.modelsFDH import FDH
from models.modelsNH import Non_Homo
from utils.datainput import xlsx2matrix
import matplotlib.pyplot as plt

def main():
    st.title("Data Envelopment Analysis App")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df_sample = pd.read_excel(uploaded_file, nrows=1)
            column_headers = df_sample.columns.tolist()
            st.write("Columns found:", column_headers)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return

        input_columns = st.multiselect("Select Input Columns", options=column_headers)
        output_columns = st.multiselect("Select Output Columns", options=column_headers)

        if st.button("Confirm Selection"):
            if set(input_columns) & set(output_columns):
                st.error("Input and Output columns must be mutually exclusive.")
            elif not input_columns or not output_columns:
                st.error("Please select at least one column for both input and output.")
            else:
                try:
                    uploaded_file.seek(0)
                    x, y = xlsx2matrix(uploaded_file, input_columns, output_columns)
                    st.success("File processed successfully!")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    return

                dea = DEA(x, y)
                fdh = FDH(x, y)
                nh = Non_Homo(x, y)

                n, m = x.shape
                s = y.shape[1]
                print(n, m, s)
                
                model_type = st.selectbox("Select Model Type", options=["DEA", "FDH", "NH"])
                if model_type == "DEA":
                    available_models = ["ccr_input", "ccr_output", "bcc_input", "bcc_output", "sbm", "add", "rdm"]
                    selected_model = dea
                elif model_type == "FDH":
                    available_models = ["fdh_input_crs", "fdh_input_vrs", "fdh_output_crs", "fdh_output_vrs", "rdm_fdh"]
                    selected_model = fdh
                elif model_type == "NH":
                    available_models = ["nhmodel1", "nhmodel2"]
                    selected_model = nh

                model_function = st.selectbox("Select Model Function", options=available_models)

                # Step 4: Run model on button click
                if st.button("Run Model"):
                    try:
                        # Dynamically call the selected model function
                        result = getattr(selected_model, model_function)()
                        # Decide which plot to generate based on matrix shapes
                        graph = None
                        if m == 1 and s == 1 and model_type != "NH":
                            graph = selected_model.plot_with_frontier(model_function)
                        elif (m == 1 and s == 2) or (m == 2 and s == 1):
                            graph = selected_model.plot3d(model_function)
                        else:
                            graph = None

                        st.subheader("Results")
                        st.dataframe(result)

                        if graph is not None:
                            st.pyplot(graph)
                    except Exception as e:
                        st.error(f"Error running model: {e}")

if __name__ == "__main__":
    main()