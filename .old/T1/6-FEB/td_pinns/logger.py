import os
import pandas as pd

class Logger:
    def __init__(self,
                 filename_data: str = "data_res.csv",
                 filename_collocation: str = "data_collocation.csv"):
        self.filename_data = filename_data
        self.filename_collocation = filename_collocation

        # Si existen, abrir; si no, DataFrame vacío
        if os.path.exists(self.filename_data):
            self.df_data = pd.read_csv(self.filename_data)
        else:
            self.df_data = pd.DataFrame()

        if os.path.exists(self.filename_collocation):
            self.df_collocation = pd.read_csv(self.filename_collocation)
        else:
            self.df_collocation = pd.DataFrame()

    def add_data_series(self, method, time, real, estimated, dt, nu, time_iteration):
        """
        Crea un DataFrame con columnas:
        method, time, real, estimated, dt, nu, time_iteration
        y lo concatena/guarda en self.filename_data
        (sin validaciones adicionales).
        """
        new_df = pd.DataFrame({
            "method": method,
            "time": time,
            "real": real,
            "estimated": estimated[:,0],
            "dt": dt,
            "nu": nu,
            "time_iteration": time_iteration
        })

        self.df_data = pd.concat([self.df_data, new_df], ignore_index=True) if not self.df_data.empty else new_df
        # Asegurar carpeta
        os.makedirs(os.path.dirname(self.filename_data) or ".", exist_ok=True)
        self.df_data.to_csv(self.filename_data, index=False)

        return self.df_data

    def add_collocation_data(self, method, point_x, time):
        """
        Crea un DataFrame con columnas:
        method, point_x, time
        y lo concatena/guarda en self.filename_collocation
        (sin validaciones adicionales).
        """
        new_df = pd.DataFrame({
            "method": method,
            "point_x": point_x,
            "time": time
        })

        self.df_collocation = pd.concat([self.df_collocation, new_df], ignore_index=True) if not self.df_collocation.empty else new_df
        os.makedirs(os.path.dirname(self.filename_collocation) or ".", exist_ok=True)
        self.df_collocation.to_csv(self.filename_collocation, index=False, engine="openpyxl")

        return self.df_collocation
