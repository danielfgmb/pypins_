import os
import pandas as pd


class Logger:
    def __init__(self, filename='data_res.csv'):
        self.filename = filename
        self.df = pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame()

    def add(self, method, time_vals, real, estimated, dt, nu, elapsed):
        new = pd.DataFrame({
            'method': method,
            'time': time_vals,
            'real': real,
            'estimated': estimated[:, 0],
            'dt': dt,
            'nu': nu,
            'time_iteration': elapsed,
        })
        self.df = pd.concat([self.df, new], ignore_index=True) if not self.df.empty else new
        os.makedirs(os.path.dirname(self.filename) or '.', exist_ok=True)
        self.df.to_csv(self.filename, index=False)
        return self.df
