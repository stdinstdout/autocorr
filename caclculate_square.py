import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

# TODO: Доделай!!!!

def get_info(filepath):
    """[summary]

    Args:
        filepath ([type]): [description]

    Returns:
        [type]: [description]
    """    
    df = pd.read_csv(filepath)
    info = df.iloc[0:5]
    del df

    cell_size = float(info.iloc[3].values[0].split()[1])
    corner_x, corner_y = float(info.iloc[1].values[0].split()[1]), float(info.iloc[2].values[0].split()[1])
    ncols = int(info.columns[0].split()[1])
    nrows = int(info.iloc[0][0].split()[1])
    
    return dict([('cell_size', cell_size), ('corner_x', corner_x), ('corner_y', corner_y), ('ncols', ncols), ('nrows', nrows)])

