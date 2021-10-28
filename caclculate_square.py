import pandas as pd
import numpy as np

def get_info(path):
    df = pd.read_csv(path, header=None, nrows=6)
    
    return dict([
        ('cell_size', float(df.iloc[4].values[0].split()[1])),
        ('corner_x', float(df.iloc[2].values[0].split()[1])),
        ('corner_y', float(df.iloc[3].values[0].split()[1])),
        ('nrows', int(df.iloc[1][0].split(' ')[-1])),
        ('ncols', int(df.iloc[0][0].split(' ')[-1])),
        ('nodata_value', np.float(df.iloc[5][0].split(' ')[-1]))
    ])

def get_nodata_value_and_cell_size(path):
    info = np.loadtxt(path, max_rows=6, dtype=str)
    cell_size = np.float(info[4][1])
    nodata_value = np.int(info[5][1])
    return cell_size, nodata_value

def get_square_of_one_cell(cell_size):
    p = np.pi/180
    a = 0.5 - np.cos((cell_size) * p) / 2
    return np.power(12742 * np.arcsin(np.sqrt(a)),2)

def get_square_all_cells(path):
    
    cell_size, nodata_value = get_nodata_value_and_cell_size(path)
    
    flooar_mult10 = lambda x: -9999 if x == -9999 else np.floor(x * 10)
    vfunc = np.vectorize(flooar_mult10)
    data = vfunc(np.loadtxt(path, skiprows=6))
    
    value, labels = list(np.histogram(data ,np.unique(data)))
    
    one_cell_square = get_square_of_one_cell(cell_size)
    value = value * one_cell_square
    
    labels = np.array([ (np.round(x / 10, 1), np.round(x / 10 + 0.1, 1)) if not x == -9999 and not x == 10 else x for x in labels ], dtype=object)
    if -9999 in labels:
        labels = np.where(labels == -9999, 'NODATA', labels)
    if 10 in labels:
        labels = np.where(labels == 10, 1, labels)
    
    df = pd.DataFrame([labels, value])
    df = df.fillna(value=0)
    
    return df

def write_csv(df, name, path=None):
    outfile = name + ".csv" if path == None else path + "\\" + name + ".csv"
    df.to_csv(outfile, header=False, index=False)
