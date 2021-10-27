import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances


def get_df(filepath, col_names, name=None):
    """ Return dataframe with longitude,latitude coorinates for every occurances.
        Data file is txt or csv.
        If data file is occurances.txt from gbif.com. 
            then col_names is ['scientificName','decimalLongitude','decimalLatitude']

    Args:
        file_path (string): path to data file 
        col_names (list): list with 3 element:
            [0] (string) - a name of columns with species name in data file,
            [1] (string) - a name of columns with latitude coordinate in data file,
            [2] (string) - a name of columns with longitude coordinate in data file.
        name (string): - The name in species columns; if None, the name will be get from file.

    Returns:
        [typle]:
            [0] (string) - name of species,
            [1] (pd.DataFrame) - 1col: (float) Longitute
                                 2col: (float) Latitude
    """    
    df = pd.read_table(filepath, float_precision='high') \
        if filepath.split('.')[-1]=='txt' \
        else pd.read_csv(filepath, float_precision='high')
   
    df = df[[col_names[0], col_names[1], col_names[2]]]

    if name == None:
        name = df.iloc[0,0]
    
    df.columns = ['Species','Latitude','Longitude']
    df = df[['Longitude','Latitude']]
    
    
    df = df.dropna()
    df = df.drop_duplicates()
     
    df = df.sort_values('Longitude')
    
    return name, df

def transform_to_array(df):
    """Transform dataframe to numpy array

    Args:
        df (pd.DataFrame): df from functions get_df_from_txt and get_df_from_csv

    Returns:
        [numpy.array]: 2 dimensions array, 1d - points, 2d - coordinates of point (longitute, latitude)
    """    
    coor_array = np.array(df[['Longitude','Latitude']])
    return coor_array

def dbscan_clustering(points, epsilon):
    """Clustering points

    Args:
        points (2d numpy array): array of points
        epsilon (float): distance between neighbours points 

    Returns:
        [pandas Series]: Series, where every element is number of cluster for every point
    """    
    db = DBSCAN(eps=epsilon/6371., min_samples=1, metric='haversine', algorithm='auto').fit(np.radians(points))
    
    return pd.Series(db.labels_)

def choose_point(points):
    """Chose only one points from one cluster points

    Args:
        points (numpy array): one cluster points

    Returns:
        [numpy array]: one point
    """    
    rad_points = np.radians(points)
    index = haversine_distances(rad_points).sum(axis=1).argmin()
    return points[index]

def auto_correction_data(coor_array, _labels):
    """Auto corralarion of spartial data

    Args:
        coor_array (numpy array): points of occurances
        _labels (pandas series): label of clusters for every point

    Returns:
        [numpy array]: points where in one cluster only one point
    """    
    final_array = np.array(list())
    unique_labels = _labels.unique()
    for label in unique_labels:
        indexes = _labels[_labels==label].index
        final_array = np.append(final_array, choose_point(coor_array[indexes]), axis=0)
        
    return final_array

def make_final_df(final_coor, name):
    """Create data frame with 3 columns: species(name), longitude, latitude 
       Suitable for maxent samples

    Args:
        final_coor (numpy array): numpy array from auto_corralarion_data function
        name (string): name of species

    Returns:
        [pandas DataFrame]: dataframe with 3 columns: species(name), longitude, latitude
    """    
    df = pd.DataFrame(final_coor)
    df['species'] = name
    df.columns = ['longitude','latitude','species']
    df = df[['species','longitude','latitude']]
    return df

def write_csv(path, name, df):
    """Write csv file with data in df file

    Args:
        path (string): path where you would like to save file
        name (string): name of saving file
        df (pandas DataFrame): dataframe with data you would like to write in csv
    """        
    outfile_name = path+"\\"+name+".csv"
    df.to_csv(outfile_name, index=False)

def pipeline(path, col_names, epsilon):
    format = path.split('.')[-1]

    if format not in ['txt','csv']:
        raise Exception("Input format isn't suitable for this function")

    name, df = get_df(path, col_names)

    coor_array = transform_to_array(df)
    del df

    _labels = dbscan_clustering(coor_array, epsilon)
    final_df = auto_correction_data(coor_array, _labels)
    final_df = np.reshape(final_df, (-1,2))
    
    return make_final_df(final_df, name)
        
    
    