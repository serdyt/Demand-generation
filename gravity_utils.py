import pandas as pd
import numpy as np

import os, sys
extra_path = os.path.join(sys.prefix, '/home/ai6644/anaconda3/envs/python3.7/lib/python3.7/site-packages')
if os.path.isdir(extra_path) and extra_path not in sys.path:
    sys.path.append(extra_path)
import shapefile

def fratar_double_constrained(prodA, attrA, cost_matrix, num_iter=100):
    """Performs matrix balancing
    all credits to https://github.com/joshchea/python-tdm

    Parameters
    ----------
    prodA : numpy array of productions by zone
    attrA : numpy array of attractions by zone
    cost_matrix : cost matrix for travels between the zones
    num_iter : a safe number of iterations (10 are working fine)
        ideally we should check if matrix balancing changes trips significantly on each iteration
    """
    trips = np.zeros((len(prodA), len(prodA)))
    print('Checking production, attraction balancing:')
    sumP = sum(prodA)
    sumA = sum(attrA)
    print('Production: ', sumP)
    print('Attraction: ', sumA)
    if sumP != sumA:
        print('Productions and attractions do not balance, attractions will be scaled to productions!')
        attrA = attrA * (sumP / sumA)
        attrT = attrA.copy()
        prodT = prodA.copy()
    else:
        print('Production, attraction balancing OK.')
        attrT = attrA.copy()
        prodT = prodA.copy()
    
    for _ in range(0, num_iter):
        for i in range(0,len(prodT)):
            trips[i,:] = prodA[i] * attrA * cost_matrix[i, :] / max(0.000001, sum(attrA * cost_matrix[i, :]))
    
        #Run 2D balancing --->
        computed_attractions = trips.sum(0)
        computed_attractions[computed_attractions==0]=1
        attrA = attrA * (attrT / computed_attractions)
    
        computed_productions = trips.sum(1)
        computed_productions[computed_productions==0]=1
        prodA = prodA * (prodT / computed_productions)
    
    
    for i in range(0, len(prodA)):
        trips[i, :] = prodA[i] * attrA * cost_matrix[i, :] / max(0.000001, sum(attrA * cost_matrix[i, :]))
        
    return trips


def draw_population(df, weights):
    df2 = df.loc[np.repeat(df.index.values, weights)]
    return df2


def read_shapefile(shp_path):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' column holding
    the geometry information. This uses the pyshp package
    """


    #read file, parse out the records and shapes
    sf = shapefile.Reader(shp_path)
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]

    #write into a dataframe
    df = pd.DataFrame(columns=fields, data=[list(x) for x in records])
    df = df.assign(coords=shps)

    return df