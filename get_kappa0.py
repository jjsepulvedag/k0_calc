import numpy as np
import pandas as pd
import scipy.stats as st

def k0_LR(df_kappa, colStn, colDist, colKappa):
    '''
    df_kappa: pandas dataframe with (at least) these three columns
              [eventID, rrup, kappa]
    '''

    allStns = df_kappa[colStn].unique()
    
    all_k0 = pd.DataFrame(columns=['stationID', 'k0_mean', 'k0_stderr', 'kr_mean', 'kr_stderr'])
    all_k0['stationID'] = allStns
    all_k0.set_index('stationID', inplace=True)

    for stn in allStns: 
        temp_df = df_kappa[df_kappa[colStn]==stn]
        lin_reg = st.linregress(temp_df[colDist], temp_df[colKappa])

        all_k0.loc[stn, 'k0_mean'] = np.round(lin_reg.intercept, 5)
        all_k0.loc[stn, 'k0_stderr'] = np.round(lin_reg.intercept_stderr, 5)
        all_k0.loc[stn, 'kr_mean'] = np.round(lin_reg.slope, 5)
        all_k0.loc[stn, 'kr_stderr'] = np.round(lin_reg.stderr, 5)

    all_k0.reset_index(inplace=True)

    return all_k0

def k0_RR():

    return None


if __name__=='__main__':

    print('hello world JJSG')