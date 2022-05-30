import numpy as np


def extra_split(df,
                target_param_name, smiles_column="SMILES",
                spl_ratio=0.95,
                top_spl_ratio=1,
                ):
    """
    prepare train and test dataset in a special way

    Parameters
    ----------
    df : pandas dataframe
        database
    target_param_name: string
        parameter name to be used as y, recorded in the column of dataframe
    spl_ratio: float
        (spl_ratio)% of the random records will be used as train
    top_spl_ratio:
        (top_spl_ratio)% of the records with highest y will be used as test


    Returns
    -------
    tr_X: np array of float
        train X
    tr_y: np array of float
        train y
    te_X: np array of float
        train X
    te_y: np array of float
        train y        
    """

    df = df.sort_values(by=target_param_name)

    total_records = df.shape[0]

    # use top 10% rec and random10% rec for test
    top_spl_pos = int(top_spl_ratio*total_records)
    temp_df = df.sort_values(by=target_param_name)
    top_df = temp_df[top_spl_pos:]
    other_df = temp_df[:top_spl_pos].sample(frac=1)
    target_df = other_df.append(top_df)

    spl_pos = int((spl_ratio)*target_df.shape[0])

    tr_df = target_df[:spl_pos]
    te_df = target_df[spl_pos:]

    if smiles_column is None:
        tr_X=np.array(tr_df.drop(target_param_name,axis=1))
        te_X=np.array(te_df.drop(target_param_name,axis=1))
    else:
        tr_X = np.array(tr_df[smiles_column])
        te_X = np.array(te_df[smiles_column])
    tr_y = np.array(tr_df[target_param_name])
    te_y = np.array(te_df[target_param_name])

    return tr_X, te_X, tr_y, te_y


# calc similarity
def compare_similarity(res, i, fp_list):
    """
    compare similarity of a) a specific fingerprint and b) list of fingerprints

    Parameters
    ----------
    res : np array of float
        float array, containing fingerprint information of a)
    i: int
        target fingerprint a) is recorded in the i-th column of res
    fp_list: list of int
        list of fingerprints b), which should be compared with a)

    Returns
    -------
    return: float
        calculated similarity 
    """
    target_fp = res[i, :len(fp_list[0])]
    return 1-np.mean(abs(np.array(fp_list)-target_fp), axis=1)


def total_similarity(res, fp_list):
    """
    calculate logarithmic similarity of a) specific fingerprints and b) list of fingerprints

    Parameters
    ----------
    res : np array of float
        float array, containing fingerprints information of a)
    fp_list: list of int
        list of fingerprints b), which should be compared with a)

    Returns
    -------
    return: float
        calculated similarity 
    """

    sim_list = []
    for i in range(res.shape[0]):
        sim = max(compare_similarity(res, i, fp_list))
        sim_list.append(sim)
    return np.array(sim_list)
