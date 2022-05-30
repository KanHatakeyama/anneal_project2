"""
utility functions to process composite database

"""
import numpy as np
import copy
import pandas as pd


def check_anion(smiles_list):
    count = 0
    for i in smiles_list:
        if i.find("-") > 0:
            count += 1

    return count

def sort_smiles(current_record):
    """
    sort smiles: anions come first. higher weight ratio components firster
    """
    wt_list = np.array(current_record["SMILES_wt_list"])
    sm_list = current_record["smiles_list"]

    anion_check = np.array([int(i.find("-") > 0)*10 for i in sm_list])
    n_wt_list = wt_list+anion_check
    sort_index = [np.argsort(-n_wt_list)]

    current_record["smiles_list"] = list(np.array(sm_list)[sort_index])
    current_record["SMILES_wt_list"] = wt_list[sort_index]


def simplify_composite_dict(composite_dict):
    """
    delete unused parameters for machine learning
    """
    composite_dict = copy.deepcopy(composite_dict)

    for k in list(composite_dict.keys()):
        current_record = composite_dict[k]

        # an electrolyte must have only one anion
        if check_anion(current_record["smiles_list"]) != 1:
            composite_dict.pop(k)
            continue

        # delete unnecessary props
        for name in (["ID", "composition", 'inorg_name', 'inorg_contain_ratio(wt)', 'Temperature',
                      # 'SMILES_wt_list',
                      "wt_ratio",
                      'structureList', 'MWList', 'fp_list']):
            current_record.pop(name)

        compound_num = len(current_record["SMILES_wt_list"])
        for i in range(3-compound_num):
            current_record["SMILES_wt_list"].append(0)

        # sort
        sort_smiles(current_record)

        # use only litfsi
        # if current_record["smiles_list"][0] !='O=S([N-]S(=O)(C(F)(F)F)=O)(C(F)(F)F)=O':
        # if current_record["smiles_list"][0].find("F")<0:
        #    composite_dict.pop(k)
    return composite_dict


def composite_dict_to_df(composite_dict, compound_database):
    """
    convert composite dict info to dataframe
    """
    db_list = []
    cond_list = []
    smiles_list = []
    for current_record in composite_dict.values():
        # prepare fp array
        for num, smiles in enumerate(current_record["smiles_list"]):
            current_fp_array = np.array(compound_database.fp_dict[smiles])

            if num == 0:
                fp_array = current_fp_array
            else:
                fp_array = np.hstack((fp_array, current_fp_array))

        cond_list.append(np.log10(current_record["Conductivity"]))
        smiles_list.append(current_record["smiles_list"])
        db_list.append(fp_array)

    df = pd.DataFrame(db_list)
    df.columns = ["FP"+str(i) for i in range(df.shape[1])]
    df["Conductivity"] = cond_list
    return df
