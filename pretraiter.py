import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import datetime
import numpy as np
    

def FE_exp(data):
    df_SSO = pd.read_excel('../FE_data/GE HealthCare Employees 2023-05-27.xlsx', sheet_name='Active')
    df_date_embauche = df_SSO[['FE_SSO', 'Exp']]
    # df_date_embauche.set_index('Employee ID',inplace=True)
    df_FE = data[['FE_SSO']]
    df_FE['FE_SSO']=df_FE['FE_SSO'].apply(lambda x: x.strip('][').split(', '))
    df_FE = df_FE.explode('FE_SSO')
    df_FE['FE_SSO']=df_FE['FE_SSO'].apply(lambda x: x[:-2] if x.endswith('.0') else x)
    df_date_embauche['FE_SSO']=df_date_embauche['FE_SSO'].astype(str)
    df_merged_exp = df_FE.explode('FE_SSO').merge(df_date_embauche, on='FE_SSO')
    df_average_exp = df_merged_exp.groupby(level=0)['Exp'].mean()
    data['FE_exp']=df_average_exp
    return data



def label_encoder(data, cols, name):
    Encoders = []
    for single_col in cols:
        data_col = data[single_col].tolist()
        l_encoder = LabelEncoder()
        encoded_data = l_encoder.fit_transform(data_col)
        data[single_col]=encoded_data
        print(f'label_encoder of {single_col}:', l_encoder.inverse_transform(np.arange(encoded_data.max()+1)))
        Encoders.append(l_encoder.inverse_transform(np.arange(encoded_data.max()+1)))
    df_encoders = pd.DataFrame(Encoders)
    df_encoders = df_encoders.T
    df_encoders.columns = cols
    df_encoders.to_excel(f'./Label_Encoders/label_encoders_{name}.xlsx')
    return data


def transformer_date(data, col):
    for index in range(len(data)):
        day = str(data.iloc[index][col])
        try:
            try:
                date_object = datetime.datetime.strptime(day, "%m/%d/%Y")
            except:
                date_object = datetime.datetime.strptime(day, '%Y-%m-%d %H:%M:%S')
            data.at[index, col] = date_object.date()
        except:
            date_object = datetime.datetime.strptime(day, "%Y-%m-%d")
            data.at[index, col] = date_object.date()

    return data


def normalisation_Z(data, critere_col):
    for col in data.columns.tolist():
        if col!=critere_col:
            mean = data[col].mean()
            std = data[col].std()
            data[col]=(data[col]-mean)/std

    return data


def normalisation_moyenne(data, critere_col):
    for col in data.columns.tolist():
        if col!=critere_col:
            print(col)
            mean = data[col].mean()
            max = data[col].max()
            min = data[col].min()
            data[col]=(data[col]-mean)/(max-min)

    return data


if __name__=='__main__':
    os.chdir('c:/Users/223102584/Box/Doc_Xin/Sujet_Principale/Construction_DB/Adj_data_collection/')
    data = pd.read_excel('Final_FOA.xlsx')
    # data = FE_exp(data)
    cols_tolabel = ['adjusted_earlylife_failure','failed_part_classification','region_description','modality','family_name',
                    'goldseal_flag','Customer Country','Customer POLE','Activity_Month_Close_Date','min_whs','max_whs','median_whs']
    data = label_encoder(data, cols_tolabel,name='name')

    data = transformer_date(data, 'elf_date')
    data=transformer_date(data, 'part_birth')
    data['part_age']=data['elf_date']-data['part_birth']
    data['part_age'] = data['part_age'].apply(lambda x: x.days)
    cols_drop = ['Part Status','Process_owner','drop_data','Distances','tot_distances', 'A_distances', 'part_birth',
                'adj_Serial_Number','FE_SSO', 'Repair_center','Repair_site_code', 'failure_type','SystemID','Carrier', 'séjourWHS',
                'elf_date','failed_part_install_date','replacement_part_no', 'replacement_part_description',
                'replacement_part_classification', 'replacement_part_item_type', 'replacement_part_relationship_type', 
                'replacement_part_setup_date','key_group_desc', 'source_name','Activity_Quarter_Close_Date' ,'rma_no', 
                'failed_part_serial_number', 'failed_part_install_date','replacement_part_serial_number','failed_part_no', 
                'asset_system_id', 'modality',
                'asset_system_id_location','sub_region_description','asset_install_date']
    for i in data.columns.tolist():
        if 'Unnamed' in str(i):
            cols_drop.append(i)
        if 'index' in str(i):
            cols_drop.append(i)

    data.drop(columns=cols_drop, inplace=True)
    data.fillna(0,inplace=True)
    data = normalisation_moyenne(data,'adjusted_earlylife_failure')
    data.to_excel(f'Pretraité{datetime.date.today()}_FOA.xlsx')