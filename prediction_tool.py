from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
import os
import datetime
from Analyse import data_sampling
import Modules_Tracing as tracing
import Data_collection_nbINOUT as collecter


# -----------------------------Read trainning data----------------------------- #
def prepareData(file_name):
    '''
    Input: nom du fichier sorti de Data_collection
    Output: data sans labels et labels
    '''
    data_labeled = pd.read_excel(file_name)
    data_labeled = data_labeled[(data_labeled['Repaire_qty']>0) & (data_labeled['Repaire_qty']<10)]
    dropIndex = data_labeled[(data_labeled['Repaire_qty']==0) | (data_labeled['SystemAge']<0)].index
    data_labeled.drop(index=dropIndex, inplace = True)
    cols_drop = ['adjusted_earlylife_failure','elf_days']
    # cols_drop = ['adjusted_earlylife_failure']
    for i in data_labeled.columns.tolist():
        if 'Unnamed' in str(i):
            cols_drop.append(i)
    data = data_labeled.drop(columns = cols_drop)
    data = data.loc[:, data.any()] # enlever les colonnes tous 0
    labels = data_labeled['adjusted_earlylife_failure']
    # FOA_label = int(input('Tell us FOA label:'))
    FOA_label = 1
    nb_FOA = len(data_labeled[data_labeled['adjusted_earlylife_failure']==FOA_label].index)
    print('Training data has', len(data_labeled.index), 'parts,', nb_FOA, 'FOAs among them.')
    return data, labels, data_labeled, FOA_label


def prepareData_prime(file_name_prime):
    '''
    Input: nom du fichier sorti de Data_collection
    Output: data sans labels et labels
    '''
    data_labeled = pd.read_excel(file_name_prime)
    cols_drop = ['adjusted_earlylife_failure']
    # cols_drop = ['adjusted_earlylife_failure']
    for i in data_labeled.columns.tolist():
        if 'Unnamed' in str(i):
            cols_drop.append(i)
    data = data_labeled.drop(columns = cols_drop)
    labels = data_labeled['adjusted_earlylife_failure']
    # FOA_label = int(input('Tell us FOA label:'))
    FOA_label = 1
    nb_FOA = len(data_labeled[data_labeled['adjusted_earlylife_failure']==FOA_label].index)
    print('We have ', len(data_labeled.index), 'parts', nb_FOA, ' FOAs among them.')
    return data, labels, data_labeled, FOA_label



# -----------------------------Sampling training data----------------------------- #
def data_sampling(data_labeled):
    sample_ELF = len(data_labeled[data_labeled['adjusted_earlylife_failure']==0].index)
    sample_FOA = len(data_labeled[data_labeled['adjusted_earlylife_failure']==1].index)
    sample_IPFR = len(data_labeled[data_labeled['adjusted_earlylife_failure']==2].index)
    nb_sample = np.min([sample_ELF, sample_FOA, sample_IPFR])

    k1=data_labeled[data_labeled['adjusted_earlylife_failure']==0].sample(nb_sample, random_state=24)
    k2=data_labeled[data_labeled['adjusted_earlylife_failure']==1].sample(nb_sample, random_state=24)
    k3=data_labeled[data_labeled['adjusted_earlylife_failure']==2].sample(nb_sample, random_state=243)
    data_labeled_sample = pd.concat([k1,k2,k3])
    labels = data_labeled_sample['adjusted_earlylife_failure']
    try:
        data_sample = data_labeled_sample.drop(columns = ['Unnamed: 0', 'adjusted_earlylife_failure','elf_days'])
    except:
        data_sample = data_labeled_sample.drop(columns = ['Unnamed: 0', 'adjusted_earlylife_failure'])
    return data_sample, labels


def data_sampling2(data_labeled):
    sample_ELF = len(data_labeled[data_labeled['adjusted_earlylife_failure']==0].index)
    sample_FOA = len(data_labeled[data_labeled['adjusted_earlylife_failure']==1].index)
    nb_sample = np.min([sample_ELF, sample_FOA])

    k1=data_labeled[data_labeled['adjusted_earlylife_failure']==0].sample(nb_sample, random_state=42)
    k2=data_labeled[data_labeled['adjusted_earlylife_failure']==1].sample(nb_sample, random_state=42)
    data_labeled_sample = pd.concat([k1,k2])
    labels = data_labeled_sample['adjusted_earlylife_failure']
    try:
        data_sample = data_labeled_sample.drop(columns = ['Unnamed: 0', 'adjusted_earlylife_failure','elf_days'])
    except:
        data_sample = data_labeled_sample.drop(columns = ['Unnamed: 0', 'adjusted_earlylife_failure'])
    return data_sample, labels


# -----------------------------Create and train the model----------------------------- #
def randomForest(labels, data):
    clf = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(data, labels,random_state=88)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    # X_test.to_excel('xtest.xlsx')
    # print(y_test, clf.predict(X_test))
    print('        Accuracy of prediction: ','{:.2f}'.format(accuracy*100), '%')
    
    return clf

# -----------------------------Generate data for predict----------------------------- #

def save_file(datas,value,filep):
    with pd.ExcelWriter(filep+'/tracing_{}.xlsx'.format(value)) as writer:
        for e in datas.keys():
            try:
                datas[e].to_excel(writer,sheet_name=e, index=False)
            except:# s'il marche pas, transformer à UTF-8 et réessayer
                datas[e]=datas[e].applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
                #applymap: apply a function to dataframe elementwise
                datas[e].to_excel(writer,sheet_name=e, index=False)
    print('saved !!')


def create_folder():
    path = './Scripts/log/'
    folder_name = path+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name


def find_PO(folder_name):
    repair_po = input('Please register your repair po: ')
    historydata= tracing.tracing('repair', repair_po, 'F')
    save_file(historydata,str(repair_po),folder_name)
    return repair_po


def whs_transformer(data):
    dict = data['séjourWHS'][0][0]
    data['max_whs']=max(dict, key=dict.get)
    data['max_whs_jours']=max(dict.values())

    data['min_whs']=min(dict, key=dict.get)
    data['min_whs_jours']=min(dict.values())

    data['median_whs']=sorted(dict, key=lambda x: x[1])[int(len(dict)/2)]
    data['median_whs_jours']=sorted(dict.values())[int(len(dict)/2)]

    return data


def FE_exp(data):
    df_SSO = pd.read_excel('./FE_data/GE HealthCare Employees 2023-05-27.xlsx', sheet_name='Active')
    df_date_embauche = df_SSO[['FE_SSO', 'Exp']]
    # df_date_embauche.set_index('Employee ID',inplace=True)
    df_FE = data[['FE_SSO']]
    df_FE['FE_SSO']=df_FE['FE_SSO'].astype(str)
    df_FE['FE_SSO']=df_FE['FE_SSO'].apply(lambda x: x.strip('][').split(', '))
    df_FE = df_FE.explode('FE_SSO')
    df_FE['FE_SSO']=df_FE['FE_SSO'].apply(lambda x: x[:-2] if x.endswith('.0') else x)
    df_date_embauche['FE_SSO']=df_date_embauche['FE_SSO'].astype(str)
    df_merged_exp = df_FE.merge(df_date_embauche, on='FE_SSO', how = 'left')
    df_merged_exp.set_index(df_FE.index,inplace = True)
    df_merged_exp.dropna(subset= ['Exp'],inplace=True)
    df_average_exp = df_merged_exp.groupby(level=0)['Exp'].mean()
    data['FE_exp']=df_average_exp
    return data


def encoder_ref(name, col, data):
    ref = pd.read_excel(f'./Label_Encoders/label_encoders_{name}.xlsx')
    for colonne in col:
        tofind = data.loc[0,colonne]
        i=0
        for elem in ref[colonne].tolist():
            # print(elem, tofind)
            if elem == tofind:
                found = i
                data[colonne]=found
                break
            i=i+1
    
    return data


def generate_file(orgs, summary, folder_name):
    df_sum = summary.T
    df_sum.columns=df_sum.iloc[0]
    df_sum=df_sum.drop(df_sum.index[0])
    df_sum.set_index('Key_value',inplace=True)
    data = df_sum[['Part Status','Customer Country','Customer POLE','Process_owner', 
               'Number of Warehouse', 'Number of returns', 'Number of Surplus', 'Number of Pole']]
    adresses = collecter.get_position(data, orgs)
    distances = collecter.col_distances(adresses)
    before_consumption, after_consumption, INOUT=collecter.col_before_and_after(data, distances)
    B_distances = [sum(e) for e in before_consumption]
    data['B_distances']=B_distances
    data['séjourWHS'] = collecter.sejour_WHS(folder_name)
    data['nb_Inout']=INOUT
    data['Repair_center'] = collecter.search_key('Vendor Name', 'repair history',folder_name)
    data['Repair_site_code'] = collecter.search_key('Vendor Site Code', 'repair history',folder_name)
    data['Repaire_qty'] = collecter.search_key('Repair QTY', 'Summary', folder_name)
    failure_type = collecter.get_failures(folder_name)
    data['failure_type'] = failure_type
    data['nb_FOA'] = collecter.failure_FOA(failure_type)
    data['FE_SSO'] = collecter.search_key('FE SSO', 'FE Returns summary',folder_name)
    data['SystemID'] = collecter.search_key('System ID','FE Returns summary',folder_name)
    data['Carrier'] = collecter.search_key('Freight Carrier', 'FE Returns summary',folder_name)
    data['SystemAge'] = collecter.system_age(folder_name)
    data['FE visited'] = data['FE_SSO'].apply(lambda x: len(x) if isinstance(x, list) else None)
    data.reset_index(inplace=True)
    data2=collecter.analyse('Repair_center',data)
    data3 = collecter.analyse('Repair_site_code', data2)
    data4=collecter.analyse('Carrier', data3)
    data5 = whs_transformer(data4)
    data6=FE_exp(data5)
    cols_drop = ['Part Status','Process_owner','séjourWHS','Repair_center','Repair_site_code',
                'failure_type',  'FE_SSO', 'SystemID', 'Carrier']
    for i in data6.columns.tolist():
        if 'Unnamed' in str(i):
            cols_drop.append(i)
        if 'Key_value' in str(i):
            cols_drop.append(i)
        if 'Columns' in str(i):
            cols_drop.append(i)
    data6.drop(columns=cols_drop, inplace=True)
    cols = ['failed_part_description','failed_part_item_type','product_group','product_identifier',
        'family_name','Activity_Month_Close_Date','region_description',	'goldseal_flag']
    for col in cols:
        data6.loc[0,col]=int(input(f'{col}:'))

    data7 = encoder_ref('Probe', ['Customer Country', 'Customer POLE','max_whs', 'min_whs', 'median_whs'], data6)
    template = pd.read_excel('./Scripts/template.xlsx')
    final = pd.concat([template, data7])
    final.fillna(0,inplace=True)
    final.drop(columns=['Unnamed: 0'],inplace=True)
    to_predict='./Scripts/kjkj.xlsx'
    final.to_excel(to_predict)

    return to_predict




# -----------------------------Predict----------------------------- #
def prediction(file_train, folder_name):
    data, labels, data_labeled, FOA_label = prepareData(file_train)
    data, labels = data_sampling(data_labeled)
    data_labeled2 = data_labeled
    data_labeled2.loc[data_labeled2['adjusted_earlylife_failure']==2, 'adjusted_earlylife_failure']=0
    data2,labels2 = data_sampling2(data_labeled2)
    print('If seperate them as 3 classes(FOA, ELF, IPFR>180):')
    clf = randomForest(labels, data)
    print('If seperate them as 2 classes(FOA and Not_FOA):')
    randomForest(labels2, data2)
    _,l = data.shape
    print(f'Training data has {l} features.')
    print('-'*10,'End of Training', '-'*10)
    orgs = pd.read_excel(os.path.join('./liste_entrepot_avec_position.xlsx'))
    repair_po = find_PO(folder_name)
    summary = pd.read_excel(f'{folder_name}/tracing_{str(repair_po)}.xlsx', sheet_name='Summary')
    predict_filename = generate_file(orgs, summary, folder_name)
    data_test = pd.read_excel(predict_filename)
    data_test.drop(columns='Unnamed: 0', inplace=True)
    # print(data_test)
    data_test = data_test.loc[0].values
    data_test = np.reshape(data_test, (-1,l))
    # print(data_test.shape)
    pred = clf.predict_proba(data_test)

    FOA_prob = float(pred[:, 1])*100
    ELF_prob = float(pred[:, 0])*100
    IPFR_prob = 100-FOA_prob-ELF_prob
    print(f'FOA probability :{FOA_prob}%, ELF probability :{ELF_prob}%, IPFR>180 probability:{IPFR_prob}%.')


def predict_prime(file_train):
    data, labels, data_labeled, FOA_label = prepareData_prime(file_train)
    data, labels = data_sampling(data_labeled)
    clf = randomForest(labels, data)
    _,l = data.shape
    print(f'Your data has {l} features.')
    features = data.columns
    test_prime = generate_prime(features)
    test_prime = test_prime.loc[0].values
    test_prime = np.reshape(test_prime, (-1,l))
    # print(data_test.shape)
    pred = clf.predict_proba(test_prime)
    FOA_prob = float(pred[:, FOA_label])
    ELF_prob = float(pred[:, 0])*100
    IPFR_prob = 100-FOA_prob-ELF_prob
    print(f'FOA probability :{FOA_prob}%, ELF probability :{ELF_prob}%, IPFR>180 probability:{IPFR_prob}%.')


def generate_prime(features):
    test_prime = pd.DataFrame()
    for col in features:
        test_prime.loc[0,col]=float(input(f'{col}:'))
    test_prime.to_excel('test_prime.xlsx')
    return test_prime

# -----------------------------For prime part----------------------------- #



if __name__=='__main__':
    warnings.filterwarnings("ignore")
    os.chdir('c:/Users/223102584/Box/Doc_Xin/Sujet_Principale/')
    folder_name = create_folder()
    repaired = str(input('Do you have a repair PO(y/n)? '))
    print('-'*40)
    if repaired =='y':
        file_train = './Probe/Avant_cluster2023-07-12_ALL.xlsx'
        prediction(file_train, folder_name)
    else:
        file_train_prime = './Analyse_Prime_probe/Final.xlsx'
        predict_prime(file_train_prime)