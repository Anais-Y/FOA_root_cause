import pandas as pd
import numpy as np
import os
from geopy import distance
from geopy.geocoders import Nominatim
import pyodbc
import datetime
import time
import re
import pretraiter as pt


def open_conn_FBI():
    conn = pyodbc.connect('Driver={SQL Server};'
                        'Server=svc-fbi-db-CFt.mgmt.cloud.ds.ge.com,3433;'
                        'Database=Dev_GPRS_Datamart;'
                        'UID=iCARE;PWD=abcde;'
                        'Trusted_Connection=no')
    return conn


def run_query_sqldb(query, conn):
    df_query = pd.read_sql_query(query, conn)
    return df_query


def get_coordinates(country,city):
    geolocator = Nominatim(user_agent='xin.yan@ge.com')
    if city=='nan':
        location = geolocator.geocode(f'{country}')
    else:
        location = geolocator.geocode(f'{city} {country}')
    if location!=None:
        return location.latitude, location.longitude
    else:
        print('echec', country,city)
        return None, None 
    

def WHSposition(org):
    org['CITY']=org['CITY'].astype(str)
    org['Latitude'], org['Longitude']=zip(*org.apply(lambda x: get_coordinates(x['COUNTRY'], x['CITY']),axis=1))
    return org


def get_position(part_consumed_EMEA,orgs):
    latitudes=[]
    longitudes=[]
    #part_consumed_EMEA.reset_index(inplace=True)
    for e in part_consumed_EMEA.index:
        whs=part_consumed_EMEA['Process_owner'][e]
        c=whs.split('>')
        c=[e for e in c if pd.isnull(e)==False and e!= ' ' and e!='']
        lat=[]
        long=[]
        for k in c:
            print(k, e)
            latitude=list(orgs[orgs.ORG==k]['Latitude'])[0]
            lat+=[latitude]
            long+=[list(orgs[orgs.ORG==k]['Longitude'])[0]]
        latitudes+=[lat]
        longitudes+=[long]
    adresses=[]
    for c in range(len(latitudes)):
        adress=[]
        for k in range(len(latitudes[c])):
            adress+=[[latitudes[c][k], longitudes[c][k]]]
        adresses+=[adress]
    return adresses


def distance_WHS(pos1, pos2):
    return distance.distance(pos1,pos2).km


def prepareData_FOA():
    #WHS_path='/Box/Doc_Xin/Sujet_Principale/Construction_DB/Data_collection/'
    renew = str(input('Wanna renew WHS list?(y/n):'))
    if renew=='y':
        # orgs=run_query_sqldb("""SELECT 
        # ORG_CODE [ORG], ORG_POLE, POLE_CODE, ORG_COUNTRY [COUNTRY], ORG_CITY [CITY] 
        # from [dbo].[GPRS_MASTER_ORGS]""", open_conn_FBI())
        orgs=pd.read_excel('listWHS.xlsx')
        orgs = WHSposition(orgs)
        #orgs.to_excel(os.path.join(WHS_path, 'liste_entrepot_avec_position.xlsx'))
        orgs.to_excel(os.path.join(os.getcwd(), 'liste_entrepot_avec_position.xlsx'))
    elif renew=='n':
        orgs = pd.read_excel('liste_entrepot_avec_position.xlsx')
    else:
        print('Follow the instruction pls!')
    
    file_path = f'./{key_group}/Tracing_FOA/rma_B_all_parts.xlsx'
    tracing_rma=pd.read_excel(file_path)
    tracing_rma=tracing_rma[['Key_value','Part Status','Customer Country','Customer POLE','Process_owner',
                             'Number of Warehouse', 'Number of returns','Number of Surplus','Number of Pole']]
    tracing_rma.set_index('Key_value',inplace=True) #Key_value-->RMA numbers
    data = tracing_rma
    data['drop_data'] = data['Process_owner'].str.contains('Unfound') | data['Process_owner'].str.contains('Unable') | data['Process_owner'].str.startswith('S')
    drop_index=data[data['drop_data']==True].index
    data.drop(index=drop_index, inplace=True)
    data.dropna(subset=['Process_owner'], inplace=True)
    return orgs, data


def prepareData_NF():
    #WHS_path='/Box/Doc_Xin/Sujet_Principale/Construction_DB/Data_collection/'
    renew = str(input('Wanna renew WHS list?(y/n):'))
    if renew=='y':
        # orgs=run_query_sqldb("""SELECT 
        # ORG_CODE [ORG], ORG_POLE, POLE_CODE, ORG_COUNTRY [COUNTRY], ORG_CITY [CITY] 
        # from [dbo].[GPRS_MASTER_ORGS]""", open_conn_FBI())
        orgs=pd.read_excel(os.path.join(os.getcwd(), 'listWHS.xlsx'))
        orgs = WHSposition(orgs)
        #orgs.to_excel(os.path.join(WHS_path, 'liste_entrepot_avec_position.xlsx'))
        orgs.to_excel(os.path.join(os.getcwd(), 'liste_entrepot_avec_position.xlsx'))
    elif renew=='n':
        orgs = pd.read_excel(os.path.join(os.getcwd(), 'liste_entrepot_avec_position.xlsx'))
    else:
        print('Follow the instruction pls!')
    
    file_path = os.path.join(f'./{key_group}/Tracing_PO_NF/repair_F_all_parts.xlsx')
    tracing_rma=pd.read_excel(file_path)
    tracing_rma=tracing_rma[['Key_value','Part Status','Customer Country','Customer POLE','Process_owner',
                             'Number of Warehouse', 'Number of returns','Number of Surplus','Number of Pole']]
    tracing_rma['Key_value']=tracing_rma['Key_value'].astype(str)
    tracing_rma['Key_value'] = tracing_rma['Key_value'].apply(lambda x: x[:-2] if x.endswith('.0') else x)
    tracing_rma.set_index('Key_value',inplace=True) #Key_value-->RMA numbers
    data = tracing_rma
    data['drop_data'] = data['Process_owner'].str.contains('Unfound') | data['Process_owner'].str.contains('Unable') | data['Process_owner'].str.startswith('S')
    drop_index=data[data['drop_data']==True].index
    data.drop(index=drop_index, inplace=True)
    data.dropna(subset=['Process_owner'], inplace=True)
    return orgs, data



def col_distances(adresses):
    distances=[]
    for p in adresses:
        distance_part=[]
        for k in range(len(p)-1):
            distance_part+=[distance_WHS(p[k],p[k+1])]#calculer chaque fois la distance entre 2 entrepôts et l'jouter dans distance_part
        distances+=[distance_part] # liste de distance, contient la liste de distance pour chaque key_value
    for i in range(len(distances)):
        for k in range(len(distances[i])):
            try:
                distances[i][k]=round(distances[i][k],0)
            except:
                distances[i][k]=0
            # rendre entiers
    return distances


def col_before_and_after(data, distances):
    # Use after using function col_distances
    before_consumption=[]
    after_consumption=[]
    i=0
    for k in data.index:
        i+=1
        c=np.array(data['Process_owner'][k].split('>'))  # split the string when we came across '>'
        FE=['S' in e for e in c] #S'il y a un 'S' dans l'entrepôt-> TRUE
        if True not in FE:
            consumption=len(FE) #Si tous les noms des entrepôt ne contient pas 'S', consumption est la longueur de FE
        else:
            consumption=max(np.where(FE)[0])+1
        before_consumption+=[distances[i-1][:consumption]] # Les distances passées avant debrief(consumption)
        after_consumption+=[distances[i-1][consumption:]]
    return before_consumption, after_consumption


def search_key(key, sheet_name, file_dir): # key, sheet_name:str
    all_files_list = os.listdir(file_dir)
    all_keys = []
    tracing_files = [i for i in all_files_list if os.path.isfile(os.path.join(file_dir,i)) and i.startswith('tracing')]
    for file in tracing_files:
        if sheet_name=='Summary':
            single_tracing = pd.read_excel(os.path.join(file_dir, file), sheet_name=sheet_name, index_col=0)
            single_tracing_key = single_tracing.loc[key, 'Values']
            all_keys.append(single_tracing_key)
        else:
            single_tracing = pd.read_excel(os.path.join(file_dir, file), sheet_name=sheet_name)
            single_key = []
            if len(single_tracing.index)!=0:
                key_list=single_tracing[key].dropna()
                single_key=key_list.values.tolist()
            all_keys.append(single_key)
    return all_keys


def cal_delta_days(critere, sheet_name, file_dir):
    all_files_list = os.listdir(file_dir)
    all_keys = []
    tracing_files = [i for i in all_files_list if os.path.isfile(os.path.join(file_dir,i)) and i.startswith('tracing')]
    for file in tracing_files:
        single_tracing = pd.read_excel(os.path.join(file_dir, file), sheet_name=sheet_name, index_col=0)
        step0 = single_tracing['Transaction Day'][single_tracing[critere]==0].tolist()
        step1 = single_tracing['Transaction Day'][single_tracing[critere]==-1].tolist()
        if len(step0)==0 or len(step1)==0:
            single_delta_days = 0
        else:
            date0 = step0[-1].date()
            date1 = step1[-1].date()
            single_delta_days = (date1-date0).days # calculer écart de jours entre FE consignment creation et PO receipt
        all_keys.append(single_delta_days)
    return all_keys


def system_age(file_dir):
    all_files_list = os.listdir(file_dir)
    tracing_files = [i for i in all_files_list if os.path.isfile(os.path.join(file_dir,i)) and i.startswith('tracing')]
    system_age = []
    for file in tracing_files:
        single_tracing = pd.read_excel(os.path.join(file_dir, file), sheet_name='txn history')
        sys_install = single_tracing[['System Install Date', 'Reason Name']]
        txn = single_tracing[['Transaction Day', 'Transaction Type']]
        # print(file)
        if sys_install['System Install Date'].isnull().all()==False:
            try:
                date1 = sys_install[sys_install['Reason Name']=='Debrief']['System Install Date'].values[0]
                date2 = txn[txn['Transaction Type']=='PO Receipt']['Transaction Day'].values[0]
            except:
                date1 = sys_install[sys_install['System Install Date'].isna()==False]['System Install Date'].values[0]
                date2 = txn[txn['Transaction Day'].isna()==False]['Transaction Day'].values[0]
            try:
                delta_days = (date2-date1).astype('timedelta64[D]')
                if (delta_days/ np.timedelta64(1, 'D'))>0:
                    age = delta_days/ np.timedelta64(1, 'D')
                else:
                    age = pd.NA
            except:
                age = pd.NA
        else:
            age = pd.NA
        system_age.append(age)
    return system_age


def part_birthday(file_dir):
    all_files_list = os.listdir(file_dir)
    tracing_files = [i for i in all_files_list if os.path.isfile(os.path.join(file_dir,i)) and i.startswith('tracing')]
    part_age = []
    for file in tracing_files:
        single_tracing = pd.read_excel(os.path.join(file_dir, file), sheet_name='txn history')
        txn = single_tracing[['Transaction Day', 'Transaction Type']]
        # print(file)
        if txn['Transaction Type'].isnull().all()==False:
            try:
                date = txn[txn['Transaction Type']=='PO Receipt']['Transaction Day'].values[-1]
            except:
                date = txn[txn['Transaction Day'].isna()==False]['Transaction Day'].values[-1]
        else:
            date = pd.NA
        part_age.append(date)
    return part_age


def find_WHS_changes(dates, warehouses):
    changes = []
    whs = []
    prev_whs = None

    for date, warehouse in zip(dates, warehouses):
        if warehouse != prev_whs:
            changes.append(date)
            whs.append(warehouse)
            prev_whs = warehouse

    whs.append(warehouses.iloc[-1])
    changes.append(dates.iloc[-1])
    
    return whs, changes


def sejour_WHS(file_dir):
    all_files_list = os.listdir(file_dir)
    tracing_files = [i for i in all_files_list if os.path.isfile(os.path.join(file_dir,i)) and i.startswith('tracing')]
    SejourEntrepot = []
    for file in tracing_files:
        single_tracing = pd.read_excel(os.path.join(file_dir, file), sheet_name='txn history')
        single_tracing = single_tracing[['Transaction Day', 'Inventory Org']]
        single_tracing.drop_duplicates(inplace=True)
        dates = single_tracing['Transaction Day']
        warehouses = single_tracing['Inventory Org']
        whs, changes = find_WHS_changes(dates, warehouses)
        sejour = pd.DataFrame()
        L = len(whs)-1
        sejour['whs']=whs[:L]
        jour = []
        for i in range(L):
            delta = (changes[i]-changes[i+1]).days
            jour.append(delta)

        sejour['jours']=jour
        df_dict = sejour.groupby('whs').sum().T
        SejourEntrepot.append(df_dict.to_dict('records'))
    return SejourEntrepot


def get_failures(file_dir):
    all_files_list = os.listdir(file_dir)
    tracing_files = [i for i in all_files_list if os.path.isfile(os.path.join(file_dir,i)) and i.startswith('tracing')]
    failures = []
    for file in tracing_files:
        txn_date_file = pd.read_excel(os.path.join(file_dir, file), sheet_name='txn history')
        repair_file = pd.read_excel(os.path.join(file_dir, file), sheet_name='repair history')
        txn_date = txn_date_file['Transaction Day'].iloc[0]
        if len(repair_file.index)!=0:
            failure_code = repair_file.loc[repair_file['PO1 Shipment Last Update Date']<txn_date, 'Failure Code']
            failure_code = failure_code.dropna()
            failure = failure_code.values.tolist()
        else:
            failure = []
        failures.append(failure)
    return failures



def failure_FOA(failure_type):
    failure_FOA=[]
    for list in failure_type:
        count = 0
        for word in list:
            if 'FOA' in str(word):
                count+=1
        failure_FOA.append(count)
    return failure_FOA


def analyse(col_name, df):
    for index, row in df.iterrows():
        if col_name=='séjourWHS':
            dict = eval(str(row[col_name]))
            whs_dict = dict[0]
            for key, value in whs_dict.items():
                if key in df.columns:
                    df.at[index, key] = value
                else:
                    df[key] = pd.Series([None] * len(df)) 
                    df.at[index, key] = value
        else:
            data = str(row[col_name])
            liste = re.findall(r"'(.*?)'", data)
            for key in liste:
                if key in df.columns:
                    value = df[key].iloc[index] 
                    df.at[index, key] = int(0 if value is None else value)+1
                else :
                    df[key] = pd.Series([None] * len(df))
                    df.at[index, key] = 1
    return df


def complé_system_age(df):
    elf_date = pd.to_datetime(df['elf_date'])
    asset_install_date = pd.to_datetime(df['asset_install_date'])
    systemAgeSS =(elf_date - asset_install_date)/np.timedelta64(1, 'D')
    systemAgeSS.apply(lambda x: pd.NA if x<0 else x)
    df['SystemAge'] = df['SystemAge'].fillna(systemAgeSS)

    return df


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


def FOA_main():
    orgs, data = prepareData_FOA()
    print('preparation done')
    adresses=get_position(data, orgs)
    distances = col_distances(adresses)
    before_consumption, after_consumption=col_before_and_after(data, distances)
    B_distances = [sum(e) for e in before_consumption]
    A_distances = [sum(e) for e in after_consumption]
    data['Distances']=distances
    tot_distances = [sum(e) for e in data.Distances]
    print('distances calculated')
    file_dir = f'./{key_group}/Tracing_FOA/'

    print('séjour dans WHS')
    SejourEntrepot = sejour_WHS(file_dir)

    print('tracing_id..')
    tracing_id = search_key('Key_value', 'Summary', file_dir)
    birthday = part_birthday(file_dir)
    Repaire_qty = search_key('Repair QTY', 'Summary', file_dir)
    print('collecting repair center')
    Repair_center = search_key('Vendor Name', 'repair history',file_dir)
    print('collecting repair center site code..')
    Repair_site_code = search_key('Vendor Site Code', 'repair history',file_dir)
    print('collecting failure types')
    failure_type = get_failures(file_dir)
    print('collecting nb FOA')
    nb_FOA = failure_FOA(failure_type)
    print('number of repair calculated')
    print('collecting FE SSO..')
    FE_SSO = search_key('FE SSO', 'FE Returns summary',file_dir)
    print('System ID...')
    SystemID = search_key('System ID','FE Returns summary',file_dir)
    print('Carriers....')
    Carrier = search_key('Freight Carrier', 'FE Returns summary',file_dir)
    print('System age..')
    Age = system_age(file_dir)
    print('system age calculated')
    delta_days = cal_delta_days('step','txn history',file_dir)

    adj_Serial_Number=search_key('Serial Number', 'Summary', file_dir)
    data['B_distances']=B_distances
    data['A_distances']=A_distances
    data['tot_distances']=tot_distances
    d = {'tracing_id': tracing_id, 'part_birth': birthday, 'adj_Serial_Number':adj_Serial_Number, 'delta_days': delta_days, 'FE_SSO': FE_SSO, 
         'Repair_center': Repair_center, 'Repair_site_code': Repair_site_code, 'Repaire_qty':Repaire_qty,'failure_type':failure_type, 'nb_FOA':nb_FOA,
         'SystemID': SystemID, 'SystemAge': Age, 'Carrier':Carrier, 'séjourWHS':SejourEntrepot}
    df_new = pd.DataFrame(data=d)
    df_new.set_index('tracing_id', inplace=True)
    df_new.index=df_new.index.astype(str)
    data.index=data.index.astype(str)
    data = data.merge(df_new, how='left', left_index=True, right_index=True)
    data['FE visited'] = data['FE_SSO'].apply(lambda x: len(x) if isinstance(x, list) else None)
    data_SS = pd.read_excel(f'./{key_group}/SS_4_classes.xlsx')
    data.reset_index(inplace=True)
    print(data.columns)
    data_SS['rma_no']=data_SS['rma_no'].astype(str)
    data['Key_value'] = data['Key_value'].astype(str)
    # data.to_excel('data.xlsx')
    Merge = pd.merge(data, data_SS, how='left', left_on='Key_value', right_on='rma_no')
    Merge.to_excel(f'./{key_group}/{datetime.date.today()}_merge_v4.xlsx')
    # print('Now we have:', Merge.columns)
    # print('Ajouter séjours de chaque WHS..')
    # FInal1 = analyse('séjourWHS',Merge)
    # print('rendre Carrier analysable..')
    # Final2 = analyse('Carrier', FInal1)
    # print('Repair center compte...')
    Final1 = analyse('Repair_center',Merge)
    Final1 = complé_system_age(Final1)
    print('Repair center site count...')
    Final2 = analyse('Repair_site_code', Final1)
    # print('Carrier...')
    # Final2 = analyse('Carrier', Final2) 
    # # en commentaire si without carrier
    # Final2.set_index(inplace=True)
    Final3 = FE_exp(Final2)
    Final3.to_excel(f'./{key_group}/{datetime.date.today()}-Final-FOA.xlsx')
    print('Work Done, Good Luck')
    return 'done'


def notFOA_main():
    orgs, data = prepareData_NF()
    print('preparation done')
    adresses=get_position(data, orgs)
    distances = col_distances(adresses)
    before_consumption, after_consumption=col_before_and_after(data, distances)
    B_distances = [sum(e) for e in before_consumption]
    A_distances = [sum(e) for e in after_consumption]
    data['Distances']=distances
    tot_distances = [sum(e) for e in data.Distances]
    print('distances calculated')
    file_dir = f'./{key_group}/Tracing_PO_NF/'

    print('séjour dans WHS')
    SejourEntrepot = sejour_WHS(file_dir)

    print('tracing_id..')
    tracing_id = search_key('Key_value', 'Summary', file_dir)
    birthday = part_birthday(file_dir)
    Repaire_qty = search_key('Repair QTY', 'Summary', file_dir)
    print('collecting repair center')
    Repair_center = search_key('Vendor Name', 'repair history',file_dir)
    print('collecting repair center site code..')
    Repair_site_code = search_key('Vendor Site Code', 'repair history',file_dir)
    print('collecting failure types')
    failure_type = get_failures(file_dir)
    print('collecting nb FOA')
    nb_FOA = failure_FOA(failure_type)
    print('number of repair calculated')
    print('collecting FE SSO..')
    FE_SSO = search_key('FE SSO', 'FE Returns summary',file_dir)
    print('System ID...')
    SystemID = search_key('System ID','FE Returns summary',file_dir)
    print('Carriers....')
    Carrier = search_key('Freight Carrier', 'FE Returns summary',file_dir)
    print('System age..')
    Age = system_age(file_dir)
    print('system age calculated')
    delta_days = cal_delta_days('step','txn history',file_dir)

    adj_Serial_Number=search_key('Serial Number', 'Summary', file_dir)
    data['B_distances']=B_distances
    data['A_distances']=A_distances
    data['tot_distances']=tot_distances
    d = {'tracing_id': tracing_id, 'part_birth': birthday, 'adj_Serial_Number':adj_Serial_Number, 'delta_days': delta_days, 'FE_SSO': FE_SSO, 
         'Repair_center': Repair_center, 'Repair_site_code': Repair_site_code, 'Repaire_qty':Repaire_qty,'failure_type':failure_type, 'nb_FOA':nb_FOA,
         'SystemID': SystemID, 'SystemAge': Age, 'Carrier':Carrier, 'séjourWHS':SejourEntrepot}
    df_new = pd.DataFrame(data=d)
    df_new.set_index('tracing_id', inplace=True)
    df_new.index=df_new.index.astype(str)
    data.index=data.index.astype(str)
    # df_new.index['FE_SSO']=df_new.index['FE_SSO'].apply(lambda x: x[:-2] if x.endswith('.0') else x)

    # df_new.to_excel('df_new.xlsx')
    data = data.merge(df_new, how='left', left_index=True, right_index=True)
    data['FE visited'] = data['FE_SSO'].apply(lambda x: len(x) if isinstance(x, list) else None)
    data_SS = pd.read_excel(f'./{key_group}/SS_4_classes.xlsx')
    data.reset_index(inplace=True)
    # print(data.columns)
    data_SS['rma_no']=data_SS['rma_no'].astype(str)

    # Ajouter PO dans donées de service suite
    rma_po = pd.read_excel(f'./{key_group}/rma_po.xlsx')
    rma_po['Forward_rma']=rma_po['Forward_rma'].astype(str)
    Merge = pd.merge(rma_po, data_SS, how='left', left_on='Forward_rma', right_on='rma_no')

    Merge['PO'] = Merge['PO'].astype(str)
    Merge['PO'] = Merge['PO'].apply(lambda x: x[:-2] if x.endswith('.0') else x)
    data['Key_value'] = data['Key_value'].astype(str)
    Merge2 = pd.merge(data, Merge, how='left', left_on='Key_value', right_on='PO')
    Merge2.to_excel(f'./{key_group}/{datetime.date.today()}_merge.xlsx')
    # print('Now we have:', Merge.columns)
    # print('Ajouter séjours de chaque WHS..')
    # FInal1 = analyse('séjourWHS',Merge)
    # print('rendre Carrier analysable..')
    # Final2 = analyse('Carrier', FInal1)
    print('Repair center compte...')
    Final1 = analyse('Repair_center',Merge2)
    Final1 = complé_system_age(Final1)
    print('Repair center site count...')
    Final2 = analyse('Repair_site_code', Final1)
    # print('Carrier...')
    # Final2 = analyse('Carrier', Final2)
    # Final2.set_index('Key_value', inplace=True)
    Final3 = FE_exp(Final2)
    Final3.to_excel(f'./{key_group}/{datetime.date.today()}-Final-NotFOA.xlsx')
    print('Work Done, Good Luck')
    return 'done'



def WHS_transformer(file_name):
    raw_data = pd.read_excel(file_name)
    raw_data.dropna(subset=['adjusted_earlylife_failure'], inplace=True)
    raw_data.dropna(subset=['séjourWHS'], inplace=True)
    raw_data['drop_data'] = raw_data['Part Status'].str.contains('Not Consumed')
    drop_index=raw_data[raw_data['drop_data']==True].index
    raw_data.drop(index=drop_index, inplace=True)
    for index, row in raw_data.iterrows():
        dict = eval(str(row['séjourWHS']))
        whs_dict = dict[0]
        raw_data.at[index, 'max_whs']=max(whs_dict, key=whs_dict.get)
        raw_data.at[index, 'max_whs_jours']=max(whs_dict.values())

        raw_data.at[index, 'min_whs']=min(whs_dict, key=whs_dict.get)
        raw_data.at[index, 'min_whs_jours']=min(whs_dict.values())

        raw_data.at[index, 'median_whs']=sorted(whs_dict, key=lambda x: x[1])[int(len(whs_dict)/2)]
        raw_data.at[index, 'median_whs_jours']=sorted(whs_dict.values())[int(len(whs_dict)/2)]

    return raw_data



def traitementNF():
    """
    Transformer la distionnaire de séjour entrepôt à six colonne(max, min, median)
    Drop les colonnes non-utiles
    """
    nFOA_filename = f'./{key_group}/{datetime.date.today()}-Final-NotFOA.xlsx'
    raw_dataNF = WHS_transformer(nFOA_filename)
    cols_drop = ['Part Status','Process_owner','drop_data','Distances','tot_distances', 'A_distances', 'part_birth',
                'adj_Serial_Number','FE_SSO', 'Repair_center','Repair_site_code', 'failure_type','SystemID','Carrier', 'séjourWHS',
                'elf_date',  'Activity_Quarter_Close_Date' ,'rma_no', 
                'failed_part_serial_number','failed_part_no', 
                'asset_system_id', 'PO','Forward_rma', 'delta_days', 
                'asset_system_id_location','sub_region_description','asset_install_date']
    for i in raw_dataNF.columns.tolist():
        if 'Unnamed' in str(i):
            cols_drop.append(i)
        if 'Key_value' in str(i):
            cols_drop.append(i)
    raw_dataNF.drop(columns=cols_drop, inplace=True)
    return raw_dataNF


def traitementF():
    FOA_filename = f'./{key_group}/{datetime.date.today()}-Final-FOA.xlsx'
    raw_dataF = WHS_transformer(FOA_filename)
    cols_drop = ['Part Status','Process_owner','drop_data','Distances','tot_distances', 'A_distances', 'part_birth',
                'adj_Serial_Number','FE_SSO', 'Repair_center','Repair_site_code', 'failure_type','SystemID','Carrier', 'séjourWHS',
                'elf_date', 'Activity_Quarter_Close_Date' ,'rma_no', 
                'failed_part_serial_number', 'failed_part_no', 
                'asset_system_id', 'delta_days',
                'asset_system_id_location','sub_region_description','asset_install_date']
    for i in raw_dataF.columns.tolist():
        if 'Unnamed' in str(i):
            cols_drop.append(i)
        if 'Key_value' in str(i):
            cols_drop.append(i)
    raw_dataF.drop(columns=cols_drop, inplace=True)
    return raw_dataF



if __name__=='__main__':
    # print(pathlib.Path().resolve())
    # print(os.getcwd())
    os.chdir('c:/Users/223102584/Box/Doc_Xin/Sujet_Principale/')
    key_group = 'Probe'
    begin=time.time()
    FOA_main()
    end=time.time()
    print('it takes',(end-begin)/60, 'mins to collect FOA data')

    begin=time.time()
    notFOA_main()
    end=time.time()
    print('it takes',(end-begin)/60, 'mins to collect not FOA data')

    raw_dataNF = traitementNF()
    raw_dataF = traitementF()
    data_final = pd.concat([raw_dataNF,raw_dataF])

    cols_tolabel = ['adjusted_earlylife_failure','failed_part_item_type','region_description','family_name', 'product_group',
                    'failed_part_description', 'product_identifier', 
                    'goldseal_flag','Customer Country','Customer POLE','Activity_Month_Close_Date','min_whs','max_whs','median_whs']
    data_labeled = pt.label_encoder(data_final, cols_tolabel, key_group)

    data_labeled['SystemAge'].fillna(data_labeled['SystemAge'].mean(),inplace = True)
    data_labeled['FE_exp'].fillna(data_labeled['FE_exp'].mean(),inplace = True)
    data_labeled['Customer Country'].fillna(data_labeled['Customer Country'].mean(),inplace = True)
    data_labeled['Customer POLE'].fillna(data_labeled['Customer POLE'].mean(),inplace = True)

    data_labeled.fillna(0, inplace = True)
    data_labeled.to_excel(f'./{key_group}/Avant_cluster{datetime.date.today()}_NC.xlsx')