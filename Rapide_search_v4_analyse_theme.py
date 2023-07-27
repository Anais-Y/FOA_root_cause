import pyodbc
import pandas as pd 
import os
import time
from tqdm.auto import tqdm
import datetime
import tkinter as tk
from tkinter import filedialog
# import signal
import warnings
import numpy as np
try:
    import cx_Oracle
except:
    import subprocess
    subprocess.call(['setup.bat'])
    import cx_Oracle

import timeout_decorator
from threading import Thread
import functools


# try:
#     cx_Oracle.init_oracle_client(lib_dir=r"C:\oracle\instantclient_19_9")
# except:
#     cx_Oracle.init_oracle_client(lib_dir=r"C:\oracle\instantclient_19_15")

warnings.filterwarnings("ignore")

# Various options vs FBI views to screen
#DB_TXN_VIEW = "vwAllTxns_2008_to_Present"
#DB_TXN_VIEW = "vwAllTxns_2016_to_Present"
DB_TXN_VIEW = "vwAllTxns_v3"
#DB_PO_VIEW = "vwAllPO_v2"
DB_ERT_VIEW = "vwErt"
#DB_ERT_VIEW = "vwErt_short"


#######################################Data Loading#############################################################

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


def open_conn_oracle():
    connection = cx_Oracle.connect(user="PDTY_EU", password="produ_2015_ctivity_eu", dsn="(DESCRIPTION = (ADDRESS=(PROTOCOL=tcp)(HOST=loesororalp01i.corporate.ge.com)(PORT=1566)) (CONNECT_DATA=(SID=esorp1)))", encoding="UTF-8")
    # connection = cx_Oracle.connect(user="223102584", password="pa55word", dsn="(DESCRIPTION = (ADDRESS=(PROTOCOL=tcp)(HOST=loesororalp01i.corporate.ge.com)(PORT=1566)) (CONNECT_DATA=(SID=esorp1)))", encoding="UTF-8")
    
    print('connected')
    return connection

def open_conn_FBI():
    """ **Open FBI connection as a pyodbc connection**

    Returns
    -------
    conn
        pyodbc.connect
    """
    conn = pyodbc.connect('Driver={SQL Server};'
                        'Server=svc-fbi-db-CFt.mgmt.cloud.ds.ge.com,3433;'
                        'Database=Dev_GPRS_Datamart;'
                        'UID=iCARE;PWD=abcde;'
                        'Trusted_Connection=no;'
                        'timeout = 60')
    return conn

def run_query_sqldb(query, conn):
    """ **Return SQL Query on SQL Data Base in form of a DataFrame**

    Parameters
    ----------
    query : str
        SQL query
    conn
        pyodbc.connect

    Returns
    -------
    pd.DataFrame
        Result of the SQL query
    """
    # Read sql query in the data frame format
    df_query = pd.read_sql_query(query, conn)
    return df_query

def close_connection(connection):
    """**Close pyodbc connection**

    Parameters
    ----------
    connection : pyodbc.connect
        ODBC Data Base Connection
    """
    # release the connection
    if connection:
        connection.close()
    else:
        raise ValueError("Connection was not open {}".format(connection))

def query_construction(param_value, table_name, full=False):
    '''
    param_value: dictionnaire(?)
    table_name: 'txn' ou 'ert'
    '''
    headerdict={'txn':'SELECT [DW_TXN_KEY] [indices],[Transaction Day],[Transaction Type],[Transaction Description] ,[Transaction Sub-Type] ,[Transaction Category] ,[Txn Quantity],[Inventory Org],[Subinventory Code],[ITEM],[Item Description],[Job Number],[SO Number],[PO Number],[LPN],[FE Txn Group] [RMA],[SO PO Link],[Job Open Date],[PO Type],[FE SSO], [FE ORG],[System ID],[System Install Date],[Customer Number],[Customer Name],[Customer Country Code],[Customer Pole],[POLE],[LOCATOR],[Transfer Locator],[Vendor Name],[Vendor Site Code], [SO Type], [Reason Name],[PO Header Description],[Failure Description],[Waybill], [Freight Carrier], [SO PO Link Type],[Obso Pivot Code]',
                'ert': 'SELECT [DW_ERT_KEY] [indices],[ITEM], [Item Description], [Current Item Type], [Current First Put-Away Date (FPAD)], [Current Item Status], [Vendor GSL Number], [Vendor Name],[Vendor Site Code], [PO1 PO Number], [PO1 Need By Date], [PO1 PO Header Description], [PO1 Approved Date], [PO1 Creation Date],  [PO1 Shipment Last Update Date], [PO1 Receipt Date], [PO1 PO Type], [PO1 PO Type DFF], [PO1 Authorization Status], [PO2 PO Number], [SO1 Shipping Warehouse], [FE Number], [FE SSO], [Failure Code],[Part Failure Analysis],[FE Comments],[Supplier Comments],[Analyst Comments], [Repair Status], [Repair Action], [Job Number], [SO1 SO Number], [SO1 SO Status], [SO1 Waybill], [SO1 Order Date], [SO1 Ship Date], [SO2 SO Number], [Serial Number], [Part Arrival Date], [Part Repaired Date]' }
    tabledict={'txn': "vwAllTxns_v3", 'ert': "vwErt"}
    if full==False:
        query_header= headerdict[table_name]
    else:
        query_header='SELECT *'
    # si full = False, seulement sélectionner les colonnes prédéfinis, sinon, sélectionner toutes les colonnes.
    query_table=f'FROM [dbo].[{tabledict[table_name]}] '
    query_criteria='Where (1=1) AND '
    last_param_keys=list(param_value.keys())[-1]
    for e in param_value.keys():
        if e not in ['date_min' , 'date_max']:
            if type(param_value[e])==list:
                if len(param_value[e])>1:
                    s=f'[{e}] IN('
                    for j in param_value[e]:
                        s+=f"'{j}'"
                        if j != param_value[e][-1]:
                            s+=','
                    s+=')'
                else:
                        s=f"[{e}] = '{param_value[e][0]}' "
            else:
                s=f"[{e}] = '{param_value[e]}' "
        elif e== 'date_min':
            s=f"[Transaction Day]>='{param_value[e]}' "
        else:
            s=f"[Transaction Day]<='{param_value[e]}' "
        if e != last_param_keys:
            s+='AND '
        query_criteria+=s
    return query_header+query_table+query_criteria


################################################### Vérification ################################################

def get_keys(data,param):
    #key: liste triée de tous les valeurs dans une colonne de data
    #ukey: liste de keys uniques
    keys=list()
    ukeys=list()
    data.drop_duplicates(inplace=True)
    for i in data.index:
        k=data[param][i] #Pour tous les valeurs dans la colonne data[param]
        if pd.isnull(k)==False and 1 not in [c in k for c in ['?','{', 'Unknown', 'u', 'STOCK', 'RDE RE']]: 
            #Si la valeur existe et n'appartient pas ?,Unknown etc.:
            keys+=[(i,k)] # Ajouter dans la liste 'key'
            if k not in ukeys:
                ukeys+=[k]
    keys.sort()
    return keys, ukeys


def validate(liste):
    #valider s'il y a valeurs redondantes dans la liste
    #return: si continue=true, il n'y a pas redondunte; s'il est vide, c'est une liste vide; sinon il y a des redondantes
    n=len(liste)
    i=0
    try:
        l=[liste[0]]
        continuer=True
    except:
        continuer='vide'
    while continuer==True and i<n:
        if liste[i]==l[-1]: #le cas i==0
            i+=1
        elif liste[i] in l: #si jamais une élément dans liste existe déjà dans l: continue = False
            continuer = False
        else: #la plus part des cas, ajout élément de liste dans l
            l+=[liste[i]]
            i+=1
    return continuer


def verify_one_step(dict_list,param,ndata,orgdata,vcases):
    keysvalues=get_keys(ndata,param) #keysvalues=[<keys>,<ukeys>]
    # cette fonciton a deux returns, du coup keysvalues[0] est key; keysvalues[1] est key unique
    if len(keysvalues[1])==0:
        return ndata, dict_list,orgdata,vcases # s'il y a rien, output=input
    for e in keysvalues[1]: # on cherche les valeurs dans unique keys
        l=dict_list[param]+[i for i in keysvalues[0] if i[1]==e] 
        #l contient dict_list[param] et les éléments dans keys qui égale à e
        l.sort()
        if validate([i[1] for i in l])==False: #s'il n'est pas unique
            ndata=ndata[ndata[param]!=e]
            orgdata1=orgdata[orgdata[param]!=e] #si non-validated, supprimer dans ces deux dicts
            if -1 in orgdata1['step']:
                orgdata=orgdata1
            if param== 'SO PO Link':
                vcases['sopo']+=[e]
            elif param =='SO Number':
                vcases['so']+=[e]
            elif param == 'Job Number':
                vcases['job']+=[e]
            else:
                vcases[param.lower()]+=[e]# Ajouter dans vcases
        else:
            dict_list[param]=l
    return ndata, dict_list,orgdata,vcases 

def get_rma_status(t):
    data=t[t['Transaction Type']=='RMA Receipt'][['Transaction Day','Transaction Sub-Type','Reason Name', 'RMA']]
    status=[]
    for i in data.index:
        e=data['Transaction Sub-Type'][i]
        rma=data['RMA'][i]
        datetxn=data['Transaction Day'][i]
        if 'DEFECTIVE' in e.upper(): #si la raison de txn est defective 
            status+=[(datetxn,'DEFECTIVE',rma,1)] # ajout une ligne dans liste 'status'
        elif 'GOOD' in e.upper():
            status+=[(datetxn,'GOOD',rma,1)]
        elif [1 in [c in e.upper() for c in ['REP', 'REPAIR']]]:
            status+=[(datetxn,'REPAIRED',rma,1)]
        else:
            status+=[(datetxn,'UNKOWN',rma,1)]
    return status

def verify_two_rmas_status(rmaold,rmanew):
    if rmaold=='DEFECTIVE' and rmanew in['GOOD', 'DEFECTIVE']:
        return False
    return True

def verify_rmas(t,laststatus):
    x="""if laststatus[-1]!=' ':
        statusrmas=[laststatus]+get_rma_status(t)
    else:
        statusrmas=get_rma_status(t)
    statusrmas.sort()
    if len(statusrmas)>1:
        for i in range(0,len(statusrmas)-1):
            if verify_two_rmas_status(statusrmas[i][1],statusrmas[i+1][1])==False:
                if statusrmas[i+1][3]==1:
                    removing=statusrmas[i+1][2]
                else:
                    removing=statusrmas[i][2]
                rmas_to_remove+=[removing]
    t=t[~t['RMA'].isin(rmas_to_remove)]
    laststatus=(statusrmas[-1][0],statusrmas[-1][1],statusrmas[-1][2],-1)"""
    return t,laststatus


def verifyItems(ndata,items):
    #extract Items number from new data
    new_items=list(pd.unique(ndata['Item Description']))
    remove_items=[]
    for e in new_items:
        if e not in items[1]:
            remove_items+=[e]
    ndata=ndata[~ndata['Item Description'].isin(remove_items)]
    return ndata
    
def verifyItemsprime(ndata,itemsprime):
    # return: une dataframe qui contient seulement les items primes
    new_items=list(pd.unique(ndata.ITEM))
    remove_items=[]
    for e in new_items:
        prime=e.split('-')[0]
        if prime not in itemsprime:
            remove_items+=[e]
    ndata=ndata[~ndata.ITEM.isin(remove_items)]
    return ndata


def multipleparts(t):
    nparts=list(t.groupby(['SO Number'])['Txn Quantity'].sum()) # Chaque So number, calculer la quantité de transaction
    if abs(sum(nparts))<=1:
        if 0 not in [abs(e)<=1 for e in nparts]: # S'il n'y a pas de 0 dans la liste de nparts 
            i=0
            n=len(nparts)
            multiple= False
            while multiple ==False and i<n-1:
                multiple= (abs(nparts[i]+nparts[i+1])>1) #si la somme de deux items successives >1(c'est pas le cas de -1;1), multiple==true
                i+=1
            return multiple
    return True

def verify(dict_list,ndata,orgdata,vcases,itemsprime,parameter,value,laststatus,job=False):
    ## verifying items:
    ndata_items=verifyItemsprime(ndata,itemsprime)
    vcases_items=vcases
    vcases_items[parameter]+=[value]
    ## verifying the rmas status
    ndata_rmas,laststatus=verify_rmas(ndata_items,laststatus)
    #cohérence verification
    ndata_verified, dict_list_verified,orgdata_verified,vcases_verified=verify_one_step(dict_list,'RMA',ndata_rmas,orgdata,vcases_items)
    ndata_verified, dict_list_verified,orgdata_verified,vcases_verified=verify_one_step(dict_list_verified,'SO PO Link',ndata_verified,orgdata_verified,vcases_verified)
    if job==False:
        ndata_verified_final, dict_list_verified_final,orgdata_verified_final,vcases_verified_final=verify_one_step(dict_list_verified,'LPN',ndata_verified,orgdata_verified,vcases_verified)
        return ndata_verified_final, dict_list_verified_final,orgdata_verified_final,vcases_verified_final,laststatus
    else:
        ndata_verified, dict_list_verified,orgdata_verified,vcases_verified=verify_one_step(dict_list_verified,'LPN',ndata_verified,orgdata_verified,vcases_verified)
        ndata_verified_final, dict_list_verified_final,orgdata_verified_final,vcases_verified_final= verify_one_step(dict_list_verified,'SO Number',ndata_verified,orgdata_verified,vcases_verified)
        return ndata_verified_final, dict_list_verified_final,orgdata_verified_final,vcases_verified_final,laststatus

############################################## Search Process ##################################################
def FOA_followup(nd,item):
    #item est la catégorie de item(oui)
    #nd: nombre des jours
    L=()
    if item=='XFD':
        L=('5341543-52-R','5341543-52','5341543-2-R','5341543-2','5341541-52-R','5341541-52','5341541-2-R','5341541-2')
    elif item=='TSUI':
        L=('5192485-R','5309869-R','5413430-R','5443025-R','5145783-2-R','5451371-R','2237456-10-R','2237456-11-R','5184723-R','5413421-R','2237457-15-R','2237457-16-R','5192486-R','5309865-R','5413431-R','5448563-R','5460295-R','5145782-3-R','5413426-R','5669691-R','5693695-R','5413428-2-R','5413429-R','5693721-R','5692509-R','5413427-R','5148356-2-R','2237458-5-R','2237459-11-R','2356630-6-R','2356630-7-R','5413432-R','5449772-R','5309862-R','5413424-R','5184194-R','2347974-4-R','5192484-R','5438999-R','5192485','5309869','5413430','5443025','5145783-2','5451371','2237456-10','2237456-11','5184723','5413421','2237457-15','2237457-16','5192486','5309865','5413431','5448563','5460295','5145782-3','5413426','5669691','5693695','5413428-2','5413429','5693721','5692509','5413427','5148356-2','2237458-5','2237459-11','2356630-6','2356630-7','5413432','5449772','5309862','5413424','5184194','2347974-4','5192484','5438999')
    elif item== 'Coils':
        L=('2418093-R','2418093','2416759-R','2416759','5806195-R','5806195','5806196-R','5806196')
    else:
        L=input("Please input the prime item number (without '-R') \n Accepted format: \n 2418093 2416759 5806195 5806196 or 2418093,2416759,5806195,5806196 \n ")
        L=[e.split(',') for e in [e for e in L.split(' ') if e !=' '] if e!= ','] 
        #Si e n'est pas vide(' '), split les string avec ' ', et puis si string n'est pas ',' split les strings avec ','
        L=[item for sublist in L for item in sublist if type(sublist)==list]+[item for item in L if type(item)!=list]
        L+=[e+'-R' for e in L]
        # Ajout un '-R' dans la liste L
        L=tuple(L)
    query=f"""SELECT DEC.DECLARATION_DATE
,REG.POLE
,REG.REGION
,REG.COUNTRY
,DEC.ITEM
,GPRS.ITEM_DESCRIPTION
,DEC.REASON_CODE
,DEC.QUANTITY QTY
,DEC.FE_SSO_NUMBER FE_SSO
,DEC.SSO_ID DECLARING_SSO
,DEC.SYSTEM_ID
,ID.MUST_INSTALL_DATE
--,ID.INSTALL_DATE as GIB_INSTALL_DATE
,DEC.RFS_NUMBER
,DEC.RMA
,DEC.FAILURE_DESCRIPTION
,GPRS.ITEM_TYPE
,GPRS.MMICV_USD
,DEC.ORDER_NUMBER
,GPRS.MODALITY
,ID.MODALITY_SERVICE_1
,ID.MODALITY_SERVICE_4
,DEC.CARRIER_NAME
,DEC.WAYBILL_NUMBER

FROM GLPROD_DECLARATIONS DEC
LEFT JOIN MAP_COUNTRY_REGION REG ON REG.COUNTRY_CODE = DEC.COUNTRY_CODE
LEFT JOIN MASTER_EU.DATA_ITEM_GPRS GPRS ON GPRS.ITEM = DEC.ITEM
LEFT JOIN DATA_MUST_SYSTEM_ID ID ON ID.SYSTEM_ID = DEC.SYSTEM_ID
LEFT JOIN MASTER_EU.DATA_GIB GIB ON GIB.SYSTEM_ID = DEC.SYSTEM_ID

WHERE (1=1)
AND DEC.ITEM IN {L}
AND DECLARATION_DATE>=(sysdate-{nd})
AND REASON_CODE IN ('DOA/FOA', 'DOI/FOI')
AND DEC.QUANTITY <> '0'

ORDER BY DEC.DECLARATION_DATE DESC"""
    conn= open_conn_oracle()
    t=run_query_sqldb(query,conn)
    close_connection(conn)
    return t 

def trace_one_step(param, value,table, item=[], dates={}):
    '''data: la base de données à trier
    param: le paramètre à suivre
    value: la valeur chercher dans data[param]'''
    if table== 'txn':
        d={'rma': 'RMA', 'lpn': 'LPN','job': 'Job Number', 'so': 'SO Number', 'sopo': 'SO PO Link', 'po':'PO Number'}
        #rma ici! est FE Txn Group
        TABLE=TXN
    else:
        d={'po':'PO1 PO Number', 'so':'SO1 SO Number', 'Item':'Item', 'sr': 'Serial Number', 'job':'Job Number'}
        TABLE=ERT
    if param not in ['rma', 'lpn'] and table== 'txn':
        param_value={d[param]:value, 'ITEM':item} 
        #si la valeur à chercher n'est pas rma ou lpn, ajout le param dans param_value
    elif len(item)>0 and table == 'txn':
        param_value={d[param]:value, 'ITEM':item}
        #Il n'y aura pas d'intersection entre ces deux conditions? 
        #par ex. - param='po' -traced=trace_one_step(param,value,'txn',items[0], date_dict)
    else:
        param_value={d[param]:value}
    if len(dates.keys())!=0:
        for e in dates.keys():
            param_value[e]=dates[e]
    try:
        df_param = pd.DataFrame(data=param_value,index=[0])
    except:
        df_param = pd.DataFrame(data=param_value,index=[0,1])
    begin=time.time()
    merge_col=df_param.columns.to_list()
    # print(merge_col)
    if 'date_min' not in merge_col and 'date_max' not in merge_col:
        sdata=pd.merge(df_param,TABLE,on=merge_col)
    elif 'date_min' in merge_col and 'date_max' in merge_col:
        # print(merge_col)
        merge_col.remove('date_min')
        merge_col.remove('date_max')
        sdata=pd.merge(df_param, TABLE[(TABLE['Transaction Day']>=df_param['date_min'].iloc[0])&(TABLE['Transaction Day']<=df_param['date_max'].iloc[0])], on=merge_col)
    elif 'date_min' in merge_col:
        merge_col.remove('date_min')
        sdata=pd.merge(df_param, TABLE[TABLE['Transaction Day']>=df_param['date_min'].iloc[0]], on=merge_col)
    else:
        merge_col.remove('date_max')
        sdata=pd.merge(df_param, TABLE[TABLE['Transaction Day']<=df_param['date_max'].iloc[0]], on=merge_col)
    end=time.time()
    # print('it takes',end-begin, 's')
    sdata=sdata.set_index('indices')
    if table=='txn':
        #sdata=sdata.astype('str')
        sdata=sdata.sort_index(ascending=False) #txn descending
    else:
        sdata=sdata.sort_index(ascending=True)
    return sdata


def get_so_from_po(po):
    q=f"SELECT [SO_NUMBER] FROM [dbo].[tblDimSO] Where (1=1) AND CUST_PO_NUMBER ='{po}'"
    so=run_query_sqldb(q,open_conn_FBI())['SO_NUMBER'][0]
    return so

def get_param(traced_data,colname):
    return [e for e in pd.unique(traced_data[colname]) if pd.isnull(e)== False and 1 not in [c in e for c in ['?','{', 'Unknown', 'u', 'STOCK','stock', 'RDE RE', 'NA']]]


def get_params(traced_data, repair=False):
    if repair==False:
        return {'rma': get_param(traced_data,'RMA'),'lpn': get_param(traced_data,'LPN'),'job':get_param(traced_data,'Job Number'), 'so': get_param(traced_data,'SO Number'), 'sopo': get_param(traced_data,'SO PO Link'),'po':get_param(traced_data,'PO Number') }
    return {'rma': [],'lpn': [],'job':get_param(traced_data,'Job Number'), 'so': get_param(traced_data,'SO1 SO Number'), 'sopo': [],'po':get_param(traced_data,'PO1 PO Number') }

def get_items(data):
    itemnumber=list(pd.unique(data.ITEM))
    itemdescription=list(pd.unique(data['Item Description']))
    itemprime=list(pd.unique([e.split('-')[0] for e in itemnumber]))
    return[itemnumber,itemdescription, itemprime]

def update_values(od,nd):
    for k in od.keys():
        od[k]+=[e for e in nd[k] if e not in od[k]]
    return od

def get_component_log(repair_po):
    query1=f"""select 
[TRANSACTION_REFERENCE] as 'Txn Reference',
[PART_NUMBER] as 'Component',
[ITEM_DESCRIPTION] as 'Component Description',
[TRANSACTION_DAY] as 'Transaction Day',
[TRANSACTION_QUANTITY] as 'Transaction Quantity'
from [dbo].[ALL_TXNSQ]
where [TRANSACTION_REFERENCE] LIKE '{repair_po}-%';"""
    data=open_conn_FBI()
    t=run_query_sqldb(query1,data)
    close_connection(data)
    used_parts=[]
    t['Repair PO']=[repair_po for _ in t.index]
    for i in t.index:
        if 'U' in t.Component[i].upper() or 'RECUP' in t['Txn Reference'][i].upper():
            used_parts+=['YES']
        else:
            used_parts+=['NO']
    t['Used Parts']=used_parts
    return t

def get_repair_log(param,value):
    t=trace_one_step(param,value,'ert')
    #obtenir tous les données dans 'ert' qui satisfait la condition de param&value
    try:
        t.drop_duplicates(inplace=True)
        sr=t['Serial Number'][t.index[0]]
        print(f"sr {sr}")
    except:
        sr='N/A'
        return -1
    if sr not in ['N/A','None', 'NONE','.','?', ' '] and pd.isnull(sr)==False:
        repairslog=trace_one_step('sr',sr, 'ert')
        #une fois obtenir le serial number, trouver les infos concernant sr dans ert
        #repairslog=repairslog[repairslog['Repair Action']!= '{Blank}'][repairslog['Repair Action'].isnull()==False].sort_values(['PO1 PO Number'], ignore_index= True)
        repairslog=repairslog[repairslog['Repair Action']!= '{Blank}'][repairslog['Repair Action'].isnull()==False]
        print(repairslog['PO1 PO Number'].reset_index(drop=True))
        return repairslog
    else:
        t=t[t['Repair Action']!= '{Blank}'][t['Repair Action'].isnull()==False]
    return t
def init_repairlog_search(traced_data):
    # l'objectif de cette fonction est d'initier la recherche dans ert
    # Si po type in traced_data est 'repair', repairkey=true
    try:
        repairkey=[po for po in traced_data['PO Number'][traced_data['PO Type'].str.upper()=='REPAIR']][0]
    except:
        return -1
    repairlog=get_repair_log('po',repairkey)
    print('repair history saved')
    return repairlog

def add_rows(data, ndata):
    d=data
    pindex=list(data.index)
    nindex=list(ndata.index)
    for l in nindex:
        if l not in pindex:
            d=d.append(ndata.loc[l])
    return d.sort_index(ascending=False)

def endtracing(vd,dtv):
    for k in vd.keys():
        s= 0 in [e in vd[k] for e in dtv[k]]
        if s==True:
            return False # S'il contient 0; return false sinon, true
    return True

def multiple_items(traced,items):
    choose_item='''Multiple items detected \n'''
    for e in items[0]:
        choose_item+= f'''- {e} \n'''
    # Afficher tous les items dans 'items' et laisser utilisateur de choisir un
    choose_item+='''Please choose one part number \n'''
    item_number=input(choose_item)
    traced=traced[traced.ITEM==item_number]
    items=get_items(traced)# unique items
    return items,traced

def extract_date(date):
    datestr=f'{date.year}'
    if date.month<10:
        datestr+=f'0{date.month}'
    else:
        datestr+=f'{date.month}'
    if date.day<10:
        datestr+=f'0{date.day}'
    else:
        datestr+=f'{date.day}'
    return datestr

def get_date(data,name, param, value, direction='A'):
    '''
    name: nom du tableau
    param: paramètre qu'on veut voir
    return: {key:value}
    '''
    if name== 'txn':
        d={'rma': 'RMA', 'lpn': 'LPN','job': 'Job Number', 'so': 'SO Number', 'sopo': 'SO PO Link', 'po':'PO Number'}
        localparam=d[param]
        # si tableau est transaction, changer le nom de param et sauvegarder
    else:
        localparam=param
    try:
        indice=data[data[localparam]==value].index[0]
        # Chercher l'indice de premier "localparam=value" dans data
        #print(indice)
    except:
        print('Error')
        return {}
    if name=='repairlog':
        dates=data[['Part Arrival Date', 'Part Repaired Date']].max(axis=1)
    else:
        dates=data['Transaction Day']
    if name=='repairlog':
        if direction in ['B','A']:
            if indice>0 and (dates[indice]-dates[indice-1]).days>=20:
                return {'date_min':extract_date(dates[indice-1]), 'date_max':extract_date(dates[indice])}
            else:
                return {'date_max':extract_date(dates[indice])}
        elif direction =='F':
            try : 
                if (dates[indice+1]-dates[indice]).days<=30:
                    return {'date_min':extract_date(dates[indice])}
                return {'date_min':extract_date(dates[indice]), 'date_max':extract_date(dates[indice+1])}
            except:
                return {'date_min':extract_date(dates[indice])}
    elif direction=='F':
        return {'date_min':extract_date(dates[indice])}
    elif direction =='B':
        return {'date_max':extract_date(dates[indice])}
    else:
        return {}

def transform_todate(txt):
    l=txt.split(' ')[0].split('T')[0].split('-')
    # obtenir la première partie séparé par ' 'et'T' et ensuite split it by '-'
    return datetime.date(int(l[0]),int(l[1]),int(l[2]))

def update_dates_reduced(hist,direction,date_dict,depart_date):
    t=hist[hist.step!=-1] # colonne 'step' dans hist où step!=-1
    continueiter=True
    j=0
    n=len(t.index)
    while continueiter==True and j<n:
        i=t.index[j]
        e=t['Transaction Sub-Type'][i]
        if 1 in [c in str(e).upper() for c in ['RDE REPAIR', 'GPO REPAIR']]:
            selected_date=extract_date(transform_todate(str(t['Transaction Day'][i])))
            continueiter= False
        j+=1
    # Boucle: obtenir indice et 'transaction sub-type', si e contient 
    # 'RDE REPAIR' ou 'GPO REPAIR', sorte de boucle et mettre selected_date comme 'Transaction Day'
    if continueiter==True:
        return date_dict
    if direction == 'F'and 'date_min' in date_dict.keys() and date_dict['date_min']<selected_date:
        date_dict['date_max']=selected_date
    if direction == 'B' and 'date_max' in date_dict.keys() and date_dict['date_max']>selected_date:
        date_dict['date_min']=selected_date
    if direction =='A':
        if depart_date>selected_date:
            date_dict['date_min']=selected_date
        elif depart_date<selected_date:
            date_dict['date_max']=selected_date
    # Changer la date_dict en fonction de la direction 
    return date_dict

def update_dates(param,value,t,repairlog,direction,date_dict):
    try:
        po=t[t['PO Type'].str.upper()=='REPAIR']['PO Number'].index[0]
    except:
        return date_dict
    if direction=='F':
        dict_local=get_date(repairlog,'repairlog','PO1 PO Number',po , 'B')
        if 'date_max' in dict_local.keys():
            date_dict['date_max']= dict_local['date_max']
    elif direction== 'B':
        dict_local=get_date(repairlog,'repairlog','PO1 PO Number',po , 'F')
        if 'date_min' in dict_local.keys():
            date_dict['date_min']= dict_local['date_min']
    elif direction=='A':
        dict_local=get_date(t, 'txn',param,value,direction='F')
        Ldates=list(repairlog['PO1 Receipt Date'].unique())
        Ldates=[extract_date(transform_todate(str(e))) for e in Ldates if pd.isnull(e)==False and str(e)!='NaT']
        Ldates+=[dict_local['date_min']]
        Ldates.sort()
        indix= Ldates.index(dict_local['date_min'])
        try:
            date_dict['date_min']=Ldates[indix-1]
        except:
            pass
        try:
            date_dict['date_max']=Ldates[indix+1]
        except:
            pass
    return date_dict

## final process
def FE_analysis(data):
    dataprocessed=data[data['Transaction Type']=='RMA Receipt'][data['PO Type'].str.upper()!='REPAIR'][['Transaction Day','Transaction Sub-Type','Reason Name','Failure Description','ITEM','FE SSO','Inventory Org','RMA', 'LPN', 'Job Number','System ID','Customer Number', 'Customer Name','Customer Country Code','Customer Pole', 'Waybill','Freight Carrier']]
    # Choisir les lignes où transaction type est rma receipt, et po type n'est pas repair; choisir les colonnes dédié
    status=[]
    for e in dataprocessed['Transaction Sub-Type']:
        if 'DEFECTIVE' in e.upper():
            status+=['DEFECTIVE']
        elif 'GOOD' in e.upper():
            status+=['GOOD']
        elif 1 in [c in e.upper() for c in ['REP', 'REPAIR']]:
            status+=['REPAIRED']
        else:
            status+=['UNKOWN']
    # Définir status par rapport à 'Transaction Sub-Type'
    dataprocessed['Transaction Sub-Type']=status 
    return dataprocessed.rename(columns={'Transaction Day': 'Return Date', 'Transaction Sub-Type':'Status'})
    
##cases builder 
def case_builder(data_index,data,datar):
    '''
    data_index: L'indice de donnée
    data
    datar: donnée de répare
    return: liste
    '''
    to_network=['PO Receipt', 'RMA Receipt', 'GPO RMA Receipt'] 
    # Ces trois mots:pour 'flow in' 
    out=['Sales order issue','GPO RMA Issue']
    # Ces deux mots pour 'shipped from'
    print('dataindex:',data_index)
    txnt=data['Transaction Type'][data_index]
    txnst=data['Transaction Sub-Type'][data_index]
    txnd=data['Transaction Description'][data_index]
    subcod=data['Subinventory Code'][data_index]
    pot=data['PO Type'][data_index]
    sot=data['SO Type'][data_index]
    tcat=data['Transaction Category'][data_index]
    print(txnt, to_network)
    if txnt in to_network:
        if txnt == 'PO Receipt':
            if pot== 'DEFECTIVE':
                return {'case':'Defective','lpn':data['LPN'][data_index], 'flow':'Received in', 'org': data['Inventory Org'][data_index], 'po': data['PO Number'][data_index]}
            if pot =='REPAIR':
                ponumber= data['PO Number'][data_index]
                repairinfo={'case':'Repair_supplier','lpn':data['LPN'][data_index], 'flow':'Received in', 'org': data['Inventory Org'][data_index], 
                            'vname': data['Vendor Name'][data_index],'po': data['PO Number'][data_index]}
                try:
                    repair=datar[datar['PO1 PO Number']==ponumber].reset_index()
                    repairinfo['pfa']=repair['Part Failure Analysis'][0]
                    repairinfo['ra']=repair['Repair Action'][0]
                    repairinfo['rs']=repair['Repair Status'][0]
                    #return repairinfo
                except:
                    repairinfo['pfa']='NO RECORD'
                    repairinfo['ra']='NO RECORD'
                    repairinfo['rs']='NO RECORD'
                return repairinfo
            if pot =='Affiliate':
                return {'case':'Allocation','lpn':data['LPN'][data_index], 'flow':'Received in', 'org': data['Inventory Org'][data_index], 'po': data['PO Number'][data_index],
                        'so': data['SO Number'][data_index]}
            if pot == 'HARVEST':
                return {'case':'Harvest','lpn':data['LPN'][data_index], 'flow':'Received in', 'org': data['Inventory Org'][data_index], 'vname': data['Vendor Name'][data_index],
                        'po': data['PO Number'][data_index], 'poheader': data['PO Header Description'][data_index]}
        if txnt in ['RMA Receipt','GPO RMA Receipt']:
            if sot== 'GPO_US_CUST_RETURN':
                return {'case':'DCOS','lpn':data['LPN'][data_index], 'flow':'Received in', 'org': data['Inventory Org'][data_index],'cname':data['Customer Name'][data_index],
                        'so': data['SO Number'][data_index]}
            if txnt == 'RMA Receipt':
                if 'Repair' in txnst:
                    if 'STR' in subcod:
                        return {'case':'RDE','lpn':data['LPN'][data_index], 'flow':'Received in', 'org': data['Inventory Org'][data_index], 'job': data['Job Number'][data_index],
                                'so': data['SO Number'][data_index]}
                    else:
                        return {'case':'Repair','lpn':data['LPN'][data_index], 'flow':'Received in', 'org': data['Inventory Org'][data_index], 'job': data['Job Number'][data_index],
                                'so': data['SO Number'][data_index]}
                mss_info={'case':'RMA', 'lpn':data['LPN'][data_index], 'flow':'Received in', 'org': data['Inventory Org'][data_index],'FE': data['FE SSO'][data_index],
                          'rma': data['RMA'][data_index],'debrief': data['Reason Name'][data_index],'so': data['SO Number'][data_index]}
                if 'DEFECTIVE' in data['Transaction Sub-Type'][data_index].upper():
                    mss_info['status']= 'DEFECTIVE'
                elif 'GOOD' in data['Transaction Sub-Type'][data_index].upper():
                    mss_info['status']= 'GOOD'
                else:
                    mss_info['status']= 'UNKOWN'
                return mss_info
        return {'case':'Supplier','lpn':data['LPN'][data_index], 'flow':'Received in', 'org': data['Inventory Org'][data_index], 'vname': data['Vendor Name'][data_index],
                'po': data['PO Number'][data_index], 'poheader': data['PO Header Description'][data_index], 'so': data['SO Number'][data_index]}
    if txnt in out:
        if txnt== 'Sales order issue':
            if 'Repair shipments' in txnst:
                return {'case':'Repair_supplier','lpn':data['LPN'][data_index], 'flow':'Shipped from', 'org': data['Inventory Org'][data_index], 'vname': data['Vendor Name'][data_index],
                        'po': data['PO Number'][data_index], 'so': data['SO Number'][data_index]}
            if 0 not in [e in sot for e in ['GPO','FE','SHIPMENT']]:
                return {'case':'Dropship', 'lpn':data['LPN'][data_index], 'flow':'Shipped from', 'org': data['Inventory Org'][data_index],'FE': data['FE SSO'][data_index] ,
                        'rma': data['RMA'][data_index], 'job':data['Job Number'][data_index], 'sysid': data['System ID'][data_index], 'so': data['SO Number'][data_index] }
            if 0 not in [e in sot for e in ['GPO','DEF','AFFL']]:
                return {'case':'Defective','lpn':data['LPN'][data_index], 'flow':'Shipped from', 'org': data['Inventory Org'][data_index], 'po': data['PO Number'][data_index],
                        'so': data['SO Number'][data_index]}
            if 0 not in [e in sot for e in ['GPO','DCOS','SHIP']]:
                return {'case':'DCOS','lpn':data['LPN'][data_index], 'flow':'Shipped from', 'org': data['Inventory Org'][data_index],'cname':data['Customer Name'][data_index],
                        'so': data['SO Number'][data_index]}
            if 'Rebalancing' in tcat:
                if data['SO PO Link Type'][data_index] == 'GOOD':
                    if data['Obso Pivot Code'][data_index]== 'Defective':
                        return {'case':'Defective','lpn':data['LPN'][data_index], 'flow':'Shipped from', 'org': data['Inventory Org'][data_index], 'po': data['PO Number'][data_index], 
                                'so': data['SO Number'][data_index]}
                    if data['Obso Pivot Code'][data_index]== 'GIT':
                        return {'case':'Rebalancing','lpn':data['LPN'][data_index], 'flow':'Shipped from', 'org': data['Inventory Org'][data_index], 'po': data['PO Number'][data_index],
                                'so': data['SO Number'][data_index]}
            if 'Allocation OU' in txnd:
                return {'case':'Allocation','lpn':data['LPN'][data_index], 'flow':'Shipped from', 'org': data['Inventory Org'][data_index], 'po': data['PO Number'][data_index],
                        'so': data['SO Number'][data_index]}
    return -1            
## message builder
def mess_builder(dfcase):
    # input dfcase: return of case_builder fonction
    # Inner function - provide the traceability message vs part origin if the part transaction is a reception
    # Return message corresponding to orgin defined by "case"
    # Used in main "trace_back_lpn" function

    # init - not mandatory
    mess_org = ""
    # simple switch vs "case" value
    if dfcase['case'] == "Defective":
        if  dfcase['flow']== 'Shipped from':
            mess_org = "{} {} allocation sales order SO {}".format(
             dfcase['flow'], dfcase["org"], dfcase["so"])
        else:
            mess_org = "{} {} on defective PO {}".format(
                 dfcase['flow'], dfcase["org"], dfcase["po"])
    elif dfcase['case'] == "RDE":
        mess_org = "{} {} on defective PO {}".format(
             dfcase['flow'], dfcase["org"], dfcase["job"])
    elif dfcase['case']=='Dropship':
        mess_org="{} {} to FE {} for Job number {}/ System ID {} on SO {} (RMA {})".format( dfcase['flow'], dfcase["org"],dfcase["FE"], dfcase['job'],dfcase['sysid'], dfcase['so'], dfcase["rma"])
    elif dfcase['case'] == "RMA":
        if dfcase['flow']== 'Received in':
            mess_org = "{} {} as {} from FE {} return using RMA {} with debrief code: {}".format(
                 dfcase['flow'], dfcase["org"], dfcase["status"], dfcase["FE"], dfcase["rma"], dfcase["debrief"])
    elif dfcase['case'] == "Allocation":
        if dfcase['flow']== 'Shipped from':
            mess_org = "{} {} on allocation sales order SO {}".format(
             dfcase['flow'], dfcase["org"], dfcase["so"])
        else:
            mess_org = "{} {} on allocation PO {} linked to SO {}".format(
             dfcase['flow'], dfcase["org"], dfcase["po"], dfcase["so"])
    elif dfcase['case'] == "Supplier":
        mess_org = "{} {} from new-buy supplier {} on PO number {} with header description {}".format(
            dfcase['flow'], dfcase["org"], dfcase["vname"], dfcase["po"], dfcase["poheader"])
    elif dfcase['case'] == "DCOS":
        if dfcase['flow']== 'Shipped from':
            mess_org = "{} {} to customer {} on SO number {}".format(
            dfcase['flow'], dfcase["org"], dfcase["cname"], dfcase["so"])
        else:
            mess_org = "{} {} from customer {} on return SO number {}".format(
            dfcase['flow'], dfcase["org"], dfcase["cname"], dfcase["so"])
    elif dfcase['case'] == "Harvest":
        mess_org ="{} {} from harvest supplier {} on PO number {} with PO header description {}".format(
            dfcase['flow'], dfcase["org"], dfcase["vname"], dfcase["po"], dfcase["poheader"])
    elif dfcase['case'] == "Repair_supplier":
        if dfcase['flow']== 'Shipped from':
              mess_org = "{} {} on repair sales order SO {}".format(
             dfcase['flow'], dfcase["org"], dfcase["so"])
        else:
            mess_org = "{} {} from repair supplier {} on PO number {} (Part failure analysis: {} " \
            "/ Repair status: {} / Repair action: {})".format(dfcase['flow'], dfcase["org"], dfcase["vname"], dfcase["po"], dfcase['pfa'], dfcase['rs'], dfcase['ra'])            
    elif dfcase['case'] == "Repair":
        mess_org = "{} {} on RDE PO number {}".format(
             dfcase['flow'], dfcase["org"], dfcase["job"])
    else:
        mess_org = "Case {} does not exist".format(dfcase['case'])

    return mess_org
##data_processing
def data_processing(results):
    dcomments=[]
    d_date=[]
    d_LPN=[]
    for e in results['txn history'].index: 
        # results contient plusieurs txn history; pour chaque transaction, construire un 'case'
        if pd.isna(e)==False:
            dcase= case_builder(e,results['txn history'],results['repair history'])
            if dcase!=-1:  # dcase n'est pas vide
                msg=mess_builder(dcase)
                if "Case" not in msg:
                    d_date+=[results['txn history']['Transaction Day'][e]]
                    d_LPN+=[results['txn history']['LPN'][e]]
                    dcomments+=[msg]
    results['comments']=pd.DataFrame()
    results['comments']['Date']=d_date
    results['comments']['LPN']=d_LPN
    results['comments']['Comments']=dcomments
    return results
#extracting next repair po
def get_next_repairpo(so):
    q=f"SELECT [CUST_PO_NUMBER] FROM [dbo].[tblDimSO] Where (1=1) AND SO_NUMBER ='{so}'"
    # Sélectionner les po number qui lient avec so
    data=open_conn_FBI()
    l=list(run_query_sqldb(q,data)['CUST_PO_NUMBER'])
    if len(l)>0:
        po= l[0]
        since=list(run_query_sqldb(f"SELECT [PO1 Receipt Date] FROM [dbo].[vwErt] Where [PO1 PO Number]='{po}'",data)['PO1 Receipt Date'])
        if len(since)>0:
            try:
                since=str([e for e in since if pd.isnull(e)==False][0]).split(' ')[0]
            except:
                since='NONE'
        else:
            since='on the way for delivery'
        return (po,since)
    return ('Unfound','Unfound')

#search summary
def summary_page(results,direction,param,value):
    
    if results['warning'].shape[0]>0:
        multiple_parts= 'This tool uses data from the GPRS datawarehouse (FBI) that is not GxP validated. Use this information with caution and at your own risk !'
    else:
        multiple_parts= 'Based on the retrieved data from FBI, there is no grouped shipment !'
    if results['repair history'].shape[0]>0:
        SN_number=[e for e in list(results['repair history']['Serial Number'].unique()) if pd.isnull(e)== False and e not in ['N/A','None', 'NONE','.','?', ' ']]
        if len(SN_number)>0:
            SN_number=SN_number[0]
        else:
            SN_number=''
    else:
        SN_number=''
    try:
        start_index=results['txn history'][results['txn history'].step.isin(['-1',-1])].index[1]
    except:
        start_index=results['txn history'][results['txn history'].step.isin(['-1',-1])].index[0]
    start_date=results['txn history']['Transaction Day'][start_index]
    if direction=='F':
        history=results['txn history'][results['txn history'].index>=start_index][results['txn history']['Txn Quantity']>0][['POLE','Inventory Org']]
    else:
        history=results['txn history'][results['txn history'].index<=start_index][results['txn history']['Txn Quantity']>0][['POLE','Inventory Org']]
    npole=1
    historypole=[e for e in history['POLE']]
    history=[e for e in history['Inventory Org']]
    nwhse=list(pd.unique(history))
    nwhse=len([e for e in nwhse if e[0]!='S'])
    history=[history[-i] for i in range(1,len(history)+1)]
    #constructing shipment process
    n=len(history)
    if n>0:
        process=history[0]+'>'
        for i in range(1,n):
            if history[i]!=history[i-1]:
                process+=(history[i]+'>')
            if historypole[i]!=historypole[i-1]:
                npole+=1
        process=process[:-1]
    else: 
        process= 'Unable to compute the process owner'
    #getting the number of repair
    repairqty= results['repair history'].shape[0]
    #last repair information
    try:
        rephistory=results['repair history'][results['repair history']['PO1 Receipt Date']<start_date].index[-1]
        previous_repair_action=results['repair history']['Repair Action'][rephistory]
        previous_repair_center_code=results['repair history']['Vendor Site Code'][rephistory]
        previous_repair_center_name=results['repair history']['Vendor Name'][rephistory]
        previous_repair_date=results['repair history']['Part Repaired Date'][rephistory]
        previous_FE_comments=results['repair history']['FE Comments'][rephistory]
        previous_supplier_comments=results['repair history']['Supplier Comments'][rephistory]
        previous_analyst_comments=results['repair history']['Analyst Comments'][rephistory]
        repair_po=results['repair history']['PO1 PO Number'][rephistory]
        Failure_code=results['repair history']['Failure Code'][rephistory]
        #number of consumption on the repair
        numbercomponents=results['repair details'][results['repair details']['Repair PO']==repair_po].shape[0]
    except:
        previous_repair_action='NONE'
        previous_repair_date='NONE'
        previous_FE_comments='NONE'
        previous_supplier_comments='NONE'
        previous_analyst_comments='NONE'
        previous_repair_center_name='NONE'
        previous_repair_center_code='NONE'
        repair_po='NONE'
        Failure_code='NONE'
        numbercomponents= 0
    #number of returns
    nreturns= results['FE Returns summary'].shape[0]
    nsurplus=results['FE Returns summary'][results['FE Returns summary']['Status']=='GOOD'].shape[0]
    #part status :
    if 'DEFECTIVE' in list(results['FE Returns summary']['Status']):
        if 'DOA/FOA' in list(results['FE Returns summary']['Reason Name'])or 'FOA' in list(results['FE Returns summary']['Reason Name']) or 'DOA' in list(results['FE Returns summary']['Reason Name']):
            status ='FOA'
            declaration=results['FE Returns summary'][results['FE Returns summary']['Reason Name'].isin(['DOA/FOA','FOA','DOA'])].reset_index()[['Customer Country Code','Customer Pole']]
            customer_country_code,customer_pole=declaration['Customer Country Code'][0],declaration['Customer Pole'][0]
        elif 'DOI/FOI' in list(results['FE Returns summary']['Reason Name']) or 'FOI' in list(results['FE Returns summary']['Reason Name']) or 'DOI' in list(results['FE Returns summary']['Reason Name']):
            status ='FOI'
            declaration=results['FE Returns summary'][results['FE Returns summary']['Reason Name'].isin(['DOI/FOI','FOI','DOI'])].reset_index()[['Customer Country Code','Customer Pole']]
            customer_country_code,customer_pole=declaration['Customer Country Code'][0],declaration['Customer Pole'][0]
        else:
            status='Debrief'
            declaration=results['FE Returns summary'][results['FE Returns summary']['Status']=='DEFECTIVE'].reset_index()[['Customer Country Code','Customer Pole']]
            customer_country_code,customer_pole=declaration['Customer Country Code'][0],declaration['Customer Pole'][0]
    elif 'GOOD' in list(results['FE Returns summary']['Status']):
        status= 'Shipped to FE and Returned Good'
        declaration=results['FE Returns summary'][results['FE Returns summary']['Status']=='GOOD'].reset_index()[['Customer Country Code','Customer Pole']]
        customer_country_code,customer_pole=declaration['Customer Country Code'][0],declaration['Customer Pole'][0]
    else:
        status= 'Not Consumed'
        if results['txn history'].shape[0]>0:
            customer_country_code=list(results['txn history']['Customer Country Code'])[0]
            customer_pole=list(results['txn history']['POLE'])[0]
    #Last_position
    if direction in ['F','A']:
        sample_data=results['txn history'][results['txn history']['LOCATOR']!='{Blank}'][['Transaction Day','Inventory Org','Subinventory Code','SO Number','LPN', 'RMA','SO PO Link', 'LOCATOR', 'SO Type','PO Number','PO Type']].reset_index(drop=True)
        sample_data['indice']=[-1*e for e in sample_data.index]
        sample_data=sample_data.sort_values(['Transaction Day','indice'], ascending=False).reset_index(drop=True)
        if 'REPAIR_SHIPMENT' in sample_data['SO Type'][0]:
            last_position='Repair Center'
            future_key='Repair PO'
            future_value,since= get_next_repairpo(sample_data['SO Number'][0])
            Locator=sample_data['LOCATOR'][0]
            subinventory= 'NONE'
        elif 'REPAIR' in sample_data['PO Type'][0]:
            last_position= sample_data['Inventory Org'][0]
            future_key='Repair PO'
            future_value= sample_data['PO Number'][0]
            since= str(sample_data['Transaction Day'][0]).split(' ')[0]
            Locator=sample_data['LOCATOR'][0]
            subinventory= sample_data['Subinventory Code'][0]
        else:
            last_position= sample_data['Inventory Org'][0]
            future_key,future_value= [e for e in [('LPN',sample_data['LPN'][0]),('RMA',sample_data['RMA'][0]), ('SO/PO', sample_data['SO PO Link'][0]),('SO Number', sample_data['SO Number'][0]),('NONE','NONE')] if pd.isnull(e[1])==False and '{' not in e[1]][0]
            since=str(sample_data['Transaction Day'][0]).split(' ')[0]
            Locator=sample_data['LOCATOR'][0]
            subinventory=sample_data['Subinventory Code'][0]
        if results['txn history'].shape[0]>0:
            pole=list(results['txn history']['POLE'])[0]
        else:
            pole='NONE'
    else:
        last_position='NONE'
        future_key='NONE'
        future_value,since=('NONE','NONE')
        Locator='NONE'
        subinventory='NONE'
        pole=''
    summary_all=pd.DataFrame()
    L=['Input_Key','Key_value','Warning','Serial Number','Part Status','Customer Country','Customer POLE','Process_owner','Number of Warehouse','Number of returns','Number of Surplus','Number of Pole','Repair QTY', 'Last repair PO', 'Last Failure Code','Last repair action','Last components consumption Number','Last repair Center Name','Last repair Center Code','Last repair date', 'FE comments (LAST REPAIR)', 'Supplier Comments (LAST REPAIR)', 'Analyst Comments (LAST REPAIR)', 'Last Localisation', 'Future Key search', 'Future value search', 'Since', 'Locator','Subinventory','POLE']
    values=[param,value,multiple_parts,SN_number,status,customer_country_code,customer_pole,process,nwhse,nreturns,nsurplus,npole,repairqty,repair_po, Failure_code,previous_repair_action,numbercomponents,previous_repair_center_name,previous_repair_center_code,previous_repair_date,previous_FE_comments,previous_supplier_comments,previous_analyst_comments,last_position,future_key,future_value,since,Locator, subinventory,pole]
    summary_all['Columns']=L
    summary_all['Values']= values
    results['Summary']=summary_all
    return results

## global function 
def tracing(param, value,direction):
    visited_values={'rma': [],'lpn': [],'job':[], 'so': [], 'sopo': [],'po':[] }
    keysummary={'po':'Repair PO', 'lpn':'LPN', 'rma': 'RMA', 'so': 'SO Number', 'job': 'Job Number'}
    real_name={'rma': 'RMA', 'lpn': 'LPN', 'po': 'PO Number', 'so': 'SO Number', 'job': 'Job Number', 'sopo':'SO PO Link'}
    multiple_warning='no'
    print('Connecting to database...')
    print('connected')
    multiple_parts_dict=dict()
    multiple_parts_dict['Search Key']=[]
    multiple_parts_dict['Search value']=[]
    if param=='repair':
        #try:
            #print(value)
        repair_log= get_repair_log('po',value)
        if repair_log.shape[0]>0:
            items=get_items(repair_log)
            #print(items[0])
            #extracting the so number associated to th repair po
            sonumber= get_so_from_po(value)
        else:
            print(f'PO {value} not found in Database')
            return -1
        try:
            date_dict= get_date(repair_log.reset_index(drop=True),'repairlog', 'PO1 PO Number', value,direction)
            # reset_index drop=true, enlever les anciens index
            print(date_dict)
        except:
            date_dict={}
        if len(items[0])==0:
            print(f'No item found under PO {value}')
            return -1
        else:
            param='po'
            traced=trace_one_step(param,value,'txn',items[0], date_dict)
    else:
        date_dict={}
        traced=trace_one_step(param,value,'txn', date_dict)
        date_dict=get_date(traced,'txn', param, value, direction)
        if 'repair_log' not in locals(): 
            repair_log = -1
    
    if traced.shape[0]==0 and type(repair_log)!=int:
        print('Getting data from repair log ....')
        repairdict=get_params(repair_log,repair=True)
        if 'sonumber' in locals():
            repairdict['so']=[sonumber]+[e for e in repairdict['so'] if e != sonumber]
        for s in ['so', 'job']:
            if len(repairdict[s])!=0:
                param=s
                values_extracted=list(pd.unique(repairdict[s]+[j for e in repairdict[s] for j in str(e).replace(';',',').split(',')]))
                #print(values_extracted)
                v_search=0
                traced_extract=pd.DataFrame()
                while traced_extract.shape[0]==0 and v_search<len(values_extracted):
                    value=values_extracted[v_search]
                    traced_extract=trace_one_step(param,value,'txn',items[0], date_dict)
                    v_search+=1
                traced=traced_extract
            if traced.shape[0]!=0:
                break 
        if traced.shape[0]==0:
            afficher_data=repair_log[['PO1 PO Number','Part Arrival Date','Part Repaired Date']].reset_index(drop=True)
            afficher_data.columns=['Repair PO','Part Arrival Date','Part Repaired Date']
            print('No transaction found for this search... \nPlease check if you get a valid Repair PO... \nOther repair POs associated to the same SN:\n', afficher_data)
            return -1
    elif traced.shape[0]==0 and type(repair_log)==int:
        return -1
    
    dv=get_params(traced.astype('str'))
    if type(repair_log)==int:
        repair_log=init_repairlog_search(traced)
        if type(repair_log)!=int:
            date_dict=update_dates(param,value,traced,repair_log,direction,date_dict)
            repairdict=get_params(repair_log,repair=True)
            dv=update_values(dv,repairdict)
        
    visited_values[param]+=[value]
    dictparams={'LPN':  get_keys(traced,'LPN')[0], 'RMA':  get_keys(traced,'RMA')[0], 'SO Number':  get_keys(traced,'SO Number')[0], 'SO PO Link':  get_keys(traced,'SO PO Link')[0]}
    Stop= False
    j=0
    traced['step']=[-1 for _ in range(traced.shape[0])]
    items=get_items(traced)
    if len(items[2])>1:
        items,traced=multiple_items(traced,items)
    laststatus=(' ',' ',' ')
    depart_date=extract_date(transform_todate(str(list(traced[traced['step']==-1]['Transaction Day'])[0])))
    while Stop == False:
        print(date_dict)
        for k in tqdm(dv.keys()):
            i=0
            n=len(dv[k])
            while i<n:
                if dv[k][i] not in visited_values[k]:
                    t=trace_one_step(k,dv[k][i],'txn',items[0],date_dict)
                    t,dictparams,traced,visited_values,laststatus= verify(dictparams,t,traced,visited_values,items[2],k,dv[k][i],laststatus,job=(k=='job'))
                    t['step']=[j for _ in range(t.shape[0])]
                    #listtxntype=[e.upper() for e in t['Transaction Sub-Type']]
                    #for e in listtxntype:
                    #   if 'RDE REPAIR' in e or 'GPO REPAIR' in e:
                    #        break
                    if type(repair_log)==int:
                        repair_log=init_repairlog_search(t)
                        if type(repair_log)!=int:
                            repairdict=get_params(repair_log,repair=True)
                            dv=update_values(dv,repairdict)
                    traced=add_rows(traced,t)
                    date_dict=update_dates_reduced(traced,direction,date_dict,depart_date)
                    items=get_items(traced)
                    nd=get_params(t.astype('str'))
                    dv=update_values(dv,nd)
                    if  multipleparts(t)== True:
                        visited_values=update_values(visited_values,{'rma': [],'lpn': [],'job':nd['job'], 'so': nd['so'], 'sopo': nd['sopo'],'po':nd['po'] })
                        multiple_warning='yes'
                        multiple_parts_dict['Search Key']+=[real_name[k]]
                        multiple_parts_dict['Search value']+=[dv[k][i]]
                    del t, nd
                    n=len(dv[k])
                i+=1
            Stop= endtracing(visited_values,dv)
        j+=1
    traced=traced.sort_index(ascending=False)
    traced=traced.drop_duplicates()
    print('Search finished\nData Processing...')
    results={'Summary':pd.DataFrame(), 'comments': pd.DataFrame(), 'repair history':pd.DataFrame(), 'repair details': pd.DataFrame(), 'FE Returns summary': pd.DataFrame()}
    results['txn history']=traced
#collecting components information
    if type(repair_log)!= int:
        repos=list(repair_log['PO1 PO Number'].unique())
        print('\nCollecting components data...')
        co_logs=pd.DataFrame()
        #for e in repos:
         #   co_log=get_component_log(e)
          #  co_logs=co_logs.append(co_log)
        results['repair history']=repair_log.sort_values(by='Part Repaired Date').reset_index(drop=True)
        results['repair details']= co_logs.reset_index(drop=True)
    else:
        results['repair history']= pd.DataFrame()
        results['repair details']= pd.DataFrame()
    results['FE Returns summary']=FE_analysis(traced)
    results=data_processing(results)
    if multiple_warning=='yes':
        results['warning']=pd.DataFrame(multiple_parts_dict)
    else:
        results['warning']=pd.DataFrame()
    results=summary_page(results,direction,keysummary[param],value)
    return results
########################################## Saving folder #######################################################

def get_path_file():
    root=tk.Tk()
    root.withdraw()
    root.attributes('-topmost',True)
    file = filedialog.askopenfile(mode ='r', title='Please select a File', filetypes =[('Excel Files', '*.xlsx'),('Csv Files','*.csv')])
    return file

def get_path_folder():
    root=tk.Tk()
    root.withdraw()
    root.attributes('-topmost',True)
    file_path= filedialog.askdirectory()
    return file_path

def save_file(datas,param,value,filep,action):
    if action=='y':
        with pd.ExcelWriter(filep+'/tracing {}_{}.xlsx'.format(param,value)) as writer:
            for e in datas.keys():
                try:
                    datas[e].to_excel(writer,sheet_name=e, index=False)
                except:# s'il marche pas, transformer à UTF-8 et réessayer
                    datas[e]=datas[e].applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
                    #applymap: apply a function to dataframe elementwise
                    datas[e].to_excel(writer,sheet_name=e, index=False)
        print('saved !!')

########################################## Utility tools #######################################################


#include this part in retrace multiple en bas
def multipletracing(values,param,direction,savefolder):
        results=pd.DataFrame()
        results['Columns']=['Input_Key','Key_value','Warning','Serial Number','Part Status','Customer Country','Customer POLE','Process_owner','Number of Warehouse','Number of returns','Number of Surplus','Number of Pole','Repair QTY', 'Last repair PO', 'Last Failure Code','Last repair action','Last components consumption Number','Last repair Center Name','Last repair Center Code','Last repair date', 'FE comments (LAST REPAIR)', 'Supplier Comments (LAST REPAIR)', 'Analyst Comments (LAST REPAIR)', 'Last Localisation', 'Future Key search', 'Future value search', 'Since', 'Locator','Subinventory','POLE']
        for value in values:
            print(f'------------------------Multiple tracing mode {param}: {value} ----------------------------')
            print('Search in progress...')
            start_time = time.time()
            # timeout_sec = 60
            # signal.signal(signal.SIGINT, timeout)
            # signal.alarm(timeout_sec)
            # try:
            func = timeout(timeout=60)(tracing)
            try:
                historydata=func(param,value,direction)
                print('data selected')
            except:
                historydata=-1
                print(f'ERROR for parameter {param} : {value}')
            # finally:
                # signal.alarm(0)
                # historydata = -1
                # print('fonction time out')
            if historydata==-1:
                L=[param,value]+['Unfound' for _ in range(len(results['Columns'])-2)]
            else:
                save_file(historydata,param,value,savefolder,'y')
                L=historydata['Summary']['Values']
            print(f"--- {round(time.time() - start_time,2)} seconds ---")
            results[value]=L
        results=results.T #transposer le résultat
        results.columns=results.iloc[0]
        results=results.drop(results.index[0])
        return results.reset_index(drop=True)

def get_direction():
    close='n'
    while close=='n':
        print('------------------------------------------------------------------')
        txt='''choose Search direction:
        1.'Forward'
        2.'Backward'
        3.'Any'
        4. Quit
        '''
        direction=input(txt)
        directionlist={'1':'F','2':'B', '3':'A', '4':'Quit'}
        while direction not in directionlist.keys():
            print('ERROR.... \nPlease choose number from 1 to 4 .... ')
            direction=input(txt)
        return directionlist[direction]

def program_multiple_parts():
    #read the file
    close='n'
    while close =='n':
        print(f'------------------------Multiple tracing mode ----------------------------')
        txt='''choose the parameter to use:
        1.'RMA'
        2.'LPN'
        3.'Repair PO'
        4.'Serial Number'
        5.'Components'
        6.'Quit'
        '''
        parameter=input(txt)
        paramlist={'1':'rma','2':'lpn', '3':'repair','4':'sr','5':'Components' ,'6':'Quit'}
        while parameter not in paramlist.keys():
            print('ERROR.... \nPlease choose number from 1 to 6 .... ')
            parameter=input(txt)
        if parameter=='6':
            break
        if parameter not in ['4','5']:
            direction=get_direction()
            if direction =='Quit':
                break
        else:
            direction=''
        print('---------------------Please Upload Your File------------------------')
        fileread=get_path_file()
        while fileread is None:
            print('ERROR')
            fileread=get_path_file()
        print('-------------------------File Uploaded :)----------------------------')
        print('--------------------Please Select directory folder-------------------')
        directory_folder=get_path_folder()
        try:
            values=pd.read_excel(fileread.name, header=None, dtype=str)
        except:
            values=pd.read_csv(fileread.name,header=None, dtype =str)
        values=list(values[values.columns[0]])
        if parameter == '4':
            summary_result=trace_one_step(paramlist[parameter],values,'ert')[['ITEM', 'Item Description', 'Serial Number','Vendor Name','Part Repaired Date','PO1 PO Number','Job Number','SO1 SO Number','PO1 PO Header Description','Failure Code','FE Comments','Supplier Comments','Analyst Comments','Repair Action']]
            print('data selected...')
        elif parameter =='5':
            print('Looking for repairs')
            summary_result=pd.DataFrame()
            for v in tqdm(values):
                summary_result=summary_result.append(get_component_log(int(v)))
            print('data selected...')
        else:
            summary_result=multipletracing(values,paramlist[parameter],direction,directory_folder)
        if direction!='':
            summary_result.to_excel(directory_folder+f'/{paramlist[parameter]}_{direction}_'+'all_parts.xlsx', index= False)
        else:
            summary_result.to_excel(directory_folder+f'/{paramlist[parameter]}_'+'all_parts.xlsx', index= False)
        close= input('''Close y/n?
            ''')

def program_one_part():
    close='n'
    direction = 'not define'#xin
    while close =='n':
        #print('Connecting to database...')
        #conn=db.open_conn_FBI()
        #print('connected')
        print('----------------------------One Part tracing mode-----------------------')
        txt='''choose the parameter to use:
        1.'RMA'
        2.'LPN'
        3.'Repair PO'
        4.'Serial Number'
        5.'Components'
        6. Quit
        '''
        parameter=input(txt)
        paramlist={'1':'rma','2':'lpn', '3':'repair', '4':'sr','5':'components','6':'Quit'}
        while parameter not in paramlist.keys():
            print('ERROR.... \nPlease choose number from 1 to 6 .... ')
            parameter=input(txt)
        if parameter=='6':
            break
        if parameter in [str(i) for i in range(1,5) ]:
            valuep= input('''Please register the reference
            ''')
        else:
            valuep= input('''Please register repair po
            ''')
        if parameter=='4': #to get repair log
            print('------------------------------------------------------------------')
            print('Search in progress...')
            start_time = time.time()
            historydata= trace_one_step(paramlist[parameter],valuep,'ert')
            historydata={'repair log':historydata[['ITEM', 'Item Description', 'Serial Number','Vendor Name','Part Repaired Date','PO1 PO Number','Job Number','SO1 SO Number','PO1 PO Header Description','Failure Code','FE Comments','Supplier Comments','Analyst Comments','Repair Action']]}
        elif parameter =='5':
            print('Looking for repair po...')
            start_time = time.time()
            historydata={'Component log':get_component_log(int(valuep))}
        else:
            direction=get_direction()
            if direction =='Quit':
                break
            print('------------------------------------------------------------------')
            print('Search in progress...')
            start_time = time.time()
            historydata= tracing(paramlist[parameter],valuep,direction)
        if type(historydata)==int:
            close='n'
            print(f"--- {round(time.time() - start_time,2)} seconds ---")
        else:
            print('data selected')
            print(f"--- {round(time.time() - start_time,2)} seconds ---")
            filep=get_path_folder()
            save_file(historydata,paramlist[parameter],str(valuep)+'_'+direction,filep,'y')
            close= input('''Close y/n?
            ''')

def FOA():
    close='n'
    while close=='n':
        jours= input('Choose Number of days to look for FOA\n')
        print('----------------------------FOA Follow up mode-------------------------')
        txt='''choose the Part:
        1. XFD_XFA
        2. TSUI
        3. Coils
        4. Others
        5. Quit
        '''
        item=input(txt)
        paramlist={'1':'XFD','2':'TSUI', '3':'Coils', '4':'Others', '5':'Quit'}
        while item not in paramlist.keys():
            print('ERROR.... \nPlease choose number from 1 to 5 .... ')
            item=input(txt)
        if item=='5':
            break
        print('Search in progress...')
        start= FOA_followup(jours, paramlist[item])
        values=[e for e in start.RMA]
        if len(values)==0:
            print(f'Congrats no FOA found for the last {jours} days')
            close= input('''close y/n?
                ''')
        else:
            print(f'{len(values)} FOA declared during the last {jours} days')
            print('--------------------Please Select directory folder-------------------')
            directory_folder=get_path_folder()
            if directory_folder=='':
                break
            print(values)
            tracing_results=multipletracing(values, 'rma','A',directory_folder )
            for k in tracing_results.columns:
                if k not in ['Input_Key','Key_value']:
                    start[k]=tracing_results[k]
            start.to_excel(directory_folder+f'/{paramlist[item]} FOA Suivi.xlsx', index=False)
            print('Saved !!')
            close= 'y'
    

def menu():
    menu='y'
    while menu =='y':
        print('------------------------------------------------------------------')
        txt='''choose search mode:
        1. One part
        2. Multiple parts
        3. FOA follow up
        4. Quit
        '''
        mode=int(input(txt))
        while mode not in range(1,5):
            print('ERROR')
            mode=int(input(txt))

        if mode==1:
            program_one_part()
        elif mode== 2:
            program_multiple_parts()
        elif mode == 3:
            FOA()
        elif mode == 4:
            print('----------Good Bye---------')
            menu='n'


if __name__=='__main__':
    os.chdir('c:/Users/223102584/Box/Doc_Xin/Sujet_Principale/Analyse_RC_5443517-R/')
    table = f'''SELECT [DW_TXN_KEY] [indices],[Transaction Day],[Transaction Type],[Transaction Description] ,
    [Transaction Sub-Type] ,[Transaction Category] ,[Txn Quantity],[Inventory Org],[Subinventory Code],[ITEM],
    [Item Description],[Job Number],[SO Number],[PO Number],[LPN],[FE Txn Group] [RMA],[SO PO Link],[Job Open Date],
    [PO Type],[FE SSO], [FE ORG],[System ID],[System Install Date],[Customer Number],[Customer Name],[Customer Country Code],
    [Customer Pole],[POLE],[LOCATOR],[Transfer Locator],[Vendor Name],[Vendor Site Code], [SO Type], [Reason Name],
    [PO Header Description],[Failure Description],[Waybill], [Freight Carrier], [SO PO Link Type],[Obso Pivot Code]
    FROM [dbo].[vwAllTxns_v3] 
    Where [ITEM] = '2416759-R' '''
    table2 = f'''SELECT [DW_ERT_KEY] [indices],[ITEM], [Item Description], [Current Item Type], [Current First Put-Away Date (FPAD)],
    [Current Item Status], [Vendor GSL Number], [Vendor Name],[Vendor Site Code], [PO1 PO Number], [PO1 Need By Date], 
    [PO1 PO Header Description], [PO1 Approved Date], [PO1 Creation Date],  [PO1 Shipment Last Update Date], [PO1 Receipt Date], 
    [PO1 PO Type], [PO1 PO Type DFF], [PO1 Authorization Status], [PO2 PO Number], [SO1 Shipping Warehouse], [FE Number], [FE SSO], 
    [Failure Code],[Part Failure Analysis],[FE Comments],[Supplier Comments],[Analyst Comments], [Repair Status], [Repair Action], 
    [Job Number], [SO1 SO Number], [SO1 SO Status], [SO1 Waybill], [SO1 Order Date], [SO1 Ship Date], [SO2 SO Number], 
    [Serial Number], [Part Arrival Date], [Part Repaired Date]
    FROM [dbo].[vwErt] 
    Where [ITEM] = '2416759-R' '''
    TXN = run_query_sqldb(f'{table}',open_conn_FBI())
    ERT = run_query_sqldb(f'{table2}',open_conn_FBI())
    # sdata = ERT[ERT['Serial Number']=='10990497']
    # print(sdata)
    menu()