import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def prepareData(file_name):
    '''
    Input: nom du fichier sorti de Data_collection
    Output: data sans labels et labels
    '''
    data_labeled = pd.read_excel(file_name)
    # data_labeled = data_labeled[(data_labeled['Repaire_qty']>0) & (data_labeled['Repaire_qty']<10)]
    # dropIndex = data_labeled[(data_labeled['Repaire_qty']==0) | (data_labeled['SystemAge']<0)].index
    # data_labeled.drop(index=dropIndex, inplace = True)
    # cols_drop = ['adjusted_earlylife_failure','elf_days']
    cols_drop = ['adjusted_earlylife_failure']
    for i in data_labeled.columns.tolist():
        if 'Unnamed' in str(i):
            cols_drop.append(i)
    data = data_labeled.drop(columns = cols_drop)
    data = data.loc[:, data.any()] # enlever les colonnes tous 0
    labels = data_labeled['adjusted_earlylife_failure']
    FOA_label = int(input('Tell us FOA label:'))
    nb_FOA = len(data_labeled[data_labeled['adjusted_earlylife_failure']==FOA_label].index)
    print('We have ', len(data_labeled.index), 'parts', nb_FOA, ' FOAs among them.')
    return data, labels, data_labeled, FOA_label


def data_sampling(data_labeled, FOA_label):
    sample_NF = int(input('How many NF?'))
    sample_FOA = int(input('How many FOA?'))
    k1=data_labeled[data_labeled['adjusted_earlylife_failure']!=FOA_label].sample(sample_NF, random_state = 42)
    k2=data_labeled[data_labeled['adjusted_earlylife_failure']==FOA_label].sample(sample_FOA, random_state = 42)
    data_labeled_sample = pd.concat([k1,k2])
    labels = data_labeled_sample['adjusted_earlylife_failure']
    data_sample = data_labeled_sample.drop(columns = ['Unnamed: 0', 'adjusted_earlylife_failure','elf_days'])

    return data_sample, labels



def feature_selec(data, labels):
    print('-'*10, 'feature selection', '-'*10)
    print('Adding Feature Selecter...')
    selector = SelectKBest(k=10)
    X_selected = selector.fit_transform(data, labels)
    pVal =selector.pvalues_
    print(pVal)
    threshold = float(input('please set your threshold for feature selection(generally 0.05):'))
    nb_features = np.count_nonzero(pVal<threshold)
    selector = SelectKBest(k=nb_features)
    X_selected = selector.fit_transform(data, labels)
    print('We\'ve selected these features:', selector.get_feature_names_out())
    print('-'*10, 'end of selection', '-'*10)
    return selector, X_selected


def cluster_support(X_selected):
    # elbow method
    kmeans = KMeans()
    visualizer = KElbowVisualizer(kmeans, k=(2,10))
    visualizer.fit(X_selected)
    visualizer.show()

    return 0


def kmeans(selector, X_selected, labels):
    print('-'*10, 'K-means Clustering', '-'*10)
    mm = MinMaxScaler() # ne change pas la distribution des données
    X_selected = mm.fit_transform(X_selected)
    nb_centers = 3 # nb centers défaut
    suggestion = input('Do you want our suggestion about nb of cluster centers?')
    if suggestion == 'y':
        cluster_support(X_selected)
    
    nb_centers = int(input('Choose number of cluster centers: '))
    km_res = KMeans(n_clusters=nb_centers).fit(X_selected)
    compare_labels = pd.DataFrame()
    compare_labels['kmeans']=km_res.labels_
    compare_labels['failures'] = labels.to_list()
    print('Distribution of true labels in clusters:')
    print(compare_labels.groupby('kmeans').value_counts())
    vrai_center = mm.inverse_transform(km_res.cluster_centers_)
    centers=pd.DataFrame(data=vrai_center)
    centers.columns = selector.get_feature_names_out().tolist()
    res_name = str(input('Input your result name: '))
    try:
        os.makedirs("./log")
    except FileExistsError:
        # directory already exists
        pass
    centers.to_excel(f'./log/{res_name}.xlsx')
    print('-'*10, 'Result saved!', '-'*10)

    return km_res


def DBSCAN_cluster(selector, X_selected, labels):
    print('-'*10, 'DBSCAN Clustering', '-'*10)
    mm = MinMaxScaler() # ne change pas la distribution des données
    X_selected = mm.fit_transform(X_selected)
    db_res = DBSCAN().fit(X_selected)
    compare_labels = pd.DataFrame()
    compare_labels['dbscan']=db_res.labels_
    compare_labels['failures'] = labels.to_list()
    print('Distribution of true labels in clusters:')
    print(compare_labels.groupby('dbscan').value_counts())

    X_vrai = mm.inverse_transform(X_selected)
    X_selected_df=pd.DataFrame(data=X_vrai)
    X_selected_df.columns = selector.get_feature_names_out().tolist()
    X_selected_df['DBSCAN_labels'] = db_res.labels_

    grouped_df = X_selected_df.groupby('DBSCAN_labels').mean()
    res_name = str(input('Input your result name: '))
    try:
        os.makedirs("./log")
    except FileExistsError:
        # directory already exists
        pass
    grouped_df.to_excel(f'./log/DBSCAN_{res_name}.xlsx')

    print('-'*10, 'Result saved!', '-'*10)
    return db_res


def dimReduce(X_selected, labels, km_res):
    pca = PCA(n_components=3)
    mm = MinMaxScaler() # ne change pas la distribution des données
    X_selected = mm.fit_transform(X_selected)
    data_pca = pca.fit_transform(X_selected)
    ex_ratio = pca.explained_variance_ratio_.sum()
    print('Explained ratio of PCA is', "{:.2f}".format(ex_ratio*100), '%')
    # visualisation PCA
    scatter_vis(data_pca, labels, km_res)
    return 0



def scatter_vis(data_pca, labels, km_res):
    label_colors = ['green', 'red', 'blue','yellow','black']
    label_alpha = [0.5, 0.5, 0.5, 0.5, 0.5]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(data_pca[mask, 0], data_pca[mask, 1], data_pca[mask, 2], 
                c=label_colors[label%5], alpha=label_alpha[label%5], label=str(label))

    cluster_centers = km_res.cluster_centers_
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], 
            marker='x', color='black', s=100, label='Cluster Center')


    ax.legend(title='Labels')

    ax.set_title('K-means Clustering')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    plt.show()

    return 0


def visualizeImportance(data, clf):
    cols = data.columns
    importances = clf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=cols)

    fig= plt.figure(figsize=(6,6))
    ax = plt.subplot()
    forest_importances.plot.bar(ax=ax) # yerr=std,
    ax.set_title("Feature importances")
    ax.set_ylabel("Importance")
    fig.tight_layout()

    plt.show()
    return 0


def randomForest(labels, data):
    clf = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('Accuracy of prediction: ','{:.2f}'.format(accuracy*100), '%')
    vis_featureImportance = input('Would you like to know the feature importance?(y/n):')
    if vis_featureImportance == 'y':
        visualizeImportance(data, clf)
    
    return 0


def distribPlot(data_labeled, bar_width):
    data_labeled = data_labeled.loc[:, data_labeled.any()]
    print('We have:', data_labeled.columns)
    Exit = 'n'
    while Exit == 'n':
        txt='''Choose the distribution plot mode:
        1. Distribution of part in FE experience
        2. Distribution of part in Repair Centers or Carriers
        3. Distribution of part in other column
        4. Exit
        '''
        mode=int(input(txt))
        while mode not in range(1,5):
            print('ERROR')
            mode=int(input(txt))

        # Different plot modes
        if mode == 1:
            data_labeled = data_labeled[['adjusted_earlylife_failure','FE_exp']]
            max_exp = data_labeled['FE_exp'].max()
            bins = [0, 730, 3650, 7300, max_exp]
            labels = ['<730', '730-3650', '3650-7300', '>7300']
            data_labeled['range'] = pd.cut(data_labeled['FE_exp'], bins=bins, labels=labels)
            color_mapping = {0: 'y', 1: 'r', 2: 'b', 3: 'y'}
            count = data_labeled.groupby(['range', 'adjusted_earlylife_failure']).size().unstack(fill_value=0)
            count.plot(kind='bar',stacked=True, color=color_mapping)
            plt.xticks(rotation = 0)
            plt.xlabel('FE_exp')
            plt.ylabel('Number of parts')
            plt.show()


        elif mode== 2:
            dist_cols = str(input('Please choose the columns you interessed: '))
            list_dist_cols = list(dist_cols.split('\', \''))
            label = ['adjusted_earlylife_failure']
            dist_cols2 = label+list_dist_cols
            data = data_labeled[dist_cols2]
            data = data.groupby(['adjusted_earlylife_failure']).sum()
            print(data)
            x_cor = np.arange(len(list_dist_cols)) +0.3
            colors = ['y','r','b','y']
            bottom = np.zeros(len(data.columns))
            for row in range(len(data.index)):
                plt.bar(x_cor, data.iloc[row].values, bar_width, label 
                        =np.arange(len(data.index))[row], color = colors[row],bottom=bottom )
                bottom = bottom+data.iloc[row].values
                
                
            plt.xticks(range(len(list_dist_cols)),list_dist_cols, rotation = 30)
            plt.xlabel('Repair Centers')
            plt.ylabel('Repair Quantity')
            plt.legend()
            plt.show()

        elif mode == 3:
            dist_col = str(input('Please choose one column: '))
            data = data_labeled[['adjusted_earlylife_failure',dist_col]]
            # max_exp = data['FE_exp'].max()
            # bins = [0, 730, 3650, 7300, max_exp]
            # labels = ['<730', '730-3650', '3650-7300', '>7300']
            try:
                data['range'] = pd.qcut(data[dist_col], 8)
            except:
                try:
                    data['range'] = pd.qcut(data[dist_col], 5)
                except:
                    data['range'] = pd.cut(data[dist_col], 5)
            color_mapping = {0: 'y', 1: 'r', 2: 'b', 3: 'y'}
            count = data.groupby(['range', 'adjusted_earlylife_failure']).size().unstack(fill_value=0)
            count.plot(kind='bar', color=color_mapping, stacked=True)
            plt.xticks(rotation = 0)
            plt.xlabel(f'{dist_col}')
            plt.ylabel('Number of parts')
            plt.show()

        elif mode == 4:
            print('----------back to principle menu---------')
            Exit='y'

        return 'Done'


def main():
    Exit = 'n'
    # PREPARE DATA
    file_name = input('Please give me your file name:')
    try:
        data, labels, data_labeled, FOA_label = prepareData(file_name)
    except:
        print('Please check your file name.')
        
    sampling = input('Do you want sample the data?(y/n)')
    if sampling=='y':
        data, labels = data_sampling(data_labeled, FOA_label)

    selector, X_selected = feature_selec(data, labels)
    while Exit == 'n':
        txt='''Choose your analyse mode:
        1. K-means clustering
        2. DBSCAN clustering
        3. Prediction(Random Forest)
        4. Distribution of data
        5. Exit
        '''
        mode=int(input(txt))
        while mode not in range(1,6):
            print('ERROR')
            mode=int(input(txt))

        # Different modes
        if mode == 1:
            km_res = kmeans(selector, X_selected, labels)
            next = input('Would you like to do some PCA and visualize the result?(y/n)')
            while next not in ['y', 'n']:
                print('Error')
                next = input('Would you like to do some PCA and visualize the result?(y/n)')
            
            if next == 'y':
                dimReduce(X_selected, labels, km_res)
            else:
                pass

        elif mode== 2:
            DBSCAN_cluster(selector, X_selected, labels)
            
        elif mode == 3:
            randomForest(labels, data)
            
        elif mode == 4:
            bar_width = 0.2
            distribPlot(data_labeled, bar_width)

        elif mode == 5:
            print('-'*10, 'Good Bye', '-'*10)
            Exit='y'



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.chdir('c:/Users/223102584/Box/Doc_Xin/Sujet_Principale/')
    main()
    

#./Amplifier - RF/Avant_cluster2023-07-03.xlsx
#./Ref_2352573/Avant_cluster2023-07-03.xlsx
