import torch
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial import distance_matrix

import pickle5 as pickle

num_class = 10 #1000
pretrained_model_path = './pretrained_models/VGG_16/vgg_16_bn.pt'
complete_pickle_filepath = './VGG_10D_Best_K_in_layers.pickle'

# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(pretrained_model_path, map_location=device)

"""
Just a util.
a more advanced class of dictionaries with capability of adding stuff to it on the go.

"""


class Sor_dictionary(dict):
    def __init__(self):
        self = dict()

    # Function to add key:value
    def add(self, key, value):
        self[key] = value

    # Function to delete key
    def delete(self, key):
        del self[key]


"""
Super important function. recieves pandas dataframe of datapoints in 2D space (avg, std), and the x divider and y divider.
output: reveals each filter is in which region.

four_regions:  0 -> 1 ^ 2 <- 3
points[0] == x axis
points[1] == y axis

and I also have a list of 4 list for filter numbers according to their region.

"""


def region_maker(pd_datapoints_2d, x_divider, y_divider):
    regions = [[] for x in range(int(4))]

    filter_num_in_region = [[] for x in range(int(4))]
    for i, points in enumerate(pd_datapoints_2d.values):

        if points[0] <= x_divider and points[1] <= y_divider:
            regions[0].append(points)
            filter_num_in_region[0].append(i)

        elif points[0] > x_divider and points[1] < y_divider:
            regions[1].append(points)
            filter_num_in_region[1].append(i)

        elif points[0] > x_divider and points[1] > y_divider:
            regions[2].append(points)
            filter_num_in_region[2].append(i)

        elif points[0] < x_divider and points[1] > y_divider:
            regions[3].append(points)
            filter_num_in_region[3].append(i)
    return regions, filter_num_in_region


"""

Gives n(n == 10 or n == 1000) features per filter in a dictionary

"""


def n_feature_per_filter():
    dict_of_n_features = Sor_dictionary()
    for num_of_filters in range(len(dict_of_ranks[0])):
        dummy_list = []
        for num_of_classes in range(num_class):
            #         print(dict_of_ranks[y][i])
            #             dummy_list.append(dict_of_ranks[y][i])
            dummy_list.append(dict_of_ranks[num_of_classes][num_of_filters])
        #     list_of_avg_std.append(np.average(dummy_list)) # 1st, avg
        #     list_of_avg_std.append(np.std(dummy_list)) # 2nd, std
        #         list_of_avg_std.append(pd.Series(dummy_list).mad()) # 3rd, Pandas Mad
        #         list_of_avg_std.append(Sor_MAD(dummy_list)) # 4th item, MY Mad (mad without abs)
        #         print(list_of_avg_std.append(Sor_MAD(dummy_list)))
        dict_of_n_features.add(num_of_filters, np.array(dummy_list))

    return dict_of_n_features


# Path
path_to_ranks = "./"
prefix = "rank_conv/" + 'vgg_16_bn' + "/rank_conv"
subfix = ".npy"

dict_of_filters_to_keep = Sor_dictionary()

# resnet_50   ->  range(1, 54)
# VGG         ->  range(1, 13)
# resnet_56   ->  range(1, 56)


for LAYER_NUMBER in range(1, 13):
    # Dict of ranks gives me the avg rank of each filter.
    dict_of_ranks = Sor_dictionary()
    for i in range(num_class):
        dict_of_ranks.add(i, np.load(path_to_ranks + prefix + str(LAYER_NUMBER) +'_Class_'+str(i)+'.npy') )

    dict_of_n_features = n_feature_per_filter()


    # Making Dict of n features to DataFrame (n == 10 or n == 1000)
    # Then turn it into pandas dataframe
    list_of_features = np.zeros((len(dict_of_n_features), num_class))
    for i in range(len(dict_of_n_features)):
        for x in range(num_class):
            list_of_features[i][x] = dict_of_n_features[i][x]
    list_of_features = np.array(list_of_features)
    pd_datapoints_nd = pd.DataFrame(list_of_features) # (n == 10 or n == 1000)

    # Normalizing Data `Column wise`
    # Using Sklearn MinMaxSacaler method
    scaler = preprocessing.MinMaxScaler()
    for i in range(len(pd_datapoints_nd.columns)):
        pd_datapoints_nd[i] = scaler.fit_transform(pd_datapoints_nd[i].values.reshape(-1,1))

    # Use Pickle file for best K.
    # It was made in google collab. Set the Random State of KMeans to 6 so the results do not vary a lot.
    # The Following Decides the number of K's and I am reading them from a Pickle file
    with open(complete_pickle_filepath, 'rb') as handle:
        Best_K_in_layers = pickle.load(handle)

    # N-D Kmeans with 6 as random state, as it should be
    Km = KMeans(n_clusters = Best_K_in_layers[LAYER_NUMBER], random_state=6, max_iter=10000).fit(pd_datapoints_nd)
    df_master = pd.DataFrame(pd_datapoints_nd.copy())
    df_master['data_index'] = pd_datapoints_nd.index.values
    df_master['cluster'] = Km.labels_


    # Making Distance Matrix to find n-D Medoids
    # (idea: CAN be replaced by max sil score in each cluster)
    list_of_medoids = []
    for i in range(Best_K_in_layers[LAYER_NUMBER]):
    #     temp_df = pd.DataFrame(cluster_map[cluster_map.cluster == i ]['data_index'])
        temp_df = pd.DataFrame(df_master[df_master.cluster == i ])
        df_distance_matrix = pd.DataFrame(distance_matrix(temp_df.values[:, :num_class],
                                                          temp_df.values[:, :num_class]),
                                          index = temp_df.index, columns = temp_df.index)
        list_of_medoids.append(np.array(df_distance_matrix.index)[np.argmin(df_distance_matrix.sum(axis=0))])


    # a flag to identify the medoids
    df_master['medoid'] = 0
    df_master.loc[list_of_medoids, 'medoid'] = 1

    # Switching to 2-D Space to find Star-Point and Visualize
    pd_datapoints_2d = pd.DataFrame(columns=['avgs', 'stds'])
    for i in range(len(pd_datapoints_nd.index)):
        pd_datapoints_2d.loc[i] = pd_datapoints_nd.loc[i].mean(), pd_datapoints_nd.loc[i].std()

    # Normalize the 2D Space
    pd_datapoints_2d['avgs'] = scaler.fit_transform(pd_datapoints_2d['avgs'].values.reshape(-1,1)).reshape(pd_datapoints_2d.shape[0],)
    pd_datapoints_2d['stds'] = scaler.fit_transform(pd_datapoints_2d['stds'].values.reshape(-1,1)).reshape(pd_datapoints_2d.shape[0],)

    # adding more info to df_master
    df_master['avgs'] = pd_datapoints_2d['avgs']
    df_master['stds'] = pd_datapoints_2d['stds']

    new_x_divider = df_master['avgs'].mean()
    new_y_divider = df_master['stds'].mean()

    # making a list of filters with below AVG silhouette sample values
    # and adding those vals as data to master df
    silhouette_sample_values = silhouette_samples(df_master.iloc[:,:num_class], df_master.cluster)
    df_master['sil_samp'] = silhouette_sample_values

    # adding the mean silhouette of each cluster, as an info to each filter
    # designing cluster based "cluster mean sil"
    df_master['cluster_mean_sil'] = None


    # Critical part of the code, assign regions to each filter
    regions, filter_no_in_each_region = region_maker(pd_datapoints_2d, new_x_divider, new_y_divider)

    # Critical
    df_master['region'] = None
    for i in range(4):
        df_master.iloc[filter_no_in_each_region[i],[-1]] = i

    # To find medoids in region 1
    medoids_in_region_1 = np.array(df_master.query("medoid == 1 & region == 1")['cluster'])


    # Getting the cluster assossiated with medoids in region one.
    # In the new temporary table, we will get the mean on all sil samps,
    # Then will assign them to their respective datapoints in the master
    for clus in medoids_in_region_1:
        temp_df = df_master.query('cluster == @clus & region == 1')

        temp_mean = temp_df['sil_samp'].mean()

        df_master.loc[temp_df.index.values, ['cluster_mean_sil']] = temp_mean



    # Comparing each datapoint sil samp to their clusters mean sil
    # If a datapoint has None as its cluster mean, it means that datapoint is not in region 1, which we will not keep it
    # TypeError is for comparinf None to float, which is not important
    df_master['below_avg'] = None
    for i in df_master['data_index']:
        try:
            df_master.loc[i, 'below_avg'] = df_master.loc[i, 'sil_samp'] <= df_master.loc[i, 'cluster_mean_sil']
        except TypeError:
            pass

    final_staying_filters = []

    staying_datapoints = df_master.query('below_avg == 1')['data_index'].tolist()
    staying_medoids = medoids_in_region_1.tolist()
    for elm in staying_datapoints:
        final_staying_filters.append(elm)

    for elm in staying_medoids:
        final_staying_filters.append(elm)

    final_staying_filters.sort()

    final_staying_filters = np.unique(final_staying_filters)
    dict_of_filters_to_keep.add( LAYER_NUMBER, final_staying_filters )

    
    
# Write to a pickle
with open('POST_CLEAN_JUST_R1_MEDOIDandBELOWAVG.pickle', 'wb') as handle:
    pickle.dump(dict_of_filters_to_keep, handle, protocol=pickle.HIGHEST_PROTOCOL)    