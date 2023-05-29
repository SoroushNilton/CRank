import torch
import numpy as np
import pandas as pd
from random import sample
# from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from scipy.spatial import distance_matrix
import matplotlib.cm as cm

import pickle5 as pickle

"""

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
get_all_scores will get all filters and calculate scores of all classes within each filter

output is a dic with all filters inside a layer.
the key to each dictionary is a number representing the filter
and the values are n number of scores where n is equal to number of classes

"""


def get_all_scores():
    dict_of_all_scores = Sor_dictionary()
    for num_of_filters in range(len(dict_of_ranks[0])):
        dummy_list = []
        for num_of_classes in range(len(dict_of_ranks)):
            #         print(dict_of_ranks[num_of_classes][num_of_filters])
            dummy_list.append(dict_of_ranks[num_of_classes][num_of_filters])
        dict_of_all_scores.add(num_of_filters, dummy_list)
    return dict_of_all_scores


"""
Calculates the mean and standard deviation for all classes whithin one layer.
the input is the dict of ranks and the output is:
a dictionary with 'number of filter' in a layer as a key and their mean and std as the values

e.g:
0: [31.142029, 0.06933148],
1: [31.543125, 0.029491415],
.
.
15: [31.381565, 0.09870682]

"""


def calc_avg_std():
    dict_of_avg_std = Sor_dictionary()
    for num_of_filters in range(len(dict_of_ranks[0])):
        dummy_list = []
        list_of_avg_std = []
        for num_of_classes in range(10):
            #         print(dict_of_ranks[y][i])
            #             dummy_list.append(dict_of_ranks[y][i])
            dummy_list.append(dict_of_ranks[num_of_classes][num_of_filters])
        list_of_avg_std.append(np.average(dummy_list))  # 1st, avg
        list_of_avg_std.append(np.std(dummy_list))  # 2nd, std
        #         list_of_avg_std.append(pd.Series(dummy_list).mad()) # 3rd, Pandas Mad
        #         list_of_avg_std.append(Sor_MAD(dummy_list)) # 4th item, MY Mad (mad without abs)
        #         print(list_of_avg_std.append(Sor_MAD(dummy_list)))
        dict_of_avg_std.add(num_of_filters, list_of_avg_std)
    return dict_of_avg_std


#     print('=======================')


"""
Verify that two selected filtes does not have intersection

gets two arrays, checks if they have similarities or not.
"""


def verify_staying_filters_dont_intersect(staying_filters, final_id_of_filters_to_keep_in_regions_2_and_3):
    for fils in staying_filters:
        if fils in final_id_of_filters_to_keep_in_regions_2_and_3:
            print(fils)
            return True

    for fils in final_id_of_filters_to_keep_in_regions_2_and_3:
        if fils in staying_filters:
            print(fils)
            return True


"""
Finds index of cluster which is nearest to best point in dataset (1, 0)

"""


def min_dist(pd_medoids):
    target = (1, 0)
    target = np.array(target)
    dists = []
    for i in range(len(pd_medoids)):
        point_in_question = np.array(tuple(pd_medoids.values[i]))
        dist = np.linalg.norm(np.array(point_in_question) - np.array(target))
        dists.append(dist)
        result = np.where(dists == np.amin(dists))
    return int(result[0])


"""

gets the data point and return a np.array of filters that are in cluster nearest to point (1, 0)

"""


def make_list_of_pruning(data_points_2d, pd_medoids):
    list_of_pruning = []

    df = pd.DataFrame(data_points_2d, columns=['avg', 'std'])

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = df.index.values
    cluster_map['cluster'] = Km.labels_

    for index in cluster_map[cluster_map.cluster == min_dist(pd_medoids)]['data_index']:
        list_of_pruning.append(index)

    return np.array(list_of_pruning)


"""

Gets filters in cluster 4 and checks if those filters are in region 1 or not.

"""


def intersect_cluster_4_region_1(regions, cluster_4):
    regions[1] = np.array(regions[1])
    intersect_of_cluster_4_and_region_1 = []

    for index in cluster_4:
        if data_points_2d[index] in regions[1]:
            intersect_of_cluster_4_and_region_1.append(index)

    return intersect_of_cluster_4_and_region_1


"""

four_regions:  0 -> 1 ^ 2 <- 3
points[0] == x axis
points[1] == y axis

and I also have a list of 4 list for filter numbers according to their region.

"""


def region_maker(data_points_2d, x_divider, y_divider):
    regions = [[] for x in range(int(4))]

    filter_num_in_region = [[] for x in range(int(4))]
    for i, points in enumerate(data_points_2d):

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
    return regions


"""

Gives 10 features per filter in a dictionary

"""


def ten_feature_per_filter():
    dict_of_10_features = Sor_dictionary()
    for num_of_filters in range(len(dict_of_ranks[0])):
        dummy_list = []
        for num_of_classes in range(10):
            #         print(dict_of_ranks[y][i])
            #             dummy_list.append(dict_of_ranks[y][i])
            dummy_list.append(dict_of_ranks[num_of_classes][num_of_filters])
        #     list_of_avg_std.append(np.average(dummy_list)) # 1st, avg
        #     list_of_avg_std.append(np.std(dummy_list)) # 2nd, std
        #         list_of_avg_std.append(pd.Series(dummy_list).mad()) # 3rd, Pandas Mad
        #         list_of_avg_std.append(Sor_MAD(dummy_list)) # 4th item, MY Mad (mad without abs)
        #         print(list_of_avg_std.append(Sor_MAD(dummy_list)))
        dict_of_10_features.add(num_of_filters, np.array(dummy_list))

    return dict_of_10_features


dict_of_silhouette = Sor_dictionary()

# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./pretrained_models/ResNet56/resnet_56.pt', map_location=device)

# ## Resnet

path_to_ranks = "./"
prefix = "rank_conv/" + 'resnet_56' + "/rank_conv"
subfix = ".npy"

# ## VGG16

# path_to_ranks  = "../../../VGG16/"
# prefix = "rank_conv/ours/"+ 'vgg_16_bn' +"/rank_conv"
# subfix = ".npy"
for LAYER_NUMBER in range(1, 56):

    print('=========================================')
    print('In Layer Number ', LAYER_NUMBER)

    dict_of_ranks = Sor_dictionary()
    for i in range(10):
        dict_of_ranks.add(i, np.load(path_to_ranks + prefix + str(LAYER_NUMBER) + '_Class_' + str(i) + '.npy'))

    dict_of_10_features = ten_feature_per_filter()

    list_of_features = np.zeros((len(dict_of_10_features), 10))

    for i in range(len(dict_of_10_features)):
        for x in range(10):
            list_of_features[i][x] = dict_of_10_features[i][x]

    list_of_features = np.array(list_of_features)
    pd_datapoints_10d = pd.DataFrame(list_of_features)

    # Using Sklearn MinMaxSacaler method
    scaler = preprocessing.MinMaxScaler()

    for i in range(len(pd_datapoints_10d.columns)):
        pd_datapoints_10d[i] = scaler.fit_transform(pd_datapoints_10d[i].values.reshape(-1, 1))

    hist = pd.DataFrame(columns=['n_clusters', 'silhouette_avg'])

    range_n_clusters = np.arange(4, len(pd_datapoints_10d))
    # range_n_clusters = np.arange(4, 56)

    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        #     clusterer = KMedoids(n_clusters=n_clusters, random_state=6, max_iter=10000)
        clusterer = KMeans(n_clusters=n_clusters, random_state=6, max_iter=10000)
        cluster_labels = clusterer.fit_predict(pd_datapoints_10d)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(pd_datapoints_10d, cluster_labels)

        # Logging the results
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        hist.loc[n_clusters] = (n_clusters, silhouette_avg)

    dict_of_silhouette.add(LAYER_NUMBER, hist['silhouette_avg'].idxmax())

with open('ResNet_10D_Best_K_in_layers.pickle', 'wb') as handle:
    pickle.dump(dict_of_silhouette, handle, protocol=pickle.HIGHEST_PROTOCOL)
