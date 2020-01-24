import numpy as np
import pandas as pd
import os
import sys
import random


class k_means():
    # CONSTRUCTOR THAT WE WILL USE TO KEEP TRACK OF THINGS THROUGHOUT OUR KMEANS RUN
    def __init__(self, dataset, k, opt):
        """
            *CONSTRUCTOR FOR MY K_MEANS:
                * dataset: THE PROPER SUBSETTED DATASET WE WILL USE
                * k: THE NUMBER OF CLUSTERS FOR OUR DATA
                * opt: THE OPTION WE CHOOSE FOR RUNNING, IMPORTANT FOR MANHATTAN VS EUCLEDIAN
        """
        # basics
        self.data_matrix = np.matrix(dataset)  # convert to a data matrix
        self.k = k  # store number of clusters for future use
        self.opt = opt # this will decide which distance metric we will use later on
        self.dimension = np.size(self.data_matrix, 0)  # just storing how many rows we have in this b
        self.classification = np.zeros(self.dimension) # store the cluster # that you classified this as

        # get a random centroid to start
        temp_hold_ind = np.zeros(k)

        # in order to get random centroid lets yolo and get some random indices
        for x in range(0, np.size(temp_hold_ind, 0)):
            i = random.randrange(self.dimension)
            temp_hold_ind[x] = i

        # convert to int so that we can actually use this as an index array
        temp_hold_ind = temp_hold_ind.astype(int)

        # TADA, random centroid generated. only took like an hour to figure this out :(
        self.centroid = self.data_matrix[temp_hold_ind, :]
        self.backup_centroid = self.data_matrix[temp_hold_ind, :]

    def DelishClusters(self):
        """
            DESCRIPTION:
                * STEP 1). go through every point in the dataset and determine the cluster membership of said datapoint
                * STEP 2). once STEP 1). is complete we can recalculate the centroids according to the procedure below:
                    * 2A). go through each point in the original data set, check if it belongs to x cluster and add to
                            array x
                * STEP 3). take the median of each cluster x array and tada the centroids are computed
                * STEP 4). repeat STEPS 1 - 3 until clusters of previous run = clusters produced by current run

            INPUT:
                * NONE

            OUTPUT:
                * NONE (STORED IN THE OBJECT ITSELF) 
        :return:
        """
        iteration = 0  # iteration tracker for while loop
        truth = False  # boolean value to control break of loop (determines when we stop k-means)

        # CALCULATE CENTROIDS BB
        while not truth:
            # print(self.centroid)
            # TIME TO GET STEP 1'D
            for point in range(0, self.dimension):
                point_diff = 0

                if self.opt == 4:
                    # OPTION 4 HAS BEEN SELECTED SO WE MUST CALL MANHATTAN:
                    # print("MANHATTAN")
                    point_diff = self.ManhattanDistance(self.data_matrix[point], self.centroid, 0)
                else:
                    # OPTION 1 SELECTED SO WE MUST CALL EUCLEDIAN:
                    point_diff = self.EucledianDistance(self.data_matrix[point], self.centroid, 0)

                self.classification[point] = np.argmin(point_diff)

            # BACKUP OLD CENTROID
            self.backup_centroid = np.copy(self.centroid)

            # STEP 2 : LET'S RECALCULATE THE CENTROIDS
            for cluster in range(0, self.k):
                # print(cluster)
                clustering_set = []  # holding variable for the point indices in this cluster

                # get all of the points that are in this cluster
                for point in range(0, self.dimension):
                    # if the point matches, this cluster assignment add the indices
                    if self.classification[point] == cluster:
                        clustering_set.append(point)

                # now take the clustering set and make values out of it
                # print(len(clustering_set))
                ind_to_p = self.data_matrix[clustering_set, :]
                list_check = list(ind_to_p)

                # STEP 3: TAKE MEAN OF THE DATA FILLED ARRAY AND YOU ARE ALMOST DONE !!!
                if len(list_check) != 0:
                    new_centroid = np.mean(ind_to_p, axis=0)  # axis needed to ensure tuple output
                else:
                    # put this condition in to avoid RuntimeWarning: Mean of empty slice
                    continue

                # STEP 4: SET THIS AS THE NEW CENTROID
                self.centroid[cluster] = new_centroid

                # maintenance stuff: increment the run counter, run the check
                iteration = iteration + 1
                truth = np.array_equal(self.backup_centroid, self.centroid)

            # print("OLD CENTROID: ")
            # print(self.backup_centroid)
            # print("NEW CENTROID SINGLE CALCULATION: ")
            # print(self.centroid)

    def TasteTest(self):
        """
            DESCRIPTION:
                * this is simply going to take the stored final cluster and score it against the function
                * we use the following steps for that:
                    * take the distance of every point in each cluster from that cluster
                    * then take the square of that cluster
                    * sum all of the data points for every cluster
                    * sum every cluster

            INPUT:
                * NONE

            OUTPUT:
                * NONE
            :return:
        """
        # CALCULATE THE WCSSE
        wc_sse_sum = 0
        for clusterd in range(0, self.k):
            for data_point in range(0, self.dimension):
                result = 0
                if self.classification[data_point] == clusterd:
                    if self.opt == 4:
                        # USE MANHATTAN
                        result = self.ManhattanDistance(self.data_matrix[data_point], self.centroid[clusterd], 1)
                    else:
                        # USE EUCLEDIAN
                        # print(data_point)
                        result = self.EucledianDistance(self.data_matrix[data_point], self.centroid[clusterd], 1)

                # print(result)
                wc_sse_sum = wc_sse_sum + np.sum(result)

        return wc_sse_sum

    def ManhattanDistance(self, point, centroid, flag):
        """
                    *  DESCRIPTION:
                        * this function not only calculates the distance of a point to the centroid
                        * this function will also classify a point and set aside its classification so we can pull some
                                crazy shit later on for speed optimization

                    * DISTANCE FORMULA USED (MANHATTAN):
                        * take the absolute difference, and then sum

                    *  INPUT PARAMETERS:
                        * point: the point to be compared
                        * centroid: all the cluster centers to subtract from
                        * flag: raises to the power if we need to calculate wcsse

                    *  OUTPUT PARAMETERS:
                        * nothing because we will store our results in an array stored in the definition
        """
        # print("POINT: ")
        # print(point)
        # print("CENTROID: ")
        # print(centroid)

        # to return
        calc_diff = 0

        # compute the calculation
        if flag == 0:
            # REGULAR MANHATTAN
            calc_diff = np.absolute(point - centroid)
            calc_diff = calc_diff.sum(axis=1)
        elif flag == 1:
            # MOD MANHATTAN
            calc_diff = np.absolute(point-centroid)
            calc_diff = calc_diff.sum(axis=1)
            calc_diff = np.square(calc_diff)

        # distances calculated - launch classifier
        return calc_diff

    def EucledianDistance(self, point, centroid, flag):
        """
                *  DESCRIPTION:
                    * this function not only calculates the distance of a point to the centroid
                    * this function will also classify a point and set aside its classification so we can pull some
                        crazy shit later on for speed optimization

                * DISTANCE FORMULA USED (EUCLEDIAN):
                    * take the square of the differences, sum and then take the sqrt

                *  INPUT PARAMETERS:
                    * point: the point to be compared
                    * centroid: all the cluster centers to subtract from
                    * flag: raises to the power if we need to calculate wcsse

                *  OUTPUT PARAMETERS:
                    * nothing because we will just pass on our calculations
        """
        # print("POINT: ")
        # print(point)
        # print("CENTROID: ")
        # print(centroid)

        # value to return
        calc_diff = 0

        # compute the calculation
        if flag == 0:
            # NON-MODIFIED EUCLEDIAN
            calc_diff = np.square(point - centroid)
            calc_diff = calc_diff.sum(axis=1)
            calc_diff = np.sqrt(calc_diff)
        elif flag == 1:
            # MODIFIED EUCLEDIAN
            calc_diff = np.square(point - centroid)
            calc_diff = calc_diff.sum(axis=1)

        # distances calculated - launch classifier
        return calc_diff


def PreProcess(dataset, percent):
    """
        *  DESCRIPTION:
            * this dataset simply subsets the entire dataset, so ya, lets eshkitit

        *  INPUT PARAMETERS:
            * dataset: this is just the unaltered dataset
            * percent: percent of the dataset to sample and return to you (WE WILL USE THIS FOR RUN OPTION 5)

        *  OUTPUT PARAMETERS:
            * actual_df: this is a dataframe representing subsetted dataset. expanded following the rules described in
              the description
    """
    temp_df = pd.read_csv(dataset)

    # select the percentage accordingly
    if percent != 1:
        temp_df = temp_df.sample(frac=percent)

    """
        * DATASET HAS BEEN SELECTED SO WE NEED TO: 
            * subset the data 
            * take care of NaNs that may exist 
    """
    act_df = temp_df[['latitude', 'longitude', 'reviewCount', 'checkins']]
    act_df = act_df.fillna(0)

    # return the dataset iz ready
    return act_df


if __name__ == '__main__':
    """
                * DESCRIPTION :
                    *  Take in the input of the user and calculate the right function to call 
                    *  Use the argv[] to make the right decision to go forward

                * INPUT PARAMETERS :
                    *argv[1] = yelp.csv 
                    *argv[2] = k # of clusters 
                    *argv[3] = type of run we would like to do: 
                        * 1: regular k-means using the 4 original attributes {latitude,longitude,reviewCount,checkins} 
                        * 2: log transform reviewCount and checkins before running k-means
                        * 3: standardize all 4 of the attributes before clustering 
                        * 4: run k-means, however, utilize manhattan distance instead 
                        * 5: take a random sample of the data for clustering 
                        * 6: use improved score function for running k-means
                    *argv[4]: THIS IS FOR MY USE: you can pass in however much of the dataset you wish to subset for p5 
        """

    # define the input arguments
    data = sys.argv[1]
    num_clusters = sys.argv[2]
    type_run = sys.argv[3]
    percent = 1  # not a required parameter
    if type_run == "5":
        percent = 0.06

    # first step lets preprocess the data - TYPE RUN 5 ALREADY BUILT IN
    proper_data = PreProcess(data, percent)

    if type_run == "2":
        # LOG TRANSFORM ONLY TWO OF THE ATTRIBUTES:
        proper_data['reviewCount'] = np.log(proper_data['reviewCount'])
        proper_data['checkins'] = np.log(proper_data['checkins'])
        # print(proper_data)
    elif type_run == "3":
        # LOG TRANSFORM ALL 4 OF THE ATTRIBUTES:
        # STACK FORMULA : normalized_df=(df-df.mean())/df.std()
        proper_data = (proper_data - proper_data.mean())/proper_data.std()
        # print(proper_data)

    # instantiate kmeans instance bbgurl - TYPE RUN 4 ALREADY BUILT IN
    k_sera = k_means(proper_data, int(num_clusters), int(type_run))
    k_sera.DelishClusters()
    score = k_sera.TasteTest()

    # print every ****** thing
    print("WC-SSE= %f" % score)
    for k in range(0, int(num_clusters)):
        conv_arr = np.asarray(k_sera.centroid[k])
        conv_arr = np.array(conv_arr).ravel()
        conv_arr = list(conv_arr)
        print("Centroid%d:" % k, conv_arr)
