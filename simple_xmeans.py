import numpy as np
import math as mt
import sys
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics

class XMeans:
    def loglikelihood(self, r, rn, var, m, k):
        l1 = - rn / 2.0 * mt.log(2 * mt.pi)
        l2 = - rn * m / 2.0 * mt.log(var)
        l3 = - (rn - k) / 2.0
        l4 = rn * mt.log(rn)
        l5 = - rn * mt.log(r)

        return l1 + l2 + l3 + l4 + l5

    def __init__(self, X, kmax = 20):
        self.X = X
        self.num = np.size(self.X, axis=0)
        self.dim = np.size(X, axis=1)
        self.KMax = kmax

    def fit(self):
        k = 1
        X = self.X
        M = self.dim
        num = self.num

        while(1):
            ok = k

            #Improve Params
            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            m = kmeans.cluster_centers_

            #Improve Structure
            #Calculate BIC
            p = M + 1

            obic = np.zeros(k)

            for i in range(k):
                rn = np.size(np.where(labels == i))
                var = np.sum((X[labels == i] - m[i])**2)/float(rn - 1)
                obic[i] = self.loglikelihood(rn, rn, var, M, 1) - p/2.0*mt.log(rn)

            #Split each cluster into two subclusters and calculate BIC of each splitted cluster
            sk = 2 #The number of subclusters
            nbic = np.zeros(k)
            addk = 0

            for i in range(k):
                ci = X[labels == i]
                r = np.size(np.where(labels == i))

                kmeans = KMeans(n_clusters=sk).fit(ci)
                ci_labels = kmeans.labels_
                sm = kmeans.cluster_centers_

                for l in range(sk):
                    rn = np.size(np.where(ci_labels == l))
                    var = np.sum((ci[ci_labels == l] - sm[l])**2)/float(rn - sk)
                    nbic[i] += self.loglikelihood(r, rn, var, M, sk)

                p = sk * (M + 1)
                nbic[i] -= p/2.0*mt.log(r)

                if obic[i] < nbic[i]:
                    addk += 1

            k += addk

            if ok == k or k >= self.KMax:
                break


        #Calculate labels and centroids
        kmeans = KMeans(n_clusters=k).fit(X)
        self.labels = kmeans.labels_
        self.k = k
        self.m = kmeans.cluster_centers_


if __name__ == '__main__':

    #Blobs (Isotropic Gaussian distributions)
    X, TrueLabels = datasets.make_blobs(n_samples=1500, centers=3, n_features=3)

    xm = XMeans(X)
    xm.fit()

    purity = metrics.adjusted_rand_score(TrueLabels, xm.labels)
    nmi = metrics.normalized_mutual_info_score(TrueLabels, xm.labels)
    ari = metrics.adjusted_rand_score(TrueLabels, xm.labels)

    print("Blobs")
    print("True k = 3, Estimated k = " + str(xm.k) + ", purity = " + str(purity) + ", NMI = " + str(nmi) + ", ARI = " + str(ari) + "\n")

    #Iris dataset
    dataset = datasets.load_iris()
    X = dataset.data
    TrueLabels = dataset.target

    xm = XMeans(X)
    xm.fit()

    purity = metrics.adjusted_rand_score(TrueLabels, xm.labels)
    nmi = metrics.normalized_mutual_info_score(TrueLabels, xm.labels)
    ari = metrics.adjusted_rand_score(TrueLabels, xm.labels)

    print("Iris dataset")
    print("True k = 3, Estimated k = " + str(xm.k) + ", purity = " + str(purity) + ", NMI = " + str(nmi) + ", ARI = " + str(ari) + "\n")
