# X-means

The X-means is one of the methods to estimate the number of clusters in a dataset. The method has been proposed by Pelleg and Moore 2000.

"simple_xmeans.py" is simple implementation of the X-means in Python. In Improve-Structure, the program does not calculate the centroids of two subclusters using the centroids calculated in Improve-Params. Furthermore, the upon next iteration, the program does not use the centroids calculated in Improve-Structure.  The program only updates the number of clusters k. Thus, the program may show lower performance than the pure X-means. However, we will more easily understand the essence of the algorithm of the X-means from the program source.

## Dependences

- numpy
- scikit-learn

## Reference

D. Pelleg and A. Moore (2000) X-means: Extending K-means with Efficient Estimation of the Number of Clusters. In Proceedings of the 17th International Conf. on Machine Learning, 727--734.

## Notice

To keep simpleness of the program code, it does not avoid to make dead units which have no data point. Therefore, this program may stop by math domain error.
