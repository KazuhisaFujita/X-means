# X-means

X-meansはデータセット内のクラスタ数を推定する方法の1つです．この方法は，Pelleg and Moore 2000によって提案されています．

"simple_xmeans.py"は，PythonによるX-meansの簡単な実装です．プログラムは，Improve-StructureでImprove-Paramsで計算されたセントロイドを使用して2つのサブクラスターのセントロイドを計算しません．さらに，次の反復では，このプログラムはImprove-Structureで計算されたセントロイドを使用しません．このプログラムは，クラスタ数kを更新するだけです．従って，このプログラムは純粋なX-meansより低い性能を示すことがあるかもしれません．しかし，このプログラムのソースからX-meansのアルゴリズムの本質をより簡単に理解することができるでしょう．


## Reference

D. Pelleg and A. Moore (2000) X-means: Extending K-means with Efficient Estimation of the Number of Clusters. In Proceedings of the 17th International Conf. on Machine Learning, 727--734.