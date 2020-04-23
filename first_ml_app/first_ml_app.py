"""
DOCSTRING
"""
import sklearn.datasets as datasets
import sklearn.metrics as metrics
import tensorflow.estimator as estimator

iris = datasets.load_iris()
classifier = estimator.LinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print('Accuracy:%f' % score)
