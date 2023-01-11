import os
import numpy as np
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import Select
from operator import add
from pyspark.ml.feature import Normalizer

"""класс для обучения модели на логистической регрессии и дереве принятия решения для оценки качество работы алгоритма
класстеризации связанной с предыдущей работой."""
class Train:
    def __init__(self, df, vec_assembler, scaler, column_init):
        self.df = df
        self.vec_assembler = vec_assembler
        self.scaler = scaler
        self.column_init = column_init

    def train_models(self):
        label = 'labels'
        print("columns selected")

        # label - метки которые присвоил алгоритм KMeans
        """иници-я Дерева принятия решения"""
        tree = DecisionTreeClassifier(labelCol=label, featuresCol='standardized',
                                     predictionCol='tree_pred', maxDepth=14)

        column_select = ['standardized', label, 'tree_pred']
        column_select_tree = Select(column_select)

        """иниц-я логистической регрессии"""
        log_reg = LogisticRegression(maxIter=8,
                                    tol=1e-6,
                                    fitIntercept=True,
                                    labelCol=label,
                                    featuresCol='standardized',
                                    probabilityCol='lr_prob')

        columns_to_select = ['standardized', label, 'tree_pred', 'lr_prob']

        column_select_logreg = Select(columns_to_select)

        #подготовка pipeline какие этапы передаюутся в него
        pipeline_stages = [self.vec_assembler, self.scaler, self.column_init, tree, column_select_tree, log_reg,
                           column_select_logreg]

        pipeline = Pipeline(stages=pipeline_stages)
        model = pipeline.fit(self.df)
        output = model.transform(self.df)
        print("print станд вероятность принад классу ", output.select(output.lr_prob))


        output = output.withColumn('lr_prob', vector_to_array(output['lr_prob'])) \
            .withColumn('logreg_pred_labels',
                        F.expr('array_position(lr_prob, array_max(lr_prob))-1').cast(DoubleType())) \
            .drop('lr_prob')

        """оцениваем результаты класстеризации"""
        clust_evaluator = ClusteringEvaluator(predictionCol=label, featuresCol='standardized')

        class_evaluator_rfc = MulticlassClassificationEvaluator(labelCol=label,
                                                                predictionCol='tree_pred',
                                                                metricName='accuracy')

        class_evaluator_logreg = MulticlassClassificationEvaluator(labelCol=label,
                                                                   predictionCol='logreg_pred_labels',
                                                                   metricName='accuracy')

        clust_score = clust_evaluator.evaluate(output)
        accuracy_tree = class_evaluator_rfc.evaluate(output)
        accuracy_logreg = class_evaluator_logreg.evaluate(output)

        y_true = np.array(output.select('labels').collect())
        y_pred_tree = np.array(output.select('tree_pred').collect())
        y_pred_logreg = np.array(output.select('logreg_pred_labels').collect())


        print('score of silhouette = ', str(clust_score))
        print('tree accuracy: ', accuracy_tree)
        print('logreg accuracy: ', accuracy_logreg, '\n')

        print('Classification Report tree:')
        print(classification_report(y_true, y_pred_tree, zero_division=0))
        print()

        print('Classification Report of the log_reg: ')
        print(classification_report(y_true, y_pred_logreg, zero_division=0))

        print('DecisionTreeClassifier confusion matrix:')
        cm_tree = confusion_matrix(y_true, y_pred_tree)
        print(cm_tree)
        print()

        print('LogisticRegresion confusion matrix: ')
        cm_log_reg = confusion_matrix(y_true, y_pred_logreg)
        print(cm_log_reg)