import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col

mlflow.set_tracking_uri("file:/app/logs/mlruns")


class TaxiTripMLPipeline:
    def __init__(self, spark):
        self.spark = spark
        self.target_col = 'label'
        self.experiment_name = "Taxi_Trip_Duration_Prediction"

    def run(self):
        mlflow.set_experiment(self.experiment_name)
        self._load_data()
        pipeline = self._prepare_pipeline()
        self._train_and_evaluate(pipeline)

    def _load_data(self):
        self.train_df = self.spark.read \
            .format("delta") \
            .load("data/gold/taxi_trips_train") \
            .cache()

        self.test_df = self.spark.read \
            .format("delta") \
            .load("data/gold/taxi_trips_test")

        # Убедимся, что целевая переменная в числовом формате
        self.train_df = self.train_df.withColumn(self.target_col, col(self.target_col).cast("float"))
        self.test_df = self.test_df.withColumn(self.target_col, col(self.target_col).cast("float"))

    def _prepare_pipeline(self):
        # Выбор модели (регрессия вместо классификации)
        lr = LinearRegression(
            featuresCol="features",
            labelCol=self.target_col,
            maxIter=100
        )

        # Создаем пайплайн
        pipeline = Pipeline(stages=[lr])

        return pipeline

    def _train_and_evaluate(self, pipeline):
        with mlflow.start_run(run_name="Linear_Regression_Baseline"):
            # Логируем параметры
            mlflow.log_param("model_type", "LinearRegression")

            # Кросс-валидация
            param_grid = ParamGridBuilder() \
                .addGrid(pipeline.getStages()[-1].getParam("regParam"), [0.01, 0.1]) \
                .addGrid(pipeline.getStages()[-1].getParam("elasticNetParam"), [0.0, 0.5, 1.0]) \
                .build()

            evaluator = RegressionEvaluator(
                labelCol=self.target_col,
                predictionCol="prediction",
                metricName="rmse"
            )

            cv = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=3,
                parallelism=4
            )

            # Обучение модели
            cv_model = cv.fit(self.train_df)
            best_model = cv_model.bestModel

            # Логирование лучших параметров
            mlflow.log_param("best_reg_param", best_model.stages[-1].getRegParam())
            mlflow.log_param("best_elastic_net", best_model.stages[-1].getElasticNetParam())

            # Предсказания на тестовых данных
            predictions = best_model.transform(self.test_df)

            # Оценка качества (метрики регрессии)
            rmse = evaluator.evaluate(predictions)
            mae = evaluator.setMetricName("mae").evaluate(predictions)
            r2 = evaluator.setMetricName("r2").evaluate(predictions)

            # Логирование метрик
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Логирование модели
            mlflow.spark.log_model(best_model, "model")