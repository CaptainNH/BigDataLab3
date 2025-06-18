from pyspark.sql.functions import col, unix_timestamp, hour, dayofweek, month
from pyspark.sql import functions as F
from delta.tables import DeltaTable
from pyspark.ml.feature import StringIndexer, VectorAssembler


class TaxiTripPipeline:
    def __init__(self, spark):
        self.spark = spark

    def transform(self):
        self._create_bronze()
        self._create_silver()
        self._create_gold()

    def _create_bronze(self):
        # Загрузка train и test данных
        train_df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv("data/train.csv")

        test_df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv("data/test.csv")

        # Очистка данных - удаляем строки с пропущенными значениями
        train_df = train_df.na.drop()
        test_df = test_df.na.drop()

        # Сохраняем в бронзовый слой
        train_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/bronze/taxi_trips_train")

        test_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/bronze/taxi_trips_test")

    def _create_silver(self):
        # Чтение из бронзового слоя
        bronze_train_df = self.spark.read \
            .format("delta") \
            .load("data/bronze/taxi_trips_train")

        bronze_test_df = self.spark.read \
            .format("delta") \
            .load("data/bronze/taxi_trips_test")

        # Извлечение признаков из даты и времени
        for df in [bronze_train_df, bronze_test_df]:
            df = df.withColumn("pickup_hour", hour("pickup_datetime"))
            df = df.withColumn("pickup_day", dayofweek("pickup_datetime"))
            df = df.withColumn("pickup_month", month("pickup_datetime"))
            df = df.withColumn("trip_duration_min",
                               (unix_timestamp("dropoff_datetime") -
                                unix_timestamp("pickup_datetime")) / 60)

        # Фильтрация аномальных значений
        bronze_train_df = bronze_train_df.filter(
            (col("trip_duration_min") > 0) &
            (col("trip_duration_min") < 180) &  # поездки меньше 3 часов
            (col("passenger_count") > 0) &
            (col("passenger_count") <= 6))

        bronze_test_df = bronze_test_df.filter(
            (col("trip_duration_min") > 0) &
            (col("trip_duration_min") < 180) &
            (col("passenger_count") > 0) &
            (col("passenger_count") <= 6))

        # Сохраняем в серебряный слой
        bronze_train_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/silver/taxi_trips_train")

        bronze_test_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/silver/taxi_trips_test")

    def _create_gold(self):
        # Чтение из серебряного слоя
        silver_train_df = self.spark.read \
            .format("delta") \
            .load("data/silver/taxi_trips_train")
        silver_test_df = self.spark.read \
            .format("delta") \
            .load("data/silver/taxi_trips_test")

        # Оптимизация таблицы
        delta_table = DeltaTable.forPath(self.spark, "data/silver/taxi_trips_train")
        delta_table.optimize().executeZOrderBy("vendor_id")

        delta_table = DeltaTable.forPath(self.spark, "data/silver/taxi_trips_test")
        delta_table.optimize().executeZOrderBy("vendor_id")

        # Подготовка фичей для ML
        categorical_cols = ['vendor_id', 'store_and_fwd_flag', 'pickup_hour', 'pickup_day', 'pickup_month']
        numerical_cols = ['passenger_count', 'pickup_longitude', 'pickup_latitude',
                          'dropoff_longitude', 'dropoff_latitude']

        # Индексация категориальных признаков
        indexers = [
            StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
            for col_name in categorical_cols
        ]

        for indexer in indexers:
            model = indexer.fit(silver_train_df)
            silver_train_df = model.transform(silver_train_df)
            silver_test_df = model.transform(silver_test_df)

        # Собираем все фичи в один вектор
        feature_cols = [f"{col}_index" for col in categorical_cols] + numerical_cols
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )

        train_df = assembler.transform(silver_train_df)
        test_df = assembler.transform(silver_test_df)

        # Добавляем целевую переменную (продолжительность поездки)
        train_df = train_df.withColumn("label", col("trip_duration_min"))
        test_df = test_df.withColumn("label", col("trip_duration_min"))

        # Сохраняем в золотой слой
        train_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/gold/taxi_trips_train")

        test_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/gold/taxi_trips_test")