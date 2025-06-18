import os
from pyspark.sql import SparkSession
from elt import TaxiTripPipeline
from ml import TaxiTripMLPipeline
from delta import configure_spark_with_delta_pip


class Session:
    def __init__(self):
        builder = SparkSession.builder.appName("TaxiTripAnalysis") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0")
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()

    def get_session(self):
        return self.spark


os.environ['GIT_PYTHON_REFRESH'] = 'quiet'


def main():
    session = Session()
    spark = session.get_session()

    elt_pipeline = TaxiTripPipeline(spark)
    elt_pipeline.transform()

    ml_pipeline = TaxiTripMLPipeline(spark)
    ml_pipeline.run()


if __name__ == "__main__":
    main()