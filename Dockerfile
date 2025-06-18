FROM bitnami/spark:3.5.0

USER root

RUN pip install pyspark==3.5.0 delta-spark==3.0.0 mlflow==2.8.0

WORKDIR /app
