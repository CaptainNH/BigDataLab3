# Лабораторная работа Lakehouse-toy

## Запуск
### 1. Скачать датасет [New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data).

### 2. Создать в корне проекта папку data и положить туда файлы train.csv и test.csv

### 3. Собрать docker образ
```
docker build -t bigdatalab3 .
```
### 4. Запустить контейнер
```
docker compose up -d
```
### 5. Остановка
```
docker compose down
```
