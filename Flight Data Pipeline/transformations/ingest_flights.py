from pyspark import pipelines as dp

from pyspark_datasources import OpenSkyDataSource

spark.dataSource.register(OpenSkyDataSource)

@dp.table
def ingest_flights():
    return spark.readStream.format("opensky").load()
