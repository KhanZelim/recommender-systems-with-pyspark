from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

class Recommender:
    def __init__(self) -> None:
        self.links_path = "./data/links.csv"
        self.movies_path = "./data/movies.csv"
        self.ratings_path = "./data/ratings.csv"
        self.tags_path = "./data/tags.csv"
        self.spark = SparkSession.builder \
            .appName("Recommender") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "4") \
            .getOrCreate()
        self.evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    def read_file(self, filepath) -> DataFrame:
        return self.spark.read.options(header=True, inferSchema=True).csv(filepath)

    def stop(self) -> None:
        self.spark.stop()

    def create_models(self) -> ALS:
        ratings = self.read_file(self.ratings_path).select("userId", "movieId", "rating")

        (training_data, test_data) = ratings.randomSplit([0.8, 0.2], seed=38)
        
        als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative = True, implicitPrefs = False)

        param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10]) \
            .addGrid(als.maxIter, [10]) \
            .addGrid(als.regParam, [.1]) \
            .build()
        
        cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=self.evaluator, numFolds=5, parallelism=2)

        model = cv.fit(training_data)
        best_model = model.bestModel

        return best_model
    
    def get_recommendations(self) -> DataFrame:
        best_model = self.create_models()
        recommendations = best_model.recommendForAllUsers(10)
        expanded_recommandations = recommendations.select("userId", explode("recommendations").alias("recommendation"))
        final_recommendations = expanded_recommandations.select("userId", "recommendation.movieId", "recommendation.rating")

        movies = self.read_file(self.movies_path)
        ratings = self.read_file(self.ratings_path).select("userId", "movieId", "rating")

        final_recommendations = final_recommendations.join(movies, ["movieId"], "left")
        final_recommendations = final_recommendations.join(ratings.withColumnRenamed("rating", "user_rating"),["userId", "movieId"],"left")
        final_recommendations = final_recommendations.filter("user_rating is NULL")
        final_recommendations = final_recommendations.select("userId", "movieId", "rating", "title", "genres")

        return final_recommendations.show()