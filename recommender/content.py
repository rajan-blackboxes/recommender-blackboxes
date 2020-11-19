"""
Contains recommender systems algorithms based on content(items-items)
"""
import pandas as pd


# currently tested on movies dataset
class FactorizationBased:
    def __init__(
        self,
        dataframe,
        product_title,
        id_column: str,
        genre_column: str,
        ratings_column: str,
    ):
        """
        Shaped based on user ratings(no of clicks, liked..) and given genres(category, tags..)
        Recommends based on similarity between content of those products(items)
        :param dataframe: Pandas Dataframe
        :param product_title: Title of product for e.g movies titles, product titles (column)
        :param id_column: Product id column
        :param genre_column: It means product category, tags, genre etc
        :param ratings_column: User ratings | clicked | liked etc
        """
        self.dataframe = dataframe
        self.id_column = id_column
        self.product_title = product_title
        self.genre_column = genre_column
        self.ratings_column = ratings_column
        self.one_hot_on_genre = dataframe.copy()
        self.genres_set = set()

        # Creates one-hot vector on genres
        for index, row in dataframe.iterrows():
            for genre in row[genre_column]:
                self.genres_set.add(genre)
                self.one_hot_on_genre.at[index, genre] = 1
        self.one_hot_on_genre = self.one_hot_on_genre.fillna(0)

    def recommend(self, user_inputs, show_upto=20):
        """
        Recommend product on given user_inputs
        :param user_inputs: dataframe product_title | id_column , ratings_column
        :param show_upto: How much to recommend!!!
        :return:
        """
        # Getting product id , based on input product titles
        input_products = pd.merge(self.dataframe, user_inputs)
        # Getting subset of products that input watched
        user_products = self.one_hot_on_genre[
            self.one_hot_on_genre[self.id_column].isin(input_products[self.id_column])
        ]
        user_products = user_products.reset_index(drop=True)
        user_genre_table = user_products[list(self.genres_set)]
        # dot product to get weights
        user_profile = user_genre_table.transpose().dot(
            input_products[self.ratings_column]
        )
        # get genre of every product in dataframe
        genre_table = self.one_hot_on_genre.set_index(
            self.one_hot_on_genre[self.id_column]
        )[list(self.genres_set)]
        # Multiply the genres by the weights and average them
        recommendation_df = (
            ((genre_table * user_profile).sum(axis=1)) / user_profile.sum()
        ).sort_values(ascending=False)
        recommendations = self.dataframe.loc[
            self.dataframe[self.id_column].isin(
                recommendation_df.head(show_upto).keys()
            )
        ]
        return recommendations
