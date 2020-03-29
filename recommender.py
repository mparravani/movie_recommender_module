import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments

class Recommender():
    '''
    Recommender system using FunkSVD, as well as knowledge and rank based methods
    '''
    def __init__(self, ):
        '''
        Instantiates recommender system class
        '''

    def fit(self, ):
        '''
        INPUT:
        reviews_pth - path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies_pth - path to csv with each movie and movie information in each row
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations
        
        OUTPUT:
        None - stores the following as attributes:
        n_users - the number of users (int)
        n_movies - the number of movies (int)
        num_ratings - the number of ratings made (int)
        reviews - dataframe with four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies - dataframe of
        user_item_mat - (np array) a user by item numpy array with ratings and nans for values
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations
        '''

    def predict_rating(self, ):
        '''
        makes predictions of a rating for a user on a movie-user combo
        '''

    def make_recs(self,):
        '''
        given a user id or a movie that an individual likes
        make recommendations
        '''


if __name__ == '__main__':
    # test different parts to make sure it works
