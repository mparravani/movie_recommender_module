import numpy as np
import pandas as pd
import recommender_functions as rf
import sys # can use sys to take command line arguments

class Recommender():
    '''
    Recommender system using FunkSVD, as well as knowledge and rank based methods
    '''
    def __init__(self, ):
        '''
        Instantiates recommender system class
        '''

    def fit(self, reviews_pth, movies_pth, latent_features = 15, learning_rate = 0.005, iters = 100):
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
        
        #import data and instantiate variables
        self.movies = pd.read_csv(movies_pth)
        self.reviews = pd.read_csv(reviews_pth)

        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        #create 

        # Create user-by-item matrix - this will keep track of order of users and movies in u and v
        self.user_item = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_item_df = self.user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.user_item_mat = self.user_item_df

        
        # Set up useful values to be used through the rest of the function
        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))
        self.user_ids_series = np.array(self.user_item_df.index)
        self.movie_ids_series = np.array(self.user_item_df.columns)
        
        # initialize the user and movie matrices with random values
        self.user_mat = np.random.rand(self.n_users, self.latent_features)
        self.movie_mat = np.random.rand(self.latent_features, self.n_movies)
        
        #initialize sse at 0 for first iteration
        sse_accum = 0
    
        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")
        
        # for each iteration
        for iteration in range(self.iters):
            
            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):
                    
                    # if the rating exists
                    if self.user_item_mat[i, j] > 0:
                        
                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.user_item_mat[i, j] - np.dot(self.user_mat[i, :], self.movie_mat[:, j])
                        
                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in rangeself.(latent_features):
                            self.user_mat[i, k] += self.learning_rate * (2*diff*self.movie_mat[k, j])
                            self.movie_mat[k, j] += self.learning_rate * (2*diff*self.user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / num_ratings))
         
        # Knowledge based fit
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)


    def predict_rating(self, ):
        '''
        INPUT:
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''
        

    def make_recs(self,):
        '''
        given a user id or a movie that an individual likes
        make recommendations
        '''


if __name__ == '__main__':
    # test different parts to make sure it works
