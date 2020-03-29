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
        self.user_item_mat = np.array(self.user_item_df)

        
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
                        for k in range(self.latent_features):
                            self.user_mat[i, k] += self.learning_rate * (2*diff*self.movie_mat[k, j])
                            self.movie_mat[k, j] += self.learning_rate * (2*diff*self.user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))
         
        # Knowledge based fit
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)


    def predict_rating(self, user_id, movie_id):
        '''
        INPUT:
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''

        # User row and Movie Column
        user_row = np.where(self.user_ids_series == user_id)[0][0]
        movie_col = np.where(self.movie_ids_series == movie_id)[0][0]
        
        # Take dot product of that row and column in U and V to make prediction
        pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])
    

    def make_recs(self,_id, _id_type='user', rec_num=5):
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        rec_num - number of recommendations to return (int)

        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        '''
        rec_ids, rec_names = None, None
        if _id_type == 'user':
            if _id in self.user_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]
                
                # take the dot product of that row and the V matrix
                preds = np.dot(self.user_mat[idx,:],self.movie_mat)
                
                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.user_item_df.columns[indices]
                rec_names = rf.get_movie_names(rec_ids, self.movies)
                
            else:
                # if we don't have this user, give just top ratings back
                rec_names = rf.popular_recommendations(_id, rec_num, self.ranked_movies)
                
        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(rf.find_similar_movies(_id, self.movies))[:rec_num]
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")
    
        return rec_ids, rec_names

if __name__ == '__main__':
    import recommender as r

    #instantiate recommender
    rec = r.Recommender()

    # fit recommender
    rec.fit(reviews_pth='data/train_data.csv', movies_pth= 'data/movies_clean.csv', learning_rate=.01, iters=1)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recs(8,'user')) # user in the dataset
    print(rec.make_recs(1,'user')) # user not in dataset
    print(rec.make_recs(1853728,'movie')) # movie in the dataset
    print(rec.make_recs(1,'movie')) # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)