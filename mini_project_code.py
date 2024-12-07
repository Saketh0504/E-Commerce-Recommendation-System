def recolduser(id,desc):
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns

  from sklearn.metrics.pairwise import cosine_similarity
  from sklearn.feature_extraction.text import TfidfVectorizer

  import os
  from scipy.sparse import coo_matrix

# Read your dataset (replace 'marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv' with your dataset path)
  train_data = pd.read_csv('marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv', sep='\t')
  train_data.columns

  train_data = train_data[['Uniq Id','Product Id', 'Product Rating', 'Product Reviews Count', 'Product Category', 'Product Brand', 'Product Name', 'Product Image Url', 'Product Description', 'Product Tags']]
  train_data.head(3)

  train_data['Product Tags']

  train_data.shape

  train_data.isnull().sum()

  # Fill missing values in 'Product Rating' with a default value (e.g., 0)
  train_data['Product Rating'] = train_data['Product Rating'].fillna(0)
  # Fill missing values in 'Product Reviews Count' with a default value (e.g., 0)
  train_data['Product Reviews Count'] = train_data['Product Reviews Count'].fillna(0)
  # Fill missing values in 'Product Category' with a default value (e.g., 'Unknown')
  train_data['Product Category'] = train_data['Product Category'].fillna('')
  # Fill missing values in 'Product Brand' with a default value (e.g., 'Unknown')
  train_data['Product Brand'] = train_data['Product Brand'].fillna('')
  # Fill missing values in 'Product Description' with an empty string
  train_data['Product Description'] = train_data['Product Description'].fillna('')

  train_data.isnull().sum()

  train_data.duplicated().sum()

  # make columns shorter
  # Define the mapping of current column names to shorter names
  column_name_mapping = {
      'Uniq Id': 'ID',
      'Product Id': 'ProdID',
      'Product Rating': 'Rating',
      'Product Reviews Count': 'ReviewCount',
      'Product Category': 'Category',
      'Product Brand': 'Brand',
      'Product Name': 'Name',
      'Product Image Url': 'ImageURL',
      'Product Description': 'Description',
      'Product Tags': 'Tags',
      'Product Contents': 'Contents'
  }
  # Rename the columns using the mapping
  train_data.rename(columns=column_name_mapping, inplace=True)

  train_data['ID'] = train_data['ID'].str.extract(r'(\d+)').astype(float)
  train_data['ProdID'] = train_data['ProdID'].str.extract(r'(\d+)').astype(float)

  # Most popular items
  popular_items = train_data['ProdID'].value_counts().head(5)

  # most rated counts
  #train_data['Rating'].value_counts().plot(kind='bar',color='red')

  import spacy
  from spacy.lang.en.stop_words import STOP_WORDS

  nlp = spacy.load("en_core_web_sm")

  def clean_and_extract_tags(text):
      doc = nlp(text.lower())
      tags = [token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
      return ', '.join(tags)

  columns_to_extract_tags_from = ['Category', 'Brand', 'Description']

  for column in columns_to_extract_tags_from:
      train_data[column] = train_data[column].apply(clean_and_extract_tags)

  # Concatenate the cleaned tags from all relevant columns
  train_data['Tags'] = train_data[columns_to_extract_tags_from].apply(lambda row: ', '.join(row), axis=1)

  average_ratings = train_data.groupby(['Name','ReviewCount','Brand','ImageURL'])['Rating'].mean().reset_index()

  top_rated_items = average_ratings.sort_values(by='Rating', ascending=False)

  rating_base_recommendation = top_rated_items.head(10)

  rating_base_recommendation['Rating'] = rating_base_recommendation['Rating'].astype(int)
  rating_base_recommendation['ReviewCount'] = rating_base_recommendation['ReviewCount'].astype(int)

  rating_base_recommendation[['Name','Rating','ReviewCount','Brand','ImageURL']] = rating_base_recommendation[['Name','Rating','ReviewCount','Brand','ImageURL']]

  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity

  tfidf_vectorizer = TfidfVectorizer(stop_words='english')
  tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
  cosine_similarities_content = cosine_similarity(tfidf_matrix_content,tfidf_matrix_content)

  item_name = 'OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'
  item_index = train_data[train_data['Name']==item_name].index[0]

  similar_items = list(enumerate(cosine_similarities_content[item_index]))

  similar_items = sorted(similar_items, key=lambda x:x[1], reverse=True)
  top_similar_items = similar_items[1:10]

  recommended_items_indics = [x[0] for x in top_similar_items]

  train_data.iloc[recommended_items_indics][['Name','ReviewCount','Brand']]

#   def content_based_recommendations(train_data, item_name, top_n=10):
#       # Check if the item name exists in the training data
#       if item_name not in train_data['Name'].values:
#           print(f"Item '{item_name}' not found in the training data.")
#           return pd.DataFrame()

#       # Create a TF-IDF vectorizer for item descriptions
#       tfidf_vectorizer = TfidfVectorizer(stop_words='english')

#       # Apply TF-IDF vectorization to item descriptions
#       tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

#       # Calculate cosine similarity between items based on descriptions
#       cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

#       # Find the index of the item
#       item_index = train_data[train_data['Name'] == item_name].index[0]

#       # Get the cosine similarity scores for the item
#       similar_items = list(enumerate(cosine_similarities_content[item_index]))

#       # Sort similar items by similarity score in descending order
#       similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

#       # Get the top N most similar items (excluding the item itself)
#       top_similar_items = similar_items[1:top_n+1]

#       # Get the indices of the top similar items
#       recommended_item_indices = [x[0] for x in top_similar_items]

#       # Get the details of the top similar items
#       recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

#       return recommended_items_details
  def content_based_recommendations(train_data, tag, top_n=10):
        # Create a TF-IDF vectorizer for item tags
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        # Apply TF-IDF vectorization to item tags
        tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

        # Calculate cosine similarity between items based on tags
        cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

        # Check if the tag is present in the TF-IDF vocabulary
        if tag not in tfidf_vectorizer.get_feature_names_out():
            print(f"Tag '{tag}' not found in the data.")
            return pd.DataFrame()

        # Find the vector representation of the tag
        tag_vector = tfidf_vectorizer.transform([tag])

        # Calculate similarity of the tag with all items in the dataset
        tag_similarities = cosine_similarity(tag_vector, tfidf_matrix_content).flatten()

        # Get the top N most similar items based on the tag
        similar_items = list(enumerate(tag_similarities))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

        # Exclude items with a similarity score of 0 (no match) and get the top N
        top_similar_items = [item for item in similar_items if item[1] > 0][:top_n]

        # Get the indices of the top similar items
        recommended_item_indices = [x[0] for x in top_similar_items]

        # Get the details of the top similar items
        recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

        return recommended_items_details

  def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
      # Create the user-item matrix
      user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)

      # Calculate the user similarity matrix using cosine similarity
      user_similarity = cosine_similarity(user_item_matrix)
      target_user_index = user_item_matrix.index.get_loc(target_user_id)
      # Get the similarity scores for the target user
      user_similarities = user_similarity[target_user_index]

      # Sort the users by similarity in descending order (excluding the target user)
      similar_users_indices = user_similarities.argsort()[::-1][1:]

      # Generate recommendations based on similar users
      recommended_items = []

      for user_index in similar_users_indices:
          # Get items rated by the similar user but not by the target user
          rated_by_similar_user = user_item_matrix.iloc[user_index]
          not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)

          # Extract the item IDs of recommended items
          recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

      # Get the details of recommended items
      recommended_items_details = train_data[train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

      return recommended_items_details.head(10)

  # Hybrid Recommendations (Combine Content-Based and Collaborative Filtering)
  def hybrid_recommendations(train_data,target_user_id, item_name, top_n=10):
      # Get content-based recommendations
      content_based_rec = content_based_recommendations(train_data,item_name, top_n)

      # Get collaborative filtering recommendations
      collaborative_filtering_rec = collaborative_filtering_recommendations(train_data,target_user_id, top_n)

      # Merge and deduplicate the recommendations
      hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()

      return hybrid_rec.head(10)

  hybrid_rec = hybrid_recommendations(train_data,id, desc, top_n=10)

  #print(f"Top 10 Hybrid Recommendations for User {id} and Item '{desc}':")
  return hybrid_rec

def recnewuser(new_product_tags):
  import numpy as np
  import pandas as pd
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.cluster import KMeans

  # Function to get top keywords per cluster
  def get_top_keywords_per_cluster(tfidf_matrix, cluster_labels, tfidf_feature_names, top_n=10):
      # Group the documents by cluster and calculate the average tf-idf score for each word in each cluster
      df = pd.DataFrame(tfidf_matrix.todense()).groupby(cluster_labels).mean()

      # For each cluster, get the top n words with the highest average tf-idf score
      top_keywords = {}
      for i, row in df.iterrows():
          top_keywords[i] = [tfidf_feature_names[j] for j in np.argsort(row)[-top_n:]]

      return top_keywords

  text_data = pd.read_csv('marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv', sep='\t')
  text_data['Product Rating'] = text_data['Product Rating'].fillna(0)
  # Fill missing values in 'Product Reviews Count' with a default value (e.g., 0)
  text_data['Product Reviews Count'] = text_data['Product Reviews Count'].fillna(0)
  # Fill missing values in 'Product Category' with a default value (e.g., 'Unknown')
  text_data['Product Category'] = text_data['Product Category'].fillna('')
  # Fill missing values in 'Product Brand' with a default value (e.g., 'Unknown')
  text_data['Product Brand'] = text_data['Product Brand'].fillna('')
  # Fill missing values in 'Product Description' with an empty string
  text_data['Product Description'] = text_data['Product Description'].fillna('')
  tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
  tfidf_matrix = tfidf_vectorizer.fit_transform(text_data['Product Tags'])
  from sklearn.metrics import silhouette_score
  import matplotlib.pyplot as plt
  silhouette_scores = []
  cluster_range = range(2, 11)

  for num_clusters in cluster_range:
      kmeans = KMeans(n_clusters=num_clusters, random_state=42)
      kmeans.fit(tfidf_matrix)
      cluster_labels = kmeans.labels_
      silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
      silhouette_scores.append(silhouette_avg)

  # The optimal number of clusters is the one with the highest silhouette score
  optimal_clusters_silhouette = cluster_range[silhouette_scores.index(max(silhouette_scores))]
  num_clusters = optimal_clusters_silhouette
  kmeans = KMeans(n_clusters=num_clusters, random_state=42)
  kmeans.fit(tfidf_matrix)
  tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
  top_keywords_per_cluster = get_top_keywords_per_cluster(tfidf_matrix, kmeans.labels_, tfidf_feature_names, top_n=10)

  #new_product_tags = "shampoo, hair care, natural oils, moisturizing"
  new_tfidf_vector = tfidf_vectorizer.transform([new_product_tags])
  predicted_cluster = kmeans.predict(new_tfidf_vector)[0]
  cluster_labels = kmeans.labels_
  text_data['cluster'] = cluster_labels
  products_in_same_cluster = text_data[text_data['cluster'] == predicted_cluster]
  same_cluster_products_info = products_in_same_cluster[['Product Name', 'Product Reviews Count', 'Product Brand', 'Product Image Url', 'Product Rating']]
  same_cluster_products_info.columns = ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
  return same_cluster_products_info.head()