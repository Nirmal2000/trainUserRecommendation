import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from tensorflow.keras.layers import Dense,Conv1D,MaxPool1D,Embedding,Dropout,Input,GlobalMaxPool1D,Flatten,Concatenate,LSTM,Dot,Reshape
from tensorflow.keras.optimizers import Adagrad
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
ps = PorterStemmer()
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l1, l2, l1_l2
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from scipy import spatial
import random
import pickle
import re
from numpy import dot
from pymongo import MongoClient
from numpy.linalg import norm
import networkx as nx
from networkx.algorithms import bipartite
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph import StellarGraph,StellarDiGraph
import requests

class Sentence2Vec:
    def __init__(self, model_file):
        self.load(model_file)

    def load(self, model_file):
        self.model = Word2Vec.load(model_file)

    def get_vector(self, sentence):        
        sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())

        vectors = [self.model.wv[w] for w in word_tokenize(sentence)
                   if w in self.model.wv]

        v = np.zeros(self.model.vector_size)

        if (len(vectors) > 0):
            v = (np.array([sum(x) for x in zip(*vectors)])) / v.size

        return v

    def similarity(self, x, y):
        xv = self.get_vector(x)
        yv = self.get_vector(y)

        score = 0

        if xv.size > 0 and yv.size > 0:
            score = dot(xv, yv) / (norm(xv) * norm(yv))

        return score

def clean_text(text): 
  try:
    text = text.lower()    
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text
  except:    
    return ' '


def model_train():
  print("Loading files..")
  cluster = MongoClient('mongodb+srv://nirmal:2000@cluster0-2aasp.mongodb.net/<dbname>?retryWrites=true&w=majority')

  db = cluster.Dataset
  pcol = db.posts  
  vcol= db.views
  fcol= db.follows
  favcol = db.favourites  


  # posts = pd.read_csv('./posts.csv',engine='python')
  # users = pd.read_csv('./users.csv')
  # views = pd.read_csv('./views.csv')
  # favorites = pd.read_csv('./favourites.csv')
  # userPosts = pd.read_csv('./usersPosts.csv')
  # print("Files loaded..")

  ''' Create DataFrame for preprocessing '''
  posts = pd.DataFrame(list(pcol.find()))  
  views = pd.DataFrame(list(vcol.find()))
  favorites = pd.DataFrame(list(favcol.find()))
  userPosts = posts[['_id','postedBy']]
  follows = pd.DataFrame(list(fcol.find()))
  print("Collections loaded..")

  print("Started preprocessing..")

  views = views[views['user_id']!='anonymous']
  posts = posts.dropna(subset=['title',' post_type','tags'])

  posts['category'] = posts['category'].fillna(posts['tags'])
  posts['tags'] = posts['tags'].apply(clean_text)

  """Splitting on '|' and '#' for getting categories"""

  uniq_category = dict()
  uniq_post_type = dict()
  i=0
  j=0
  for cats,pt in zip(posts['category'].values,posts[' post_type'].values):
    for cat in re.split('[#|]',cats):
      if cat not in uniq_category.keys():
        uniq_category[cat]=i
        i+=1
    if pt not in uniq_post_type.keys():
      uniq_post_type[pt]=j
      j+=1

  category_ohe = np.zeros((len(posts),513))

  for i,cats in enumerate(posts['category'].values):
    for cat in re.split('[#|]',cats):
      category_ohe[i][uniq_category[cat]]=1

  token_tag = [word_tokenize(tag) for tag in posts['tags'].values.tolist()]
  tag_model = Word2Vec(token_tag,sg=1,size=100,window=5, min_count=5, workers=4,iter=100)
  tag_model.save('./tag.model')

  tag_model = Sentence2Vec('./tag.model')

  processed_title = posts['title'].apply(clean_text)
  token_title = [word_tokenize(tag) for tag in processed_title]
  title_model = Word2Vec(token_title,sg=1,size=100,window=5, min_count=5, workers=4,iter=100)
  title_model.save('./title.model')

  title_model = Sentence2Vec('./title.model')

  posts_info = dict()
  for pid,title,cat,tag in zip(posts['_id'],posts['title'].values,category_ohe,posts['tags'].values):
    posts_info[pid] = dict()
    posts_info[pid]['title'] = title_model.get_vector(title)
    posts_info[pid]['tag'] = tag_model.get_vector(tag)
    posts_info[pid]['cat'] = cat

  """Removing rows in views.csv, favorites.csv and usrPosts.csv
  that has pid not present in posts.csv
  """

  pidr=set()
  for pid in views['post_id']:
    if posts_info.get(pid,0) == 0:
      pidr.add(pid)
  for pid in favorites['post_id']:
    if posts_info.get(pid,0) == 0:
      pidr.add(pid)
  for pid in userPosts['post_id']:
    if posts_info.get(pid,0) == 0:
      pidr.add(pid)
  
  for pid in list(pidr):  
    views = views[views['post_id']!=pid]
    userPosts = userPosts[userPosts['post_id']!=pid]
    favorites = favorites[favorites['post_id']!=pid]

  """Representing the user based on the categories seen by the user"""

  users_info = defaultdict(lambda :np.zeros((513)))
  for uid,pid in zip(views['user_id'],views['post_id']):    
    a = posts_info[pid]['cat'] #,posts_info[pid]['pt']))#,posts_info[pid]['title_ohe']))
    users_info[uid] = np.add(users_info[uid],a)
    assert(np.sum(users_info[uid])!=0)

  """Increasing the weightage for categories by 100% for posts posted by user"""

  for uid,pid in zip(userPosts['user_id'],userPosts['post_id']):    
    a = posts_info[pid]['cat'] #,posts_info[pid]['pt']))#,posts_info[pid]['title_ohe']))
    users_info[uid] = np.add(users_info[uid],a)
    assert(np.sum(users_info[uid])!=0)

  """Increasing weightage for categories by 50% for favorite posts"""

  for uid,pid in zip(favorites['user_id'],favorites['post_id']):    
    a = 1/2*posts_info[pid]['cat'] #,posts_info[pid]['pt'])))#,posts_info[pid]['title_ohe'])))
    users_info[uid] = np.add(users_info[uid],a)
    assert(np.sum(users_info[uid])!=0)

  """## MODEL 

  Generating -ive datapoints for each user where the posts chosen have categories that are not seen by the user
  """

  def gen_pseudoDP(user_id):
    cat_user = users_info[uid]
    arr=[]
    k=0
    for pid in posts_info.keys():
      cat = posts_info[pid]['cat']
      flag=0
      for i in range(len(cat)):
        if (cat[i]!=0 and cat_user[i] != 0):        
          flag=1
          break    
      if flag==0:
        arr.append([uid,pid,0])
        k+=1
      if k==4:
        break
    return arr

  pseudo = pd.DataFrame(np.zeros((len(users_info)*4,3)),columns=['user_id','post_id','view'])
  i=0
  for uid in list(users_info.keys()):
    arr = gen_pseudoDP(uid)  
    if len(arr):
      pseudo[i:i+len(arr)] = arr
      i+=4

  views['view'] = np.ones((len(views)))
  views = views.drop(columns=['timestamp'],axis=1)
  data = views.append(pseudo)

  print("Preprocessing done!")

  class Datagenerator(tf.keras.utils.Sequence):
    def __init__(self,X,y=None,batch_size=1,shuffle=True):
      super().__init__()
      self.X = X
      self.y = y
      self.batch_size = batch_size    
      self.on_epoch_end()
      

    def __getitem__(self,index):
      
      indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]    
      batch = self.X.iloc[indices]    
      y = self.y.iloc[indices]
      
      user=np.zeros((self.batch_size,len(uniq_category)))    
      title=np.zeros((self.batch_size,100))
      tag=np.zeros((self.batch_size,100))
      category = np.zeros((self.batch_size,len(uniq_category)))
      
      for i in range(self.batch_size): 
        title[i] = posts_info[batch.post_id.values[i]]['title']
        tag[i] = posts_info[batch.post_id.values[i]]['tag']
        category[i] = posts_info[batch.post_id.values[i]]['cat']
        user[i] = users_info[batch['user_id'].values[i]]
      
      return [user,title,tag,category],y.values.reshape(-1,1)
      

    def __len__(self):
      return int(np.floor(len(self.X) / self.batch_size))

    def on_epoch_end(self):    
      self.indexes = np.arange(len(self.X))
      np.random.shuffle(self.indexes)

  y = data['view']
  X = data.drop(['view'],axis=1)

  X_train, X_test, y_train, y_test = train_test_split(X,y)

  train_dg = Datagenerator(X_train,y_train,128)
  test_dg = Datagenerator(X_test,y_test,128)

  """Model predicts whether a user will see a post or not. Based on that user embeddings will be learnt which will then be used for recommendation"""

  def create_model():

    user_inp = Input((len(uniq_category)))
    embed = Embedding(input_dim=len(uniq_category),output_dim=50)(user_inp)
    dense = Dense(2056)(Flatten()(embed))
    user = Dense(500,activation='relu')(dense)
    user = Dense(400,activation='relu')(user)

    cat = Input((len(uniq_category)))
    cat_ = Dense(300,activation='relu')(cat)

    title = Input((100))
    title_ = Dense(50,activation='relu')(title)
    tag = Input((100))
    tag_ = Dense(50,activation='relu')(tag)

    post_concat = Concatenate()([cat_,title_,tag_])

    output = Dot(axes=[-1,-1],normalize=True)([user,post_concat])    

    model = tf.keras.Model([user_inp,title,tag,cat],output)

    return model

  model = create_model()

  model.compile(optimizer=Adagrad(lr=0.0001), loss='binary_crossentropy',metrics=['accuracy'])

  print("model started training...")
  model.fit_generator(train_dg,validation_data=test_dg,epochs=1)
  print("Model trained")

  """Retrieving trained user embeddings"""

  user_embeddings = model.get_layer('embedding').get_weights()[0]

  follows = pd.read_csv('./follows.csv')

  follows = follows.drop(['timestamp'],axis=1)

  """Users present in follows.csv"""

  uids = np.concatenate((follows['user_id'].values,follows['follower_id'].values))
  uids = set(uids)

  """Creating Edges"""

  edges = [(y,x) for x,y in zip(follows['user_id'],follows['follower_id'])]

  """Creating Directional Graph and adding the edges"""

  G = nx.DiGraph()
  G.add_edges_from(edges)

  edges_dict = dict()
  for edge in edges:
    edges_dict[edge]=1

  rw = UniformRandomMetaPathWalk(StellarDiGraph(G))

  """Creating random walks.

  Each walk can be seen as a chain:  uid->uid->uid ... 

  They are of length 100
  """

  walks = rw.run(nodes=list(uids),length=100,n=2,metapaths=[['default','default']])

  """Word2Vec on those chains"""

  user_model =  Word2Vec(walks,size=128,window=5)
  user_model.wv.vectors.shape

  """Each user represented by 128 dim vector"""

  node_ids = user_model.wv.index2word
  node_embed = user_model.wv.vectors

  print("Pushing to database...")
  userCollection = cluster.Users.User_Embeddings
  userCollection.delete_many({})
  followCollection = cluster.Users.Follows
  followCollection.delete_many({})
  posted = cluster.Users.Posted
  posted.delete_many({})
  catCol = cluster.Users.Categories
  catCol.delete_many({})
  embedCol = cluster.Users.Embedding_Matrix
  embedCol.delete_many({})

  folDict = dict()
  for i,id in enumerate(node_ids):
    folDict[id]=i

  user_ins=[]
  for user in tqdm(users_info.keys()):
    embed = list(np.matmul(users_info[user],user_embeddings))
    if folDict.get(user,-1) == -1:
      #userCollection.insert_one({'user_id':user, 'user_embed':embed})
      user_ins.append({'user_id':user, 'user_embed':embed})
    else:
      yo = node_embed[folDict[user]].tolist()
      #userCollection.insert_one({'user_id':user, 'user_embed':embed, 'node_embed':yo})
      user_ins.append({'user_id':user, 'user_embed':embed, 'node_embed':yo})

  userCollection.insert_many(user_ins)

  fol=[]
  for uid,fid in tqdm(zip(follows['user_id'],follows['follower_id'])):
      d = dict()
      d['user_id'] = uid
      d['follower_id'] = fid
      fol.append(d)

  followCollection.insert_many(fol)

  categories = pickle.dumps(uniq_category)
  user_embed = pickle.dumps(user_embeddings)

  catCol.insert_one({"Categories":categories})
  embedCol.insert_one({"Matrix":user_embed})


  uids = set()
  for uid in userPosts['user_id']:
      uids.add(uid)
  to_ins=[]
  for uid in uids:
      noob = dict()
      noob['user_id']=uid    
      to_ins.append(noob)

  posted.insert_many(to_ins)
  requests.get('http://3.7.185.166/train')
  print("Done!")
