from math import nan
import pandas as pd
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, dot
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

def data():
    data = [
        [1, '1', 1],
        [1, '2', 0],
        [1, '4', 1],
        [1, '5', 1],
        [1, '7', 0], 
        [2, '1', 1],
        [2, '2', 1],
        [2, '5', 1],
        [2, '6', 0],
        [3, '1', 1],
        [3, '2', 1],
        [3, '3', 0],
        [3, '4', 1], 
        [4, '1', 1], 
        [4, '2', 1],
        [4, '3', 0],
        [4, '4', 0],
        [4, '5', 1],
        [5, '1', 1],
        [5, '2', 0],
        [5, '3', 0],
        [5, '7', 0], 
        [6, '1', 1], 
        [6, '2', 1],
        [6, '3', 0],
        [6, '4', 0],
        [6, '7', 1],
        [7, '1', 1],
        [7, '3', 0],
        [7, '4', 1],
        [7, '5', 1],
        [7, '6', 0], 
        [7, '7', 1],
        [8, '2', 1],
        [8, '5', 1],
        [8, '6', 0],
        [9, '1', 1],
        [9, '2', 1],
        [9, '3', 0],
        [9, '5', 1], 
        [10, '1', 1], 
        [10, '2', 1],
        [10, '3', 0],
        [10, '4', 0],
        [10, '5', 1],
        [11, '1', 1],
        [11, '2', 0],
        [11, '3', 0],
        [11, '7', 0], 
        [12, '1', 1], 
        [12, '3', 1],
        [12, '4', 0],
        [12, '5', 0],
        [12, '7', 1],
        [13, '2', 1],
        [13, '3', 0],
        [13, '4', 1],
        [13, '5', 1],
        [13, '6', 0], 
        [14, '1', 1],
        [14, '2', 1],
        [14, '3', 1],
        [14, '6', 0],
        [15, '2', 1],
        [15, '3', 1],
        [15, '4', 0],
        [16, '1', 1], 
        [16, '2', 1], 
        [16, '3', 1],
        [16, '4', 0],
        [16, '5', 0],
        [16, '6', 1],
        [16, '7', 1],
        [17, '1', 0],
        [17, '3', 0],
        [17, '5', 0], 
        [17, '6', 1], 
        [17, '7', 1],
        [18, '1', 0],
        [18, '2', 0],
        [18, '3', 1],
        [18, '4', 1],
        [18, '5', 0],
        [18, '6', 1],
        [18, '7', 1],
        [19, '1', 0], 
        [19, '3', 1],
        [20, '2', 1],
        [20, '3', 1],
        [20, '4', 0],
        [20, '6', 1],
        [21, '1', 1],
        [21, '2', 0],
        [21, '3', 1], 
        [21, '4', 1], 
        [21, '5', 1],
        [21, '6', 0],
        [21, '7', 0],
        [22, '1', 1],
        [22, '2', 1],
        [22, '3', 0],
        [22, '4', 0],
        [22, '5', 0], 
        [22, '6', 1], 
        [23, '2', 1],
        [23, '3', 0],
        [23, '5', 0],
        [23, '6', 1]


    ]

    return data


def table(data):

    mini_df=pd.DataFrame(data=data,columns=['user','item','like'])

    return mini_df

def table2(mini_df):

    df_table=pd.pivot_table(mini_df,values='like',index='item',columns='user',fill_value='')

    return df_table

def learning(mini_df, df_table):
       
    #숫자로 변환 과정 
    mini_df.item=mini_df.item.astype('category').cat.codes.values
    #데이터 변환 
    users=mini_df.user.unique()
    items=mini_df.item.unique()

    userid2idx={o:i for i,o in enumerate(users)}
    itemid2idx={o:i for i,o in enumerate(items)}

    mini_df['user']=mini_df['user'].apply(lambda x : userid2idx[x])
    mini_df['item']=mini_df['item'].apply(lambda x : itemid2idx[x])

    '''훈련, 테스트 데이터 분할'''
    split=np.random.rand(len(mini_df))<0.8
    train_df=mini_df[split]
    test_df=mini_df[~split]

    print('shape of train data is',train_df.shape)
    print('shape of test data is ',test_df.shape)

    '''임베딩'''
    n_items=len(mini_df.item.unique())
    n_users=len(mini_df.user.unique())
    n_latent_factors=64

    item_input=Input(shape=[1])
    user_input=Input(shape=[1])

    item_embedding = Embedding(n_items, n_latent_factors, 
                            embeddings_regularizer=regularizers.l2(0.00001), #regularizers.l2(0.001) : 가중치 행렬의 모든 원소를 제곱하고 0.001을 곱하여 네트워크의 전체 손실에 더해진다는 의미, 이 규제(패널티)는 훈련할 때만 추가됨
                            name='item_embedding')(item_input)
    user_embedding = Embedding(n_users, n_latent_factors, 
                            embeddings_regularizer=regularizers.l2(0.00001),
                            name='user_embedding')(user_input)

    item_vec=Flatten()(item_embedding) #잠재 벡터 2D->1D
    user_vec=Flatten()(user_embedding)

    r_hat=dot([item_vec,user_vec],axes=-1)
    model=Model([user_input,item_input],r_hat)
    model.compile(optimizer='adam',loss='mean_squared_error') #평균 제곱 오차

    history = model.fit([mini_df.user, mini_df.item], mini_df.like, batch_size=128, epochs=4000, verbose=0) 
    print('loss: ',history.history['loss'][-1]) #a[-1]: 오른쪽에서 첫번째 값

    plt.plot(history.history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    '''예측'''
    Q = model.get_layer(name='item_embedding').get_weights()[0]
    P = model.get_layer(name='user_embedding').get_weights()[0]
    P_t = np.transpose(P)

    R_hat = np.dot(Q, P_t)
    pred_rec=pd.DataFrame(R_hat)
    pd.options.display.float_format='{:.2f}'.format

    return pred_rec

def unseen(rating_mat,user):
    rating_mat=rating_mat.replace('',np.NaN)
    user_rating=rating_mat.loc[:,user]
    user_unseen_list=user_rating[user_rating.isnull()].index.tolist()
    recipe_list=rating_mat.index.tolist()
    unseen_list=[recipe for recipe in recipe_list if recipe in user_unseen_list]
    for i in range(len(unseen_list)):
        unseen_list[i]=str(int(unseen_list[i])-1)

    return unseen_list

def recommend_recipe_by_userID(pred_rec,userID, unseen_list,top_n):
    unseen_list=map(int,unseen_list)
    recomm_recipe=pred_rec.loc[unseen_list,userID-1].sort_values(ascending=False)[:top_n]

    return recomm_recipe

if __name__=="__main__":
    userID=int(input("userID를 입력하시오:"))
    data=data()
    data1=table(data)
    data2=table2(data1)
    pred_rec=learning(data1,data2)
    data4=unseen(data2,userID)
    print(recommend_recipe_by_userID(pred_rec,userID,data4,3))
