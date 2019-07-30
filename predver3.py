import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import mnist

frame_size = 100  #何点を見るかのサイズ
ema_max = 75

#試してみたいこと
#emaの詳しい値を消して。パーフェクトオーダーのラベルだけを特徴量にしてみる

def get_feature():
	#元のcsv取得
	whole_df = pd.read_csv('./EUR_USD_H1.csv')
	close_s = whole_df['close']
	#volume = whole_df['volume']

	#指数平滑移動平均線取得
	ema21_s = close_s.ewm(span=21).mean()
	ema34_s = close_s.ewm(span=34).mean()
	ema55_s = close_s.ewm(span=55).mean()
	ema75_s = close_s.ewm(span=75).mean()

	#emaをデータフレームに追加する
	whole_df['ema21'] = ema21_s
	whole_df['ema34'] = ema34_s
	whole_df['ema55'] = ema55_s
	whole_df['ema75'] = ema75_s

	#0~74点のデータを削除（最大ema期間の75が正しく適用できるように）
	for i in range(ema_max):
		whole_df = whole_df.drop(i,axis=0)
	whole_df = whole_df.reset_index(drop=True) #インデックスが75からになってしまうから、0からに振りなおす

	#パーフェクトオーダーのラベル作成（上昇:1, 下降:2, その他:0）
	perfect = []
	for i in range(len(whole_df)):
		ema21 = whole_df.at[whole_df.index[i],'ema21'] #行と列を指定して、要素を取得する
		ema34 = whole_df.at[whole_df.index[i],'ema34']
		ema55 = whole_df.at[whole_df.index[i],'ema55']
		ema75 = whole_df.at[whole_df.index[i],'ema75']
		if ema21 > ema34 and ema34 > ema55 and ema55 > ema75:
			perfect.append(1)
		elif ema75 > ema55 and ema55 > ema34 and ema34 > ema21:
			perfect.append(2)
		else:
			perfect.append(0)

	perfect_s = pd.Series(perfect)
	whole_df['perfect'] = perfect_s #データフレームにパーフェクトオーダーラベルを追加
	del whole_df['comp']
	del whole_df['time']
	return whole_df

def make_feature(whole_df):
	#100点の終値を1つのシーケンスとしたデータセットを作成する
	filepath = './EUR_USD_H1_predver3.csv'
	if os.path.exists(filepath):
		feature_df = pd.read_csv(filepath,index_col=0)
	else:
		data_size = whole_df.shape[0]
		close_df = pd.DataFrame()
		for i in range(data_size-frame_size):
			close_s = whole_df.loc[i:i+frame_size-1,'close']
			close_s = close_s.reset_index(drop=True)
			#ここに特徴量を追加していく
			close_s[frame_size] = whole_df.loc[i+frame_size-1,'ema21']
			close_s[frame_size+1] = whole_df.loc[i+frame_size-1,'ema34']
			close_s[frame_size+2] = whole_df.loc[i+frame_size-1,'ema55']
			close_s[frame_size+3] = whole_df.loc[i+frame_size-1,'ema75']
			close_s[frame_size+4] = whole_df.loc[i+frame_size-1,'perfect']
			close_df[i] = close_s
			print(i)
		close_df_T = close_df.T #転置して、行方向に100点を並べる
		close_df_T.to_csv(filepath)
		feature_df = pd.read_csv(filepath,index_col=0)

	return feature_df

def make_target(feature_df):
	close_target = feature_df[str(frame_size-1)].copy()
	close_target = close_target.drop(0)
	close_target = close_target.reset_index(drop=True)
	return close_target

def dnn_model():
	inputs = Input(shape=(105,))
	x = Dense(400,activation='relu')(inputs)
	x = Dense(200,activation='relu')(x)
	x = Dense(100,activation='relu')(x)
	x = Dense(50,activation='relu')(x)
	prediction = Dense(1,activation='linear')(x)
	model = Model(input=inputs,output=prediction)
	optimizer = Adam()
	model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])
	model.summary()

	return model


whole_df = get_feature()
feature_df = make_feature(whole_df)
close_target = make_target(feature_df)
feature_df = feature_df.drop(feature_df.shape[0]-1,axis=0) #targetに合わせて特徴量の長さを変更する

features = pd.concat([feature_df,close_target],axis=1)
features = features.T.reset_index(drop=True).T #列名をリセットする方法

#データ全体をtrainとtestに分ける(正規化をするため、ターゲットはまだくっつけたまま)
data_train, data_test = train_test_split(features, train_size=0.8, shuffle=False)

#minmax正規化のインスタンス生成
scaler = MinMaxScaler(feature_range=(-1, 1))

#正規化スケールを作成(学習データに合わせなければいけない)
scaler.fit(data_train)

#正規化(numpyに変換されることに注意)
data_train_norm = scaler.transform(data_train)
data_test_norm = scaler.transform(data_test)

#特徴量とターゲット変数を分ける
X_train = data_train_norm[:,:-1]
y_train = data_train_norm[:,-1]
X_test = data_test_norm[:,:-1]
y_test = data_test_norm[:,-1]

#学習
model = dnn_model()
history = model.fit(X_train, y_train, 
					batch_size=1000, 
					epochs=50, 
					validation_split=0.2,
					validation_data=(X_test,y_test))
score = model.evaluate(X_test,y_test,verbose=0)
print('Test loss',score[0])
print('Test accuracy',score[1])

#予測
predicted = model.predict(X_test)

#予測値の正規化を直す
predicted_inv = np.concatenate((X_test,predicted),axis=1)
predicted_inv = scaler.inverse_transform(predicted_inv)

#評価するためのテストデータの正規化を直す
temp = y_test.reshape(len(y_test),1)
data_test_inv = np.concatenate((X_test,temp),axis=1)
data_test_inv = scaler.inverse_transform(data_test_inv)

print(data_test_inv)
print(predicted_inv)

#結果表示
fig = plt.figure()
plt.plot(data_test_inv[10000:10600,105])
plt.plot(predicted_inv[10000:10600,105])
#line1, = plt.plot(a[16000:16800,4])
#line2, = plt.plot(predicted_inv[16000:16800,4])
plt.show()

#Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()