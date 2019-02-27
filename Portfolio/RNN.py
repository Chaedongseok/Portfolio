
import tensorflow as tf
import numpy as np
import csv

import matplotlib.pyplot as plt

def MinMaxScaler(data):

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def reMinMaxScaler(data,test_prediction):
 
    numerator = test_prediction
    denominator = (np.max(data, 0) - np.min(data, 0))[0]
    # noise term prevents the zero division
    return np.min(data , 0)[0]+numerator*denominator
def LoadData(url):
    f = open(url,'r',encoding = 'utf-8')
    rdr = csv.reader(f)
    data = list(rdr)
    data = data[1:len(data)]
    xy = np.array(data)
    xy = xy[1:len(xy),1:]
    return xy
    
def MakeData(data,seq_length,test_data_size):
    for i in range(0,np.size(data,axis=0)):
        for j in range(0,np.size(data,axis=1)):
            if data[i,j]=='':
                print(i)
                data[i,j] = data[i-1,j]
    data = data.astype(float)
    
    
    # Open, High, Low, Volume, Close
    xy_mm = MinMaxScaler(data)
    x = xy_mm[:,1:]
    y = xy_mm[:, [0]]  # Close as label
    
    dataX = []
    dataY = []
    for i in range(0, len(y) - seq_length):
        _x = x[i:i + seq_length]
        _y = y[i]  # Next close price
        #print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)

    train_size = np.size(dataX, axis = 0)-test_data_size
    trainX = np.array(dataX[0:train_size])
    trainY = np.array(dataY[0:train_size])
    testX = np.array(dataX[train_size:len(dataX)])
    testY = np.array(dataY[train_size:len(dataY)])
    
    return data,trainX,trainY,testX,testY

def regionMakeData(xy,region,test_data_size,seq_length):
    region = np.append(xy,region)
    region = np.reshape(region,[-1,6])
    for i in range(0,np.size(region,axis=0)):
        for j in range(0,np.size(region,axis=1)):
            if region[i,j]=='':
                print(i)
                region[i,j] = region[i-1,j]
    region = region.astype(float)
    
    region = region[(len(region)-test_data_size-seq_length):len(region),:]
    return region
##-------------------------------------------------------------------------------------------------


xy = LoadData(r'C:\Users\student\Desktop\multiproj\multiproj_data.csv')
region_data = LoadData(r'C:\Users\student\Desktop\multiproj\region_data.csv')
#강남,강동,강북,관악,광진,구로,금천,노원,동대문,동자국,마포구,서대문구,성북구,
#송파구,양천구,영등포구,용산구,은평구,종로구,중구,중랑구
seq_length = 24#24
data_dim = np.size(xy,axis=1)
hidden_dim = 7#7
output_dim = 1
learning_rate = 0.02
iterations = 5000
test_data_size = 24

GN = region_data[seq_length*0:seq_length*1]
GD = region_data[seq_length*1:seq_length*2]
GP = region_data[seq_length*2:seq_length*3]
GA = region_data[seq_length*3:seq_length*4]
GJ = region_data[seq_length*4:seq_length*5]
GR = region_data[seq_length*5:seq_length*6]
KC = region_data[seq_length*6:seq_length*7]
NW = region_data[seq_length*7:seq_length*8]
DDM = region_data[seq_length*8:seq_length*9]
DJK = region_data[seq_length*9:seq_length*10]
MP = region_data[seq_length*10:seq_length*11]
SDM = region_data[seq_length*11:seq_length*12]
SPK = region_data[seq_length*12:seq_length*13]
SP = region_data[seq_length*13:seq_length*14]
YC = region_data[seq_length*14:seq_length*15]
YDP = region_data[seq_length*15:seq_length*16]
YS = region_data[seq_length*16:seq_length*17]
EP = region_data[seq_length*17:seq_length*18]
JL = region_data[seq_length*18:seq_length*19]
J = region_data[seq_length*19:seq_length*20]
JR = region_data[seq_length*20:seq_length*21]

GN=regionMakeData(xy,GN,test_data_size,seq_length)
GD=regionMakeData(xy,GD,test_data_size,seq_length)
GP=regionMakeData(xy,GP,test_data_size,seq_length)
GA=regionMakeData(xy,GA,test_data_size,seq_length)
GJ=regionMakeData(xy,GJ,test_data_size,seq_length)
GR=regionMakeData(xy,GR,test_data_size,seq_length)
KC=regionMakeData(xy,KC,test_data_size,seq_length)
NW=regionMakeData(xy,NW,test_data_size,seq_length)
DDM=regionMakeData(xy,DDM,test_data_size,seq_length)
DJK=regionMakeData(xy,DJK,test_data_size,seq_length)
MP=regionMakeData(xy,MP,test_data_size,seq_length)
SDM=regionMakeData(xy,SDM,test_data_size,seq_length)
SPK=regionMakeData(xy,SPK,test_data_size,seq_length)
SP=regionMakeData(xy,SP,test_data_size,seq_length)
YC=regionMakeData(xy,YC,test_data_size,seq_length)
YDP=regionMakeData(xy,YDP,test_data_size,seq_length)
YS=regionMakeData(xy,YS,test_data_size,seq_length)
EP=regionMakeData(xy,EP,test_data_size,seq_length)
JL=regionMakeData(xy,JL,test_data_size,seq_length)
J=regionMakeData(xy,J,test_data_size,seq_length)
JR=regionMakeData(xy,JR,test_data_size,seq_length)


xy,trainX,trainY,testX,testY=MakeData(xy,seq_length,0)
test_size=24

_,_,_,testGNX,testGNY=MakeData(GN,seq_length,test_size)
_,_,_,testGDX,testGDY=MakeData(GD,seq_length,test_size)
_,_,_,testGPX,testGPY=MakeData(GP,seq_length,test_size)
_,_,_,testGAX,testGAY=MakeData(GA,seq_length,test_size)
_,_,_,testGJX,testGJY=MakeData(GJ,seq_length,test_size)
_,_,_,testGRX,testGRY=MakeData(GR,seq_length,test_size)
_,_,_,testKCX,testKCY=MakeData(KC,seq_length,test_size)
_,_,_,testNWX,testNWY=MakeData(NW,seq_length,test_size)
_,_,_,testDDMX,testDDMY=MakeData(DDM,seq_length,test_size)
_,_,_,testDJKX,testDJKY=MakeData(DJK,seq_length,test_size)
_,_,_,testMPX,testMPY=MakeData(MP,seq_length,test_size)
_,_,_,testSDMX,testSDMY=MakeData(SDM,seq_length,test_size)
_,_,_,testSPKX,testSPKY=MakeData(SPK,seq_length,test_size)
_,_,_,testSPX,testSPY=MakeData(SP,seq_length,test_size)
_,_,_,testYCX,testYCY=MakeData(YC,seq_length,test_size)
_,_,_,testYDPX,testYDPY=MakeData(YDP,seq_length,test_size)
_,_,_,testYSX,testYSY=MakeData(YS,seq_length,test_size)
_,_,_,testEPX,testEPY=MakeData(EP,seq_length,test_size)
_,_,_,testJLX,testJLY=MakeData(JL,seq_length,test_size)
_,_,_,testJX,testJY=MakeData(J,seq_length,test_size)
_,_,_,testJRX,testJRY=MakeData(JR,seq_length,test_size)

# input place holders
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, seq_length, data_dim-1])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network

cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, 0], output_dim, activation_fn=None)  ## # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss(SSE): {}".format(i, step_loss))
 #   if i%1==0:
 #       print("[step: {}] loss: {}".format(i, step_loss))
    print("1")
    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    predict_GN = sess.run(Y_pred, feed_dict={X: testGNX})
    predict_GD= sess.run(Y_pred, feed_dict={X: testGDX})
    predict_GP= sess.run(Y_pred, feed_dict={X: testGPX})
    predict_GA= sess.run(Y_pred, feed_dict={X: testGAX})
    predict_GJ= sess.run(Y_pred, feed_dict={X: testGJX})
    predict_GR= sess.run(Y_pred, feed_dict={X: testGRX})
    predict_KC= sess.run(Y_pred, feed_dict={X: testKCX})
    predict_NW= sess.run(Y_pred, feed_dict={X: testNWX})
    predict_DDM= sess.run(Y_pred, feed_dict={X: testDDMX})
    predict_DJK = sess.run(Y_pred, feed_dict={X: testDJKX})
    predict_MP= sess.run(Y_pred, feed_dict={X: testMPX})
    predict_SDM= sess.run(Y_pred, feed_dict={X: testSDMX})
    predict_SPK= sess.run(Y_pred, feed_dict={X: testSPKX})
    predict_SP= sess.run(Y_pred, feed_dict={X: testSPX})
    predict_YC= sess.run(Y_pred, feed_dict={X: testYCX})
    predict_YDP= sess.run(Y_pred, feed_dict={X: testYDPX})
    predict_YS= sess.run(Y_pred, feed_dict={X: testYSX})
    predict_EP= sess.run(Y_pred, feed_dict={X: testEPX})
    predict_JL= sess.run(Y_pred, feed_dict={X: testJLX})
    predict_J= sess.run(Y_pred, feed_dict={X: testJX})
    predict_JR= sess.run(Y_pred, feed_dict={X: testJRX})
    
    
    # reminmaxscaler
    test_predict = reMinMaxScaler(xy,test_predict)
    
    predict_GN = reMinMaxScaler(xy,predict_GN)
    predict_GD = reMinMaxScaler(xy,predict_GD)
    predict_GP = reMinMaxScaler(xy,predict_GP)
    predict_GA = reMinMaxScaler(xy,predict_GA)
    predict_GJ = reMinMaxScaler(xy,predict_GJ)
    predict_GR = reMinMaxScaler(xy,predict_GR)
    predict_KC = reMinMaxScaler(xy,predict_KC)
    predict_NW = reMinMaxScaler(xy,predict_NW)
    predict_DDM = reMinMaxScaler(xy,predict_DDM)
    predict_DJK = reMinMaxScaler(xy,predict_DJK)
    predict_MP = reMinMaxScaler(xy,predict_MP)
    predict_SDM = reMinMaxScaler(xy,predict_SDM)
    predict_SPK = reMinMaxScaler(xy,predict_SPK)
    predict_SP = reMinMaxScaler(xy,predict_SP)
    predict_YC = reMinMaxScaler(xy,predict_YC)
    predict_YDP = reMinMaxScaler(xy,predict_YDP)
    predict_YS = reMinMaxScaler(xy,predict_YS)
    predict_EP = reMinMaxScaler(xy,predict_EP)
    predict_JL = reMinMaxScaler(xy,predict_JL)
    predict_J = reMinMaxScaler(xy,predict_J)
    predict_JR = reMinMaxScaler(xy,predict_JR)
    
    
    testY = reMinMaxScaler(xy,testY)
    testGNY = reMinMaxScaler(xy,testGNY)
    testGDY = reMinMaxScaler(xy,testGDY)
    testGPY = reMinMaxScaler(xy,testGPY)
    testGAY = reMinMaxScaler(xy,testGAY)
    testGJY = reMinMaxScaler(xy,testGJY)
    testGRY = reMinMaxScaler(xy,testGRY)
    testKCY = reMinMaxScaler(xy,testKCY)
    testNWY = reMinMaxScaler(xy,testNWY)
    testDDMY = reMinMaxScaler(xy,testDDMY)
    testDJKY = reMinMaxScaler(xy,testDJKY)
    testMPY = reMinMaxScaler(xy,testMPY)
    testSDMY = reMinMaxScaler(xy,testSDMY)
    testSPKY = reMinMaxScaler(xy,testSPKY)
    testSPY = reMinMaxScaler(xy,testSPY)
    testYCY = reMinMaxScaler(xy,testYCY)
    testYDPY = reMinMaxScaler(xy,testYDPY)
    testYSY = reMinMaxScaler(xy,testYSY)
    testEPY = reMinMaxScaler(xy,testEPY)
    testJLY = reMinMaxScaler(xy,testJLY)
    testJY = reMinMaxScaler(xy,testJY)
    testJRY = reMinMaxScaler(xy,testJRY)
    #testX = reMinMaxScaler(xy,testX)
    print("2")
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("3")
    print("RMSE: {}".format(rmse_val))
    
 # Plot predictions


fw = np.asarray([ predict_GN, predict_GD,predict_GP,predict_GA,predict_GJ,predict_GR,predict_KC,predict_NW,predict_DDM,predict_DJK,predict_MP,
                predict_SDM,predict_SPK,predict_SP,predict_YC,predict_YDP,predict_YS,predict_EP,predict_JL,predict_J,
                predict_JR]) 
fw = fw[:,:,0]

np.savetxt(r'C:\java_class\hadoop_workspace\TeamProject\WebContent\jsp\csv\regionData.csv', fw[:,:], delimiter=",")



