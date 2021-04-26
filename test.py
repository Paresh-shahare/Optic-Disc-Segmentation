from keras.models import load_model
model = load_model('C:/Users/Paresh/Downloads/dataset/UNetW_final.h5')
print(model.summary())


Training_data = r'C:/Users/Paresh/Downloads/dataset/test/origin1/'
training_data = [x for x in sorted(os.listdir(Training_data))]
print(len(training_data))
print(training_data)
x_train_data = np.empty((len(training_data),width,height),dtype = 'float32')
for i,name  in enumerate(training_data):
    im = cv2.imread(Training_data + name).astype('int16').astype('float32')/255.
    img = im[:,:,2]
    img = cv2.resize(img,dsize = (width,height),interpolation = cv2.INTER_NEAREST)
    x_train_data[i] = img

#im = cv2.imread(Training_data + name).astype('int16').astype('float32')/255.
#img = im[:,:,2]
#img = cv2.resize(img,dsize = (width,height),interpolation = cv2.INTER_NEAREST)
#x_train_data[0] = img


print(x_train_data.shape)
#fig, ax = plt.subplots(1,2,figsize = (8,4))
#ax[0].imshow(x_train_data[0])
#ax[1].imshow(x_train_data[1],cmap='gray')
x_train_data = x_train_data.reshape(x_train_data.shape[0],width,height,1)
print(x_train_data.shape)

#x_train_data = x_train_data.reshape(1,width,height,1)
#x, y = valid_gen.__getitem__(1)
result = model.predict(x_train_input_data,verbose=1)

result = (result > 0.5).astype(np.uint8)



fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(y[0]*255, (width, height)), cmap="gray")

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255, (width, height)), cmap="gray")



fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(y[1]*255, (width, height)), cmap="gray")

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[1]*255, (width, height)), cmap="gray")