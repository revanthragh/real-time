# Set the paths to your train and test data folders
train_path = '/kaggle/input/ferdata/train'
test_path = '/kaggle/input/ferdata/test'

img_size = 48
batch_size = 64

# Data Augmentation for Training
datagen_train = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

# Create a generator for training data
train_generator = datagen_train.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Data Augmentation for Validation
datagen_validation = ImageDataGenerator(rescale=1./255)

# Create a generator for validation data
validation_generator = datagen_validation.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define the CNN model
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

opt = Adam(lr=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Learning Rate Schedule
def lr_schedule(epoch):
    lr = 0.0005
    if epoch > 10:
        lr *= 0.5
    elif epoch > 5:
        lr *= 0.1
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model
epochs = 120

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size,
    callbacks=[lr_scheduler]
)

# Save the model
model.save('/kaggle/working/e30.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_generator = datagen_validation.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load the trained model
saved_model = load_model('/kaggle/working/e30.h5')
