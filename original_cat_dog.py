from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from solve_cudnn_error import solve_cudnn_error

solve_cudnn_error()

base_dir = Path('cats_and_dogs_small')

train_dir = base_dir / 'train'
validation_dir = base_dir / 'validation'
test_dir = base_dir / 'test'
train_cats_dir = train_dir / 'cats'
train_dogs_dir = train_dir / 'dogs'
validation_cats_dir = validation_dir / 'cats'
validation_dogs_dir = validation_dir / 'dogs'
test_cats_dir = test_dir / 'cats'
test_dogs_dir = test_dir / 'dogs'

print('total training cat images:', len(list(train_cats_dir.glob('*'))))
print('total training dog images:', len(list(train_dogs_dir.glob('*'))))
print('total validation cat images:', len(list(validation_cats_dir.glob('*'))))
print('total validation dog images:', len(list(validation_dogs_dir.glob('*'))))


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=0.01),
                  metrics=['acc'])

    return model


model = create_model()
model.summary()

train_datagen = ImageDataGenerator(rescale=1./255,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')

callbacks = []
callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint.h5',
                                                    save_best_only=True,
                                                    save_weights_only=False))

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.__len__(),
                              callbacks=callbacks,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.__len__())

# Plot training history
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('acc.png')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss.png')
