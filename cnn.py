import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/train'
validation_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/validation'
test_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,

    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir, # Path to training data
    target_size=(224, 224), # Resize images to 224x224
    batch_size=16,  #  Load images in batches of 16
    class_mode='categorical'  # Multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,  # Path to validation data
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Model
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

# Compile the model with a reduced learning rate
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.00001),
    metrics=['accuracy']
)

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('cnn_leukemia.h5')

# Evaluate on test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy:.2f}')


"""
Class	      precision	  recall	f1-score	support
Benign	      0.8152	  0.9868	0.8929	    76
Early	      1.0000	  0.8986	0.9466	    148
Pre	          0.9730	  0.9931	0.9829	    145
Pro	          1.0000	  0.9669	0.9832	    121
accuracy             0.968
macro avg	  0.9470      0.9614	0.9514	    490
weighted avg  0.9633	  0.9571	0.9581	    490

"""