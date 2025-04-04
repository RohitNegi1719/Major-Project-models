import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Paths
train_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/train'
validation_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/validation'
test_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Transfer Learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,  # pre-trained VGG16 model
    layers.Flatten(),  # converts the 3D output into a 1D vector
    layers.Dense(256, activation='relu'),  # First Dense Layer
    layers.Dropout(0.3),  # Dropout Layer
    layers.BatchNormalization(),  # Batch Normalization Layer
    layers.Dense(128, activation='relu'),  # Second Dense Layer
    layers.Dropout(0.3),  # Second Dropout layer
    layers.Dense(4, activation='softmax')  # Output layer (4 classes)
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    min_lr=1e-6
)

# Training
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('vgg16_leukemia.h5')

# Evaluate on test data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy:.2f}')
"""
Class	        precision	recall	f1-score	support
Benign	        0.9600  	0.9474	0.9536	    76
Early	        0.9728  	0.9662	0.9695	    148
Pre	            1.0000	    1.0000	1.0000	    145
Pro         	0.9837	    1.0000	0.9918	    121
accuracy				 0.9816
macro avg	    0.9791	    0.9784	0.9787	    490
weighted avg	0.9816	    0.9816	0.9816	    490


"""