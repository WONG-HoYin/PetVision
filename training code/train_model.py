import os
import json
import math
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import datetime
import random
import itertools
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Multiply, GlobalAveragePooling2D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from collections import defaultdict
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight



base_dir = "/workspace"


logs_base_dir = os.path.join(base_dir, "logs")
train_data_dir = os.path.join(logs_base_dir, "train_data")

# create subdirectories
training_log_dir = os.path.join(train_data_dir, "training_log")
training_analysis_dir = os.path.join(train_data_dir, "training_analysis")
training_models_dir = os.path.join(train_data_dir, "training_models")
training_plots_dir = os.path.join(train_data_dir, "training_plots")
confusion_matrix_dir = os.path.join(train_data_dir, "confusion_matrix")
grad_cam_base_dir = os.path.join(train_data_dir, "grad_cam")
grad_cam_breed_dir = os.path.join(grad_cam_base_dir, "breed")


os.makedirs(training_log_dir, exist_ok=True)
os.makedirs(training_analysis_dir, exist_ok=True)
os.makedirs(training_models_dir, exist_ok=True)
os.makedirs(training_plots_dir, exist_ok=True)
os.makedirs(confusion_matrix_dir, exist_ok=True)
os.makedirs(grad_cam_base_dir, exist_ok=True)
os.makedirs(grad_cam_breed_dir, exist_ok=True)


current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")


existing_runs = []
for filename in os.listdir(training_log_dir):
    if filename.startswith("run_") and filename.endswith("_log.txt"):
        try:
            run_num = int(filename.split("_")[1])
            existing_runs.append(run_num)
        except:
            pass


if existing_runs:
    run_number = max(existing_runs) + 1
else:
    run_number = 1

run_id = f"run_{run_number:03d}_{timestamp}"


log_file_name = f"{run_id}_log.txt"
log_file_path = os.path.join(training_log_dir, log_file_name)

class SimpleLogger:
    def __init__(self, log_file):
        self.log_file = open(log_file, 'w', encoding='utf-8')
        self.terminal = sys.stdout
        
    def write(self, msg):
        self.log_file.write(msg)
        self.log_file.flush()
        self.terminal.write(msg)
        self.terminal.flush()
    
    def flush(self):
        self.log_file.flush()
        self.terminal.flush()
    
    def close(self):
        self.log_file.close()

sys.stdout = SimpleLogger(log_file_path)
print(f"[LOG] Training output will be saved to: {log_file_path}")
print(f"[LOG] Run ID: {run_id}")
print(f"[LOG] Start time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")


GPUs = tf.config.list_physical_devices('GPU')
print("GPUs available:", GPUs)
if GPUs:
    try:
        for gpu in GPUs:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(GPUs)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except Exception as e:
        print("Failed to set memory growth:", e)
else:
    print("No GPU detected — training will run on CPU.")

csv_path = os.path.join(base_dir, "final_dataset.csv")
model_save_path = os.path.join(base_dir, "petvision_model.h5")
map_save_path = os.path.join(base_dir, "category_map.json")
model_backup_path = os.path.join(training_models_dir, f"{run_id}_model")


IMG_SIZE = (224, 224)
BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 0.0001
if BATCH_SIZE >= 128:
    LEARNING_RATE = 0.00015

elif BATCH_SIZE >= 64:
    LEARNING_RATE = 0.0001

else:
    LEARNING_RATE = 0.00005
LOSS_WEIGHTS = [0.8, 1.2, 1.0, 1.0] 
GRADIENT_ACCUMULATION_STEPS = 1


initial_epochs = 45
fine_tune_epochs = 255
total_epochs = initial_epochs + fine_tune_epochs

print(f"Base Directory: {base_dir}")
print("Reading and encoding data...")
df = pd.read_csv(csv_path)

def ensure_container_path(path):
    path_str = str(path)
    
    path_str = path_str.replace('\\', '/')
    
    if path_str.startswith('/workspace/'):
        return path_str
    
    if path_str.startswith('C:/Users/A/Desktop/PetVision_Data/'):
        return path_str.replace('C:/Users/A/Desktop/PetVision_Data/', '/workspace/')
    elif path_str.startswith('c:/Users/A/Desktop/PetVision_Data/'):
        return path_str.replace('c:/Users/A/Desktop/PetVision_Data/', '/workspace/')
    
    if 'PetVision_Data/' in path_str:
        parts = path_str.split('PetVision_Data/')
        if len(parts) > 1:
            return f'/workspace/{parts[1]}'
    
    if not os.path.isabs(path_str):
        return os.path.join(base_dir, path_str)
    
    return path_str

print("Ensuring path...")
original_paths = df['filepath'].tolist()[:3]
df['filepath'] = df['filepath'].apply(ensure_container_path)

print("Original path sample:")
for i, path in enumerate(original_paths[:3]):
    print(f"  {i}: {path}")

print("Converted path sample:")
for i in range(min(3, len(df))):
    print(f"  {i}: {df['filepath'].iloc[i]}")

print("\nvalidating file accessibility(5 random sample):")
sample_indices = random.sample(range(len(df)), min(5, len(df)))
all_exist = True
for idx in sample_indices:
    filepath = df.iloc[idx]['filepath']
    exists = os.path.exists(filepath)
    status = "yes" if exists else "no"
    print(f"  {status} {os.path.basename(filepath)}")
    if not exists:
        all_exist = False

if not all_exist:
    print("Warning: Some files are missing! But continuing with training...")
else:
    print("File validation complete!")

target_cols = ['species_label', 'breed_label', 'size_label', 'coat_label']
mappings = {}

for col in target_cols:
    unique_values = sorted(df[col].astype(str).unique())
    mapping = {val: i for i, val in enumerate(unique_values)}
    mappings[col] = mapping
    df[col] = df[col].map(mapping)

with open(map_save_path, 'w') as f:
    json.dump(mappings, f)
print(f"Category map saved to: {map_save_path}")

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
val_df = val_df.reset_index(drop=True)
print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}")   

print("\n[CLASS BALANCE] Calculating class weights for breed task...")

breed_classes = train_df['breed_label'].values
unique_breeds = np.unique(breed_classes)
breed_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_breeds,
    y=breed_classes
)

breed_class_weight_dict = dict(zip(unique_breeds, breed_weights))

species_class_weight = {i: 1.0 for i in range(len(mappings['species_label']))}
size_class_weight = {i: 1.0 for i in range(len(mappings['size_label']))}
coat_class_weight = {i: 1.0 for i in range(len(mappings['coat_label']))}

print(f"  • species: {len(species_class_weight)} classes, all weights=1.0")
print(f"  • breed: {len(breed_class_weight_dict)} classes, weighted")
print(f"  • size: {len(size_class_weight)} classes, all weights=1.0")
print(f"  • coat: {len(coat_class_weight)} classes, all weights=1.0")

print(f"[CLASS BALANCE] Breed class distribution:")
print(f"  • Total classes: {len(unique_breeds)}")
print(f"  • Weight range: {min(breed_weights):.2f} - {max(breed_weights):.2f}")
print(f"  • Average weight: {np.mean(breed_weights):.2f}")
print(f"  • Median weight: {np.median(breed_weights):.2f}")

weight_bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, float('inf'))]
print(f"[CLASS BALANCE] Weight distribution:")
for low, high in weight_bins:
    count = sum(1 for w in breed_weights if low <= w < high)
    if count > 0:
        print(f"    {low}-{high}: {count} classes ({count/len(breed_weights)*100:.1f}%)")

def custom_generator(dataframe, directory, batch_size, is_train=True):
    if is_train:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=8 if is_train else 0,
            width_shift_range=0.08 if is_train else 0,
            height_shift_range=0.08 if is_train else 0,
            shear_range=0.03 if is_train else 0,
            zoom_range=[0.95, 1.05] if is_train else 0,
            horizontal_flip=True if is_train else False,
            brightness_range=[0.95, 1.05] if is_train else None,
            #channel_shift_range=5.0 if is_train else 0,

            fill_mode='reflect',
            
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)
    
    gen = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='filepath',
        y_col=target_cols,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=is_train,
        validate_filenames=False
    )
    
    while True:
        batch_x, batch_y = next(gen)
        yield batch_x, {
            'species_output': batch_y[:, 0],
            'breed_output':   batch_y[:, 1],
            'size_output':    batch_y[:, 2],
            'coat_output':    batch_y[:, 3]
        }

train_gen = custom_generator(train_df, base_dir, BATCH_SIZE, is_train=True)
val_gen = custom_generator(val_df, base_dir, BATCH_SIZE, is_train=False)

def get_weighted_categorical_crossentropy(class_weight_dict):
    '''
    Create a weighted categorical crossentropy loss function.
    param class_weight_dict: A dictionary mapping class indices to weights.
    return: A loss function that applies the specified class weights.
    '''
    class_weight_tensor = tf.constant([class_weight_dict[i] for i in sorted(class_weight_dict.keys())], dtype=tf.float32)

    def weighted_categorical_crossentropy(y_true, y_pred):
        y_true_flat = tf.reshape(y_true, [-1])
        y_true_int = tf.cast(y_true_flat, tf.int32)

        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)
        unweighted_loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred, from_logits=False)
        gathered_weights = tf.gather(class_weight_tensor, y_true_int)
        weighted_loss = unweighted_loss * gathered_weights
        return tf.reduce_mean(weighted_loss)

    return weighted_categorical_crossentropy

def focal_loss(alpha=0.35, gamma=1.1):
    """
    Focal Loss for multi-class classification
    alpha: balancing factor (list or dict)
    gamma: focusing parameter
    """

    def focal_loss_function(y_true, y_pred):
        y_true_flat = tf.reshape(y_true, [-1])
        y_true_int = tf.cast(y_true_flat, tf.int32)
        
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)
        
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        weight = tf.pow(1. - y_pred, gamma)

        alpha_val = alpha
        if isinstance(alpha_val, (list, np.ndarray)):
            alpha_tensor = tf.constant(alpha_val, dtype=tf.float32)
            alpha_weight = tf.gather(alpha_tensor, y_true_int)
            alpha_weight = tf.reshape(alpha_weight, [-1, 1])
        else:
            alpha_weight = tf.constant(alpha_val, dtype=tf.float32)
            alpha_weight = tf.reshape(alpha_weight, [])
        
        focal_weight = alpha_weight * weight
        
        focal_cross_entropy = focal_weight * cross_entropy
        loss = tf.reduce_sum(focal_cross_entropy, axis=1)   
        
        return tf.reduce_mean(loss)
    
    return focal_loss_function

def get_weighted_focal_loss(class_weight_dict, gamma=1.1):
    max_weight = max(class_weight_dict.values())
    min_weight = min(class_weight_dict.values())
    
    num_classes = len(class_weight_dict)
    alpha_list = []
    for i in sorted(class_weight_dict.keys()):
        weight = class_weight_dict[i]
        if max_weight == min_weight:
            norm_alpha = 0.5
        else:
            norm_alpha = 0.3 + 0.4 * ((weight - min_weight) / (max_weight - min_weight))
        alpha_list.append(norm_alpha)
    
    print(f"[Focal Loss] Alpha Range: {min(alpha_list):.3f} - {max(alpha_list):.3f}")
    print(f"[Focal Loss] Gamma: {gamma}")
    print(f"[Focal Loss] Num Classes: {num_classes}")
    
    return focal_loss(alpha=alpha_list, gamma=gamma)

# ============= Custom Callbacks ==============
#class DynamicLossWeights(tf.keras.callbacks.Callback):
    # Dynamically adjust loss weights based on breed task performance
    def __init__(self, initial_weights=[1.0, 4.5, 1.2, 1.1]):
        super().__init__()
        self.initial_weights = initial_weights
        self.current_weights = initial_weights.copy()
        self.weight_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        # Adjust weights every 5 epochs
        if epoch % 5 == 0 and epoch > 0:
            breed_acc = logs.get('val_breed_output_accuracy', 0.5)
            species_acc = logs.get('val_species_output_accuracy', 0.9)
            
            if breed_acc < 0.6:
                new_breed_weight = 6.0
            elif breed_acc < 0.7:
                new_breed_weight = 5.0
            else:
                new_breed_weight = 4.0
                
            # adjust species weight if accuracy is very high
            new_species_weight = 0.8 if species_acc > 0.98 else 1.0
            
            # Construct new weights list
            new_weights = [new_species_weight, new_breed_weight, 1.2, 1.1]
            
            # recompile model with new weights
            self.model.compile(
                optimizer=self.model.optimizer,
                loss=self.model.loss,
                loss_weights=new_weights,
                metrics=self.model.metrics
            )
            
            self.current_weights = new_weights
            self.weight_history.append({
                'epoch': epoch,
                'weights': new_weights,
                'breed_acc': breed_acc
            })
            
            print(f"\n[Dynamic Weight] Epoch {epoch}: Adjusted weight to: {new_weights}")
            print(f"[Dynamic Weight] Breed Accuracy: {breed_acc:.4f}, Species Accuracy: {species_acc:.4f}")

class GradientAccumulation(tf.keras.callbacks.Callback):
    # Gradient accumulation
    def __init__(self,):
        super().__init__()
        self.accumulation_steps = GRADIENT_ACCUMULATION_STEPS
        self.gradients = None
        self.accumulation_count = 0
        
    def on_train_batch_begin(self, batch, logs=None):
        # Initialize gradients
        if batch % self.accumulation_steps == 0:
            self.gradients = [tf.zeros_like(w) for w in self.model.trainable_weights]
            self.accumulation_count = 0

class AdaptiveEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=25, min_delta=0.0005, task='breed', monitor='loss'):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.task = task
        self.monitor = monitor
        self.best_weights = None
        if 'acc' in monitor or 'accuracy' in monitor:
            
            self.best_value = -np.inf
        else:
            self.best_value = np.inf

        self.wait = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        monitor_name = f'val_{self.task}_output_{self.monitor}'
        current_value = logs.get(monitor_name)
        
        if current_value is None:

            if monitor_name == 'val_breed_output_loss' and 'val_breed_output_categorical_crossentropy' in logs:
                current_value = logs.get('val_breed_output_categorical_crossentropy')

            elif monitor_name == 'val_breed_output_accuracy' and 'val_breed_output_categorical_accuracy' in logs:
                current_value = logs.get('val_breed_output_categorical_accuracy')

            else:
                print(f"[Warning] Value {monitor_name} not exist in logs: {list(logs.keys())}")
                return
        
        is_improvement = False
            
        if 'accuracy' in self.monitor or 'acc' in self.monitor: # If monitoring accuracy

            if current_value - self.best_value > self.min_delta:
                is_improvement = True
        else:
            # If monitoring loss
            if self.best_value - current_value > self.min_delta:
                is_improvement = True
        
        if is_improvement:
            self.best_value = current_value
            self.best_weights = self.model.get_weights()
            self.wait = 0
            print(f"[Early Stopping] {monitor_name} improved to {current_value:.4f}")
                
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

                print(f"\n[Early Stop Triggered] epoch {epoch}: {self.task} {self.monitor} did not improve")
                print(f"[Early Stop Triggered] Best Value: {self.best_value:.4f}")

class BreedTaskMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.breed_train_acc = []
        self.breed_val_acc = []
        self.gap_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('breed_output_accuracy', 0)
        val_acc = logs.get('val_breed_output_accuracy', 0)
        
        self.breed_train_acc.append(train_acc)
        self.breed_val_acc.append(val_acc)
        
        gap = train_acc - val_acc
        self.gap_history.append(gap)
        
        if len(self.gap_history) >= 5: 
            avg_gap = np.mean(self.gap_history[-5:])
            avg_train_acc = np.mean(self.breed_train_acc[-5:])
            avg_val_acc = np.mean(self.breed_val_acc[-5:])
            
            if avg_gap < -0.08 and avg_train_acc < 0.65:
                print(f"[Monitor]   Serious Underfit: Train={avg_train_acc:.4f}, Val={avg_val_acc:.4f}, Gap={avg_gap:.4f}")
            elif avg_gap > 0.2 and avg_val_acc < 0.7:
                print(f"[Monitor]   Probably Overfit but validation bad: Train={avg_train_acc:.4f}, Val={avg_val_acc:.4f}, Gap={avg_gap:.4f}")
            elif avg_gap < 0.15 and avg_gap > -0.05:
                print(f"[Monitor]  Status Normal: Train={avg_train_acc:.4f}, Val={avg_val_acc:.4f}, Gap={avg_gap:.4f}")
            else:
                print(f"[Monitor]  Training: Train={train_acc:.4f}, Val={val_acc:.4f}, Gap={gap:.4f}")
        else:
            print(f"[Monitor]  Training: Train={train_acc:.4f}, Val={val_acc:.4f}, Gap={gap:.4f}")

# ============= Training Monitor ==============
class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.stop_reason = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        if self.model.stop_training:
            self.stop_reason = "Training stopped by early stopping."
            print(f"[Warning!!] Training has been STOPPED on epoch {epoch+1}!")
            print(f"[Warning!!] Logs: {logs}")
            
    def on_train_end(self, logs=None):
        print(f"\n[Training Summary] Total epochs: {len(self.epoch_times)}")
        print(f"[Training Summary] Average time cost per epoch: {np.mean(self.epoch_times):.2f}sec")
        if self.stop_reason:
            print(f"[Training Summary] Stop Reason: {self.stop_reason}")

print("Constructing MobileNetV2 architecture...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False, 
    input_shape=(224, 224, 3))
base_model.trainable = False 

x = base_model.output

x = tf.keras.layers.Conv2D(512, (1, 1), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.BatchNormalization()(x)
shared_features = GlobalAveragePooling2D()(x)

shared_features = tf.keras.layers.Dense(512, activation='relu')(shared_features)
shared_features = tf.keras.layers.BatchNormalization()(shared_features)
shared_features = tf.keras.layers.Dropout(0.4)(shared_features)

def channel_attention(input_tensor, reduction=8):
    channel = input_tensor.shape[-1]
    shared_dense_one = tf.keras.layers.Dense(channel // reduction, activation='relu')
    shared_dense_two = tf.keras.layers.Dense(channel, activation='sigmoid')
    
    avg_pool = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)
    
    max_pool = tf.reduce_max(input_tensor, axis=1, keepdims=True)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)
    
    attention = tf.keras.layers.Add()([avg_pool, max_pool])
    attention = tf.keras.layers.Activation('sigmoid')(attention)
    
    return Multiply()([input_tensor, attention])

simple_features = tf.keras.layers.Dense(256, activation='relu')(shared_features)
simple_features = tf.keras.layers.BatchNormalization()(simple_features)
simple_features = tf.keras.layers.Dropout(0.3)(simple_features)

simple_branch = tf.keras.layers.Dense(128, activation='relu')(simple_features)
simple_branch = tf.keras.layers.Dropout(0.2)(simple_branch)

breed_features_1 = tf.keras.layers.Dense(512, activation='relu', name='breed_features_1',
                                         kernel_regularizer=regularizers.l2(0.0005))(shared_features)
breed_features_1 = tf.keras.layers.BatchNormalization()(breed_features_1)
breed_features_1 = tf.keras.layers.Dropout(0.65)(breed_features_1)

breed_features_2 = tf.keras.layers.Dense(256, activation='relu', name='breed_features_2', 
                                         kernel_regularizer=regularizers.l2(0.0004))(breed_features_1)
breed_features_2 = tf.keras.layers.BatchNormalization()(breed_features_2)
breed_features_2 = tf.keras.layers.Dropout(0.6)(breed_features_2)

breed_features_3 = tf.keras.layers.Dense(128, activation='relu', name='breed_features_3',
                                         kernel_regularizer=regularizers.l2(0.0003))(breed_features_2)
breed_features_3 = tf.keras.layers.BatchNormalization()(breed_features_3)
breed_features_3 = channel_attention(breed_features_3)
breed_features_3 = tf.keras.layers.Dropout(0.55)(breed_features_3)

breed_shortcut = tf.keras.layers.Dense(128, activation='relu')(shared_features)
breed_features_combined = tf.keras.layers.Add()([breed_features_3, breed_shortcut])
breed_branch = tf.keras.layers.Dropout(0.5)(breed_features_combined)

breed_branch = tf.keras.layers.Dense(128, activation='relu')(breed_branch)
breed_branch = tf.keras.layers.Dropout(0.5)(breed_branch)
# ======== Size Task Branch ========
size_features = tf.keras.layers.Dense(192, activation='relu')(shared_features)
size_features = tf.keras.layers.BatchNormalization()(size_features)
size_branch = tf.keras.layers.Dropout(0.3)(size_features)


outputs = []
task_branches = {
    'species_output': simple_branch,
    'breed_output': breed_branch,
    'size_output': size_branch,
    'coat_output': simple_branch
}
loss_dict = {}
metrics_dict = {}

for col in target_cols:
    num_classes = len(mappings[col])
    output_name = col.replace('_label', '_output')
    branch = task_branches[output_name] 
    
    out = Dense(num_classes, activation='softmax', name=output_name)(branch)
    outputs.append(out)
    
    if output_name == 'breed_output':
        loss_dict[output_name] = get_weighted_focal_loss(
            breed_class_weight_dict,  
            gamma=1.1
        )
    else:
        loss_dict[output_name] = 'sparse_categorical_crossentropy'
    metrics_dict[output_name] = 'accuracy'

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=loss_dict,
    loss_weights=LOSS_WEIGHTS,
    metrics=metrics_dict
)

overall_early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=25,
    restore_best_weights=True,
    min_delta=0.0005,
    verbose=1
)

breed_early_stop = AdaptiveEarlyStopping(
    patience=25,
    min_delta=0.0001,
    task='breed',
    monitor='accuracy'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_breed_output_accuracy',
    factor=0.5,
    patience=8,
    min_lr=1e-7,
    min_delta=0.0005,
    cooldown=2,
    mode='max',
    eps=1e-08,
    verbose=1
)
def cosine_annealing(epoch, lr):
    if epoch < initial_epochs:
        if epoch < 8:
            warmup_factor = (epoch + 1) / 8
            return LEARNING_RATE * warmup_factor
        else:
            # Cosine decay
            progress = (epoch - 8) / (initial_epochs - 8)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return LEARNING_RATE * (0.7 + 0.3 * cosine_decay)

    
    else:
        fine_tune_epoch = epoch - initial_epochs
        
        fine_tune_lr = LEARNING_RATE / 20
        
        if fine_tune_epoch < 5:
            warmup_factor = (fine_tune_epoch + 1) / 5
            return fine_tune_lr * warmup_factor
        else:
            restart_period = 30
            cycle = fine_tune_epoch // restart_period
            epoch_in_cycle = fine_tune_epoch % restart_period
            
            progress = epoch_in_cycle / restart_period
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            cycle_decay = 0.97 ** cycle
            return fine_tune_lr * cosine_decay * cycle_decay
        
cosine_lr = LearningRateScheduler(cosine_annealing, verbose=1)
    
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(logs_base_dir, f"tensorboard_{run_id}")
tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

breed_monitor = BreedTaskMonitor()
training_monitor = TrainingMonitor()

best_checkpoint = ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_breed_output_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

backup_checkpoint = ModelCheckpoint(
    filepath=model_backup_path,
    monitor='val_loss',
    mode='max',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

callbacks = [
    training_monitor,
    overall_early_stop,
    breed_early_stop,
    breed_monitor,
    #dynamic_weights,
    reduce_lr,
    cosine_lr,
    tb_cb,
    best_checkpoint,
    backup_checkpoint
]

print(f"Starting training for {EPOCHS} epochs...")

steps_per_epoch = math.ceil(len(train_df) / BATCH_SIZE)
validation_steps = math.ceil(len(val_df) / BATCH_SIZE)

print(f"[TRAINING] Starting initial training for {initial_epochs} epochs...")

history = model.fit(
    train_gen,
    epochs=initial_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

print(f"\n[TRAINING] Initial training complete. Proceeding to fine-tuning...")

N = 30
for layer in base_model.layers[-N:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 20),
    loss=loss_dict,
    loss_weights=LOSS_WEIGHTS,
    metrics=metrics_dict
)


print(f"\n[FINE-TUNING] Starting fine-tuning for {fine_tune_epochs} epochs...")

history_fine = model.fit(
    train_gen,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)    

print("\n[FINE-TUNING] Fine-tuning complete.")

model.save(model_save_path)
model.save(model_backup_path)
print(f"Training complete. Model saved to: {model_save_path}")
print(f"Model backup saved to: {model_backup_path}")
print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*50)
print(f"Log file: {log_file_path}")
print(f"Model file: {model_save_path}")
print(f"Category map: {map_save_path}")
print(f"Model backup: {model_backup_path}")
print(f"Confusion matrix: {confusion_matrix_dir}")
print(f"TensorBoard logs: {log_dir}")
print(f"Training analysis directory: {training_analysis_dir}")
print(f"Training plots directory: {training_plots_dir}")
print(f"Grad-CAM Heatmaps Directory: {grad_cam_base_dir}")
print(f"  - Breed: {grad_cam_breed_dir}/{run_id}/")
print("="*50 + "\n")

# ================= 5. Plot Training History =================
print("Generating training visualization...")

plot_path = None

def plot_training_history(history_initial, history_fine, run_id, output_dir):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History - PetVision Model (Run: {run_id})', fontsize=16, fontweight='bold')
    
    # Combine histories for plotting
    epochs_initial = list(range(1, len(history_initial.history['loss']) + 1))
    epochs_fine = list(range(len(epochs_initial) + 1, len(epochs_initial) + len(history_fine.history['loss']) + 1))
    
    all_loss = history_initial.history['loss'] + history_fine.history['loss']
    all_val_loss = history_initial.history['val_loss'] + history_fine.history['val_loss']
    all_epochs = epochs_initial + epochs_fine
    
    axes[0, 0].plot(all_epochs, all_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(all_epochs, all_val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].axvline(x=len(epochs_initial), color='g', linestyle='--', alpha=0.7, label='Fine-tuning Start')
    axes[0, 0].set_xlabel('Epoch', fontsize=10)
    axes[0, 0].set_ylabel('Loss', fontsize=10)
    axes[0, 0].set_title('Total Loss Curve', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    species_acc = history_initial.history['species_output_accuracy'] + history_fine.history['species_output_accuracy']
    val_species_acc = history_initial.history['val_species_output_accuracy'] + history_fine.history['val_species_output_accuracy']
    axes[0, 1].plot(all_epochs, species_acc, 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(all_epochs, val_species_acc, 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=10)
    axes[0, 1].set_ylabel('Accuracy', fontsize=10)
    axes[0, 1].set_title('Species Classification Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    breed_acc = history_initial.history['breed_output_accuracy'] + history_fine.history['breed_output_accuracy']
    val_breed_acc = history_initial.history['val_breed_output_accuracy'] + history_fine.history['val_breed_output_accuracy']
    axes[1, 0].plot(all_epochs, breed_acc, 'b-', label='Training', linewidth=2)
    axes[1, 0].plot(all_epochs, val_breed_acc, 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=10)
    axes[1, 0].set_ylabel('Accuracy', fontsize=10)
    axes[1, 0].set_title('Breed Classification Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    size_acc = history_initial.history['size_output_accuracy'] + history_fine.history['size_output_accuracy']
    val_size_acc = history_initial.history['val_size_output_accuracy'] + history_fine.history['val_size_output_accuracy']
    coat_acc = history_initial.history['coat_output_accuracy'] + history_fine.history['coat_output_accuracy']
    val_coat_acc = history_initial.history['val_coat_output_accuracy'] + history_fine.history['val_coat_output_accuracy']
    
    axes[1, 1].plot(all_epochs, size_acc, 'b-', label='Size (Train)', linewidth=2)
    axes[1, 1].plot(all_epochs, val_size_acc, 'r-', label='Size (Val)', linewidth=2)
    axes[1, 1].plot(all_epochs, coat_acc, 'g-', label='Coat (Train)', linewidth=2)
    axes[1, 1].plot(all_epochs, val_coat_acc, 'm-', label='Coat (Val)', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=10)
    axes[1, 1].set_ylabel('Accuracy', fontsize=10)
    axes[1, 1].set_title('Size & Coat Classification Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plot_filename = f"{run_id}_training_history.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f" Training history plot saved to: {plot_path}")
    
    try:
        plt.show()
    except Exception as display_error:
        print(f"  Could not display plot (running in headless environment): {display_error}")
    plt.close()
    return plot_path

try:
    plot_path = plot_training_history(history, history_fine, run_id, training_plots_dir)
except Exception as e:
    print(f"  Could not plot history: {e}")

# ================= 6. Generate Training Analysis JSON =================
def generate_training_json(history_initial, history_fine, run_id, output_dir):
    '''Generate a compact JSON file summarizing training details and performance'''
    json_filename = f"{run_id}_analysis.json"
    json_path = os.path.join(output_dir, json_filename)
    
    analysis_data = {
        "run_info": {
            "run_id": run_id,
            "run_number": run_number,
            "timestamp": timestamp,
            "start_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": (datetime.datetime.now() - current_time).total_seconds()
        },
        "config": {
            "model": "MobileNetV2",
            "image_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "initial_epochs": initial_epochs,
            "fine_tune_epochs": fine_tune_epochs,
            "total_epochs": total_epochs,
            "learning_rate": LEARNING_RATE,
            "fine_tune_learning_rate": LEARNING_RATE / 20
        },
        "dataset": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "num_classes": {col: len(mappings[col]) for col in target_cols}
        },
        "training_phases": {
            "initial": {
                "epochs": list(range(1, len(history_initial.history['loss']) + 1)),
                "loss": [float(x) for x in history_initial.history['loss']],
                "val_loss": [float(x) for x in history_initial.history['val_loss']],
            },
            "fine_tune": {
                "epochs": list(range(initial_epochs + 1, total_epochs + 1)),
                "loss": [float(x) for x in history_fine.history['loss']],
                "val_loss": [float(x) for x in history_fine.history['val_loss']],
            }
        },
        "task_performance": {},
        "output_files": {
            "log_file": log_file_path,
            "model_file": model_save_path,
            "model_backup": model_backup_path,
            "category_map": map_save_path,
            "plot_file": plot_path if plot_path else "Not generated"
        }
    }
    
    for col in target_cols:
        output_name = col.replace('_label', '_output')
        
        train_acc = []
        if f'{output_name}_accuracy' in history_initial.history:
            train_acc.extend([float(x) for x in history_initial.history[f'{output_name}_accuracy']])
        if f'{output_name}_accuracy' in history_fine.history:
            train_acc.extend([float(x) for x in history_fine.history[f'{output_name}_accuracy']])
        
        val_acc = []
        if f'val_{output_name}_accuracy' in history_initial.history:
            val_acc.extend([float(x) for x in history_initial.history[f'val_{output_name}_accuracy']])
        if f'val_{output_name}_accuracy' in history_fine.history:
            val_acc.extend([float(x) for x in history_fine.history[f'val_{output_name}_accuracy']])
        
        analysis_data["task_performance"][col] = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "final_train_accuracy": train_acc[-1] if train_acc else None,
            "final_val_accuracy": val_acc[-1] if val_acc else None
        }
    
    all_val_loss = analysis_data["training_phases"]["initial"]["val_loss"] + \
                   analysis_data["training_phases"]["fine_tune"]["val_loss"]
    
    analysis_data["statistics"] = {
        "best_val_loss": float(min(all_val_loss)),
        "best_val_loss_epoch": all_val_loss.index(min(all_val_loss)) + 1,
        "final_val_loss": float(all_val_loss[-1]),
        "final_train_loss": float(analysis_data["training_phases"]["fine_tune"]["loss"][-1]),
        "convergence_trend": "improving" if all_val_loss[-1] < all_val_loss[0] else "not_improving"
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"[ANALYSIS] Compact training data saved to: {json_path}")
    return json_path

json_file = generate_training_json(history, history_fine, run_id, training_analysis_dir)

def generate_confusion_matrices(model, val_gen, validation_steps, target_cols, mappings, run_id, output_dir):
   
    print("\n" + "="*60)
    print("GENERATING CONFUSION MATRICES")
    print("="*60)
    
    all_true = {col: [] for col in target_cols}
    all_pred = {col: [] for col in target_cols}
    
    print(f"[CONFUSION] Collecting predictions from validation set...")
    print(f"[CONFUSION] Validation steps: {validation_steps}")
    
    for step in range(validation_steps):
        if step % 5 == 0:  # Print every 5 steps
            print(f"[CONFUSION] Processing batch {step+1}/{validation_steps}...")
        
        batch_x, batch_y = next(val_gen)
        
        predictions = model.predict(batch_x, verbose=0)
        
        for idx, col in enumerate(target_cols):
            output_name = col.replace('_label', '_output')
            
            true_labels = batch_y[output_name]
            
            pred_probs = predictions[idx]
            pred_labels = np.argmax(pred_probs, axis=1)
            
            all_true[col].extend(true_labels)
            all_pred[col].extend(pred_labels)
    
    print(f"[CONFUSION] Prediction collection complete.")
    
    for col in target_cols:
        print(f"\n[CONFUSION] Generating matrix for {col}...")
        
        y_true = np.array(all_true[col])
        y_pred = np.array(all_pred[col])
        
        num_classes = len(mappings[col])
        y_true = np.clip(y_true, 0, num_classes - 1)
        y_pred = np.clip(y_pred, 0, num_classes - 1)
        
        print(f"[CONFUSION] Samples: {len(y_true)}")
        print(f"[CONFUSION] Classes: {num_classes}")
        print(f"[CONFUSION] Unique true labels: {len(np.unique(y_true))}")
        print(f"[CONFUSION] Unique pred labels: {len(np.unique(y_pred))}")
        
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        print(f"[CONFUSION] Accuracy from confusion matrix: {accuracy:.4f}")
        
        plt.figure(figsize=(12, 10))
        
        if col == 'breed_label' and num_classes > 30:
            print(f"[CONFUSION] Breed task has {num_classes} classes, showing top 30...")
            
            class_counts = np.sum(cm, axis=1)
            top_indices = np.argsort(class_counts)[-30:]  # choose top 30
            
            cm_subset = cm[np.ix_(top_indices, top_indices)]
            
            class_labels = [f"Class {i}" for i in top_indices]
            
            plt.imshow(cm_subset, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {col} (Top 30 Classes)\nRun: {run_id}', fontsize=14)
            plt.colorbar()
            plt.xticks(np.arange(len(class_labels)), class_labels, rotation=90, fontsize=8)
            plt.yticks(np.arange(len(class_labels)), class_labels, fontsize=8)
            
            thresh = cm_subset.max() / 2.
            for i, j in itertools.product(range(cm_subset.shape[0]), range(cm_subset.shape[1])):
                plt.text(j, i, format(cm_subset[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm_subset[i, j] > thresh else "black",
                        fontsize=6)
        else:
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {col}\nRun: {run_id}', fontsize=14)
            plt.colorbar()
            
            tick_marks = np.arange(num_classes)
            if num_classes <= 20:  # If not too many classes, show all
                plt.xticks(tick_marks, [f"Class {i}" for i in range(num_classes)], rotation=90, fontsize=8)
                plt.yticks(tick_marks, [f"Class {i}" for i in range(num_classes)], fontsize=8)
            else:  # Show fewer labels for large number of classes
                plt.xticks(tick_marks, [f"{i}" for i in range(num_classes)], rotation=90, fontsize=6)
                plt.yticks(tick_marks, [f"{i}" for i in range(num_classes)], fontsize=6)
            
            if num_classes <= 20:
                thresh = cm.max() / 2.
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=8)
        
        plt.tight_layout()
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        cm_filename = f"{run_id}_confusion_matrix_{col}.png"
        cm_path = os.path.join(output_dir, cm_filename)
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[CONFUSION] Saved to: {cm_path}")
        
        cm_csv_filename = f"{run_id}_confusion_matrix_{col}.csv"
        cm_csv_path = os.path.join(output_dir, cm_csv_filename)
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(cm_csv_path, index=False)
        print(f"[CONFUSION] Data saved to: {cm_csv_path}")
        
        if num_classes <= 50:
            report = classification_report(y_true, y_pred, 
                                          target_names=[f"Class {i}" for i in range(num_classes)],
                                          output_dict=True)
            
            report_filename = f"{run_id}_classification_report_{col}.json"
            report_path = os.path.join(output_dir, report_filename)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"[CONFUSION] Classification report saved to: {report_path}")
    
    print("\n" + "="*60)
    print("CONFUSION MATRICES GENERATION COMPLETE")
    print("="*60)
    
    return True

# ============8. Generate Grad-CAM Heatmaps ===========
#def get_last_conv_layer(model):
    """
    Get the name of the last Conv2D layer in the model, for generating Grad-CAM.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Conv2D Layer not found.")

#def create_grad_cam_heatmap(img_array, model, last_conv_layer_name, pred_class_index=None, task_index=0):
    """ Generates GRAD-CAM heatmap
    Params:
        img_array: image array after preprocessing (224, 224, 3)
        model: trained model
        last_conv_layer_name: name of the last convolutional layer
        pred_class_index: index of the predicted class
        task_index: index of the task (default 0)
    Returns:
        heatmap: heatmap array
    """
    # Create Conv2D output
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array[np.newaxis, ...], training=False)
    
    # Use the specified class or the class with the highest predicted probability
    if pred_class_index is not None:
        class_output = predictions[1][task_index][0, pred_class_index]
    else:
        class_output = predictions[1][task_index][0, np.argmax(predictions[1][task_index][0])]
    
    # Calculate gradients
    grads = tape.gradient(class_output, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Calculate weighted average
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    
    # Normalize to 0-1
    heatmap = heatmap.numpy()
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    
    return heatmap

#def overlay_heatmap_on_image(heatmap, original_img, alpha=0.4):
    """
    Overlays the heatmap on the original image
    
    Params:
        heatmap: heatmap array (H, W)
        original_img: original image array (H, W, 3), values in range 0-1
        alpha: transparency
    
    Returns:
        cam_img: image after overlaying
    """
    
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to 0-255 range
    if original_img.max() <= 1.0:
        original_img_255 = np.uint8(255 * original_img)
    else:
        original_img_255 = original_img.astype(np.uint8)
    
    # Overlay
    cam = cv2.addWeighted(heatmap, alpha, original_img_255, 1 - alpha, 0)
    
    return cam

#def generate_grad_cam_breed_images(model, val_df, mappings, run_id, grad_cam_breed_dir, samples_per_class=1, max_total_samples=20):
    """ 
    Generates and saves Grad-CAM heatmaps for breed task.
    Now only selects 20 random breeds instead of all breeds.
    """
    print("\n" + "="*60)
    print("GENERATING GRAD-CAM HEATMAPS (BREED TASK ONLY) - SELECTING 20 RANDOM BREEDS")
    print("="*60)
    
    # Get last convolutional layer name
    try:
        last_conv_layer_name = get_last_conv_layer(model)
        print(f"[GRAD-CAM] Last convolutional layer: {last_conv_layer_name}")
    except ValueError as e:
        print(f"[GRAD-CAM] Error: {e}")
        return False

    # Create subdirectory using run_id
    task_run_dir = os.path.join(grad_cam_breed_dir, run_id)
    os.makedirs(task_run_dir, exist_ok=True)
    print(f"\n[GRAD-CAM] Save directory: {task_run_dir}")

    # Get Breed class count
    breed_num_classes = len(mappings['breed_label'])
    print(f"[GRAD-CAM] Breed classes: {breed_num_classes}")

    # Select 20 random breeds to generate heatmaps for
    selected_breeds = random.sample(range(breed_num_classes), min(20, breed_num_classes))
    print(f"[GRAD-CAM] Selected {len(selected_breeds)} breeds: {selected_breeds}")

    # Collect sample indices for selected classes (only 1 sample per class)
    class_samples = {i: [] for i in selected_breeds}
    for idx, row in val_df.iterrows():
        # Check if idx is within valid range
        if idx >= len(val_df):
            print(f"[GRAD-CAM] Skipping invalid index {idx} (val_df has {len(val_df)} rows)")
            continue
            
        label = int(row['breed_label'])
        # Only add this sample if the label matches one of our selected breeds
        if label in selected_breeds and len(class_samples[label]) < samples_per_class:
            class_samples[label].append(idx)

    # Calculate total samples to process
    total_samples = sum(len(samples) for samples in class_samples.values())
    print(f"[GRAD-CAM] Total samples to process: {total_samples}")

    # Generate Grad-CAM for each sample
    sample_count = 0
    error_count = 0
    for class_idx, sample_indices in class_samples.items():
        for sample_idx in sample_indices:
            try:
                # Get image path using the row index
                img_path = val_df.iloc[sample_idx]['filepath']
                full_img_path = os.path.join(base_dir, img_path)
                
                # Check if file exists
                if not os.path.exists(full_img_path):
                    print(f"[GRAD-CAM] File not found: {full_img_path}")
                    error_count += 1
                    continue
                
                # Load original image (for display)
                original_img = tf.keras.utils.load_img(full_img_path, target_size=(224, 224))
                original_img_array = tf.keras.utils.img_to_array(original_img) / 255.0
                
                # Preprocess image for model input
                preprocessed_img = original_img_array.copy()
                if preprocessed_img.max() > 1.0:
                    preprocessed_img = preprocessed_img / 255.0
                
                # Get the true label for this sample
                true_label = int(val_df.iloc[sample_idx]['breed_label'])
                
                # Get model predictions
                batch_x = preprocessed_img[np.newaxis, ...]
                predictions = model.predict(batch_x, verbose=0)
                
                # Get output index for breed task
                pred_probs = predictions[1][0]
                pred_label = np.argmax(pred_probs)
                
                # Generate Grad-CAM heatmap for predicted class
                heatmap = create_grad_cam_heatmap(
                    preprocessed_img, model, last_conv_layer_name, 
                    pred_class_index=pred_label, task_index=1
                )
                
                # Overlay heatmap on original image
                cam_img = overlay_heatmap_on_image(heatmap, original_img_array)
                
                # Save image
                filename = f"{run_id}_grad_cam_breed_{sample_count:04d}_true{true_label:03d}_pred{pred_label:03d}.png"
                save_path = os.path.join(task_run_dir, filename)
                
                # Save using PIL
                cam_img_pil = Image.fromarray(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB))
                cam_img_pil.save(save_path, quality=95)
                
                sample_count += 1
                if sample_count % 5 == 0:
                    print(f"[GRAD-CAM] Processed {sample_count}/{total_samples} images...")
            except Exception as e:
                print(f"[GRAD-CAM] Error processing sample {sample_idx}: {e}")
                error_count += 1
                continue

    print(f"\n[GRAD-CAM] Breed task complete.")
    print(f"[GRAD-CAM] Successfully saved: {sample_count} images")
    if error_count > 0:
        print(f"[GRAD-CAM] Errors encountered: {error_count} images")
    print(f"[GRAD-CAM] Save directory: {task_run_dir}")
    print("\n" + "="*60)
    print("GRAD-CAM HEATMAPS GENERATION COMPLETE")
    print("="*60)
    return True

try:
    # Create a new validation generator
    val_gen_for_cm = custom_generator(val_df, base_dir, BATCH_SIZE, is_train=False)
    
    # Generate confusion matrices
    cm_success = generate_confusion_matrices(
        model=model,
        val_gen=val_gen_for_cm,
        validation_steps=min(100, validation_steps),  # limit steps for speed
        target_cols=target_cols,
        mappings=mappings,
        run_id=run_id,
        output_dir=confusion_matrix_dir
    )
    
    if cm_success:
        print(" Confusion matrices generated successfully!")
    else:
        print("  Failed to generate confusion matrices")
    '''
    grad_cam_success = generate_grad_cam_breed_images(
        model=model,
        val_df=val_df,
        mappings=mappings,
        run_id=run_id,
        grad_cam_breed_dir=grad_cam_breed_dir,
        samples_per_class=1,
        max_total_samples=20  # 500 images maximum
    )
    
    if grad_cam_success:
        print(" Grad-CAM heatmaps generated successfully!")
    else:
        print("  Failed to generate Grad-CAM heatmaps")
    '''
except Exception as e:
    print(f"  Error generating confusion matrices: {e}")
    print("Continuing with final output...")

# ================= 9. Final Output =================
print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*60)
print(f"Run ID: {run_id}")
print(f"Start Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration: {(datetime.datetime.now() - current_time).total_seconds():.2f} seconds")
print("-"*60)
print(f"Training Log: {log_file_path}")
print(f"Analysis JSON: {json_file}")
print(f"Training Plot: {plot_path if plot_path else 'Not generated'}")
print(f"Confusion Matrices Directory: {confusion_matrix_dir}")
print(f"Grad-CAM Heatmaps Directory: {grad_cam_base_dir}")
print(f"  - Breed: {grad_cam_breed_dir}/{run_id}/")
print(f"Model Backup: {model_backup_path}")
print(f"TensorBoard Logs: {log_dir}")
print(f"Main Model: {model_save_path}")
print(f"Category Map: {map_save_path}")
print("="*60 + "\n")

sys.stdout.close()
sys.stdout = sys.stdout.terminal