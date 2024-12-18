import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix



# ---------------------------- DATA INGESTION ---------------------------------------------
# Function to load the training data
def csv_to_tf_dataset(file_path, batch_size, validation_split=0):
    datasets = []

    # Read the CSV in chunks
    for chunk in pd.read_csv(file_path, chunksize=10000):
        # Drop unnecessary columns
        chunk = chunk.drop(columns=['path'], errors='ignore')
        chunk = chunk.loc[:, chunk.columns != 'Unnamed: 0']

        # Separate features and labels
        features = chunk.drop(columns=['label'])
        labels = chunk['label']

        # Convert to TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
        datasets.append(dataset)

    # Concatenate all datasets
    full_dataset = datasets[0]
    for dataset in datasets[1:]:
        full_dataset = full_dataset.concatenate(dataset)
    
    del datasets

    # Calculate buffer size for shuffling and split index for validation set
    total_samples = len(full_dataset)
    print('Buffer Size', total_samples)
    # Shuffle and split the dataset
    full_dataset = full_dataset.shuffle(buffer_size=total_samples)

    if (validation_split != 0):

        print(validation_split)

        val_samples = int(total_samples * validation_split)
        train_samples = total_samples - val_samples
        
        train_dataset = full_dataset.take(train_samples).batch(batch_size)
        val_dataset = full_dataset.skip(train_samples).batch(batch_size)

        del full_dataset

        return train_dataset, val_dataset
    
    else:

        full_dataset = full_dataset.batch(batch_size)
        return full_dataset

# Load the training data
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
file_path = 'train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
print("Loading data into memory....")
train_dataset, val_dataset = csv_to_tf_dataset(file_path, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

# for features, labels in val_dataset.take(1):
#     print("Feature shape:", features.shape)
#     print("Label shape:", labels.shape)
#     print("Labels:", labels.numpy())


# --------------------------- MODEL BUILDING AND TRAINING ----------------------------------
# Function to create the CNN model
def create_cnn_model(input_shape, num_classes, dropout_rate=0.5, learning_rate=0.001):  # dropout_rate and learning_rate to regulate overfitting and 
    # Input layer
    inputs = keras.Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv1D(64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # Second convolutional block
    x = layers.Conv1D(128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # Third convolutional block
    x = layers.Conv1D(256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # Flattening the 1D data
    x = layers.Flatten()(x)

    # Fully connected layer with dropout
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout layer

    # Output layer for multi-class classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with a customizable learning rate
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    
    return model

# Model parameters
FEATURE_LENGTH = 1024
NUM_CLASSES = 1000
input_shape = (FEATURE_LENGTH, 1)  
EPOCHS = 1
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001

model = create_cnn_model(input_shape, NUM_CLASSES)

# # Train the model using the train_dataset and val_dataset
print('Training the model...')
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS)

# # Saving the model
# model.save('model.keras')


#---------------------------- LOAD TEST SETS AND TEST THE MODEL ------------------------------------
# Load the test sets
test1_file_path = 'val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
test2_file_path = 'v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
test1_dataset = csv_to_tf_dataset(test1_file_path, BATCH_SIZE)
test2_dataset = csv_to_tf_dataset(test2_file_path, BATCH_SIZE)

# Loading the saved model
loaded_model = model
# loaded_model = keras.models.load_model('model.keras')
# loaded_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])

# Eval on test sets
print('Eavluating the model on test sets')
test1_loss, test1_accuracy = loaded_model.evaluate(test1_dataset)
test2_loss, test2_accuracy = loaded_model.evaluate(test2_dataset)

# Function to extract model prdiction and true label
def get_predictions(model, test_dataset):
    true_labels = []
    pred_labels = []

    for batch in test_dataset:
        # Extract images and labels from the batch
        images, labels = batch
        
        # Get model predictions
        predictions = model(images, training=False)
        
        # Convert predictions to class labels (assuming multi-class classification)
        pred_class = np.argmax(predictions, axis=1)
        
        # Store true and predicted labels
        true_labels.extend(labels.numpy())  # Convert tensor to numpy array
        pred_labels.extend(pred_class)      # Predicted labels

    return np.array(true_labels), np.array(pred_labels)

# Get predictions for both test sets
true_labels_test_set_1, pred_labels_test_set_1 = get_predictions(loaded_model, test1_dataset)
true_labels_test_set_2, pred_labels_test_set_2 = get_predictions(loaded_model, test2_dataset)


# ---------------------------------- CLASS DISTRIBUTION --------------------------------------------------------
print("Caculating class distribution.....")
all_labels = []
for _, labels in train_dataset:
    all_labels.extend(labels.numpy())
classes = pd.DataFrame(all_labels, columns=['label'])
class_counts = classes['label'].value_counts()

# Plot top 10 and bottom 10 classes by frequency
top_10_classes = class_counts.nlargest(10)
bottom_10_classes = class_counts.nsmallest(10)

plt.figure(figsize=(12, 6))

# Top 10 classes with annotations
plt.subplot(1, 2, 1)
top_10_bars = top_10_classes.plot(kind='bar', color='skyblue')
plt.title('Top 10 Classes by Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
for bar in top_10_bars.containers[0]:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(bar.get_height())}', 
             ha='center', va='bottom')

# Bottom 10 classes with annotations
plt.subplot(1, 2, 2)
bottom_10_bars = bottom_10_classes.plot(kind='bar', color='salmon')
plt.title('Bottom 10 Classes by Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
for bar in bottom_10_bars.containers[0]:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(bar.get_height())}', 
             ha='center', va='bottom')

# Plot a bell curve (distribution of class frequencies)
plt.figure(figsize=(10, 6))
sns.histplot(class_counts, kde=True, color='green')
plt.title('Class Frequency Distribution (Bell Curve)')
plt.xlabel('Frequency')
plt.ylabel('Density')
plt.show()

plt.tight_layout()
plt.show()



# ------------------ Distribution of Classes in Accuracy range --------------------------------------------------

def evaluate_and_cluster_classes(dataset, model, num_classes, interval=0.1):
    # Step 1: Calculate per-class accuracies
    all_predictions = []
    all_labels = []

    for features, labels in dataset:
        predictions = model.predict(features, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted_classes)
        all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))

    class_accuracies = {class_id: cm[class_id, class_id] / cm[class_id].sum() if cm[class_id].sum() != 0 else 0.0
                        for class_id in range(num_classes)}

    # Step 2: Cluster labels by accuracy intervals
    accuracy_clusters = {}
    for class_id, accuracy in class_accuracies.items():
        cluster_key = round(accuracy // interval * interval, 1)
        if cluster_key not in accuracy_clusters:
            accuracy_clusters[cluster_key] = []
        accuracy_clusters[cluster_key].append(class_id)

    # Step 3: Prepare data for plotting
    clusters = sorted(accuracy_clusters.keys())
    cluster_counts = [len(accuracy_clusters[cluster]) for cluster in clusters]

    # Step 4: Plot the clusters with annotations
    plt.figure(figsize=(10, 6))
    bars = plt.bar([f'{cluster:.1f} - {cluster + interval:.1f}' for cluster in clusters], cluster_counts, color='teal')
    plt.xlabel('Accuracy Range')
    plt.ylabel('Number of Classes')
    plt.title('Distribution of Classes by Accuracy Range')
    plt.xticks(rotation=45)

    # Add annotations on top of each bar
    for bar, count in zip(bars, cluster_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha='center', va='bottom')

    plt.show()

    return accuracy_clusters

# Example usage
num_classes = 1000

# Evaluate and plot for test set 2
print("Test Set 2 Accuracy Clusters:")
test2_accuracy_clusters = evaluate_and_cluster_classes(test2_dataset, loaded_model, num_classes)

# Evaluate and plot for test set 1
print("Test Set 1 Accuracy Clusters:")
test1_accuracy_clusters = evaluate_and_cluster_classes(test1_dataset, loaded_model, num_classes)




# ---------------------------- CLASSES WITH BIGGEST DROP ------------------------------------
print("Caculating performance drop....")
def evaluate_and_cluster_classes(dataset, model, num_classes):
    all_predictions = []
    all_labels = []

    for features, labels in dataset:
        predictions = model.predict(features, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted_classes)
        all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))

    # Calculate class accuracies
    class_accuracies = {class_id: cm[class_id, class_id] / cm[class_id].sum() if cm[class_id].sum() != 0 else 0.0
                        for class_id in range(num_classes)}

    # Cluster classes by rounding accuracies to nearest 0.1 intervals
    clustered_classes = defaultdict(list)
    for class_id, accuracy in class_accuracies.items():
        cluster_key = round(accuracy, 1)
        clustered_classes[cluster_key].append(class_id)

    return clustered_classes

# Example usage
num_classes = 1000
test1_clustered_classes = evaluate_and_cluster_classes(test1_dataset, loaded_model, num_classes)
test2_clustered_classes = evaluate_and_cluster_classes(test2_dataset, loaded_model, num_classes)

# Function to convert clustered classes to a class-to-accuracy dictionary
def convert_to_accuracy_dict(clustered_classes):
    accuracy_dict = {}
    for accuracy, classes in clustered_classes.items():
        for class_id in classes:
            accuracy_dict[class_id] = accuracy
    return accuracy_dict

# Convert clustered classes to dictionaries for both test sets
test1_accuracy_dict = convert_to_accuracy_dict(test1_clustered_classes)
test2_accuracy_dict = convert_to_accuracy_dict(test2_clustered_classes)

# Calculate accuracy drop for each class
accuracy_drop = {}
for class_id in test1_accuracy_dict:
    accuracy_test1 = test1_accuracy_dict.get(class_id, 0)
    accuracy_test2 = test2_accuracy_dict.get(class_id, 0)
    accuracy_drop[class_id] = accuracy_test1 - accuracy_test2

# Sort classes by the largest accuracy drop
sorted_accuracy_drop = sorted(accuracy_drop.items(), key=lambda x: x[1], reverse=True)

# Extract top classes and their drops
top_classes = [class_id for class_id, _ in sorted_accuracy_drop[:10]]
accuracy_drops = [drop for _, drop in sorted_accuracy_drop[:10]]

top_classes, accuracy_drops

plt.figure(figsize=(10, 6))
plt.bar(range(len(top_classes)), accuracy_drops, color='orange')
plt.xticks(range(len(top_classes)), top_classes)  # Use class IDs as labels
plt.xlabel('Class ID')
plt.ylabel('Accuracy Drop')
plt.title('Top 10 Classes with Largest Accuracy Drop from Test 1 to Test 2')

# Add annotations to show exact drop values on top of each bar
for i, drop in enumerate(accuracy_drops):
    plt.text(i, drop + 0.02, f'{drop:.1f}', ha='center', va='bottom')

plt.show()


# -------- FEATURE MEAN AND VARIANCE DIFFERENCES IN TEST SETS FOR CORRECT AND MISCLASSIFIED CLASSES -----------
print("Analysing feature space of the test sets....")
cols = pd.read_csv('val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv', nrows=0)
exclude_cols = ['path', 'Unnamed: 0']
use_cols = [col for col in cols if col not in exclude_cols]

# Load deep features for Test Set 1 and Test Set 2
test_set_1_df = pd.read_csv('val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv', usecols=use_cols)
test_set_2_df = pd.read_csv('v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv', usecols=use_cols)

test_set_1_labels = test_set_1_df['label']
test_set_1_features = test_set_1_df.drop('label', axis=1)

test_set_2_labels = test_set_2_df['label']
test_set_2_features = test_set_2_df.drop('label', axis=1)

features_test_set_1 = test_set_1_features
features_test_set_2 = test_set_2_features

# Function for feature analysis
def get_feature_analysis(features, true_labels, pred_labels):
    # Identify correctly predicted and misclassified indices
    correct_indices = np.where(true_labels == pred_labels)[0]
    misclassified_indices = np.where(true_labels != pred_labels)[0]
    
    # Split features into correctly predicted and misclassified
    features_correct = features.iloc[correct_indices]
    features_misclassified = features.iloc[misclassified_indices]
    
    # Compute mean and variance for each feature
    mean_correct = features_correct.mean()
    mean_misclassified = features_misclassified.mean()
    var_correct = features_correct.var()
    var_misclassified = features_misclassified.var()
    
    # Compute the difference in means to see prominence of each feature
    mean_diff = mean_correct - mean_misclassified
    var_diff = var_correct - var_misclassified
    
    # Perform t-test to check if the differences are statistically significant
    p_values = []
    for feature in features.columns:
        t_stat, p_value = ttest_ind(features_correct[feature], features_misclassified[feature], equal_var=False)
        p_values.append(p_value)
    
    # Create a DataFrame to store analysis results
    analysis_df = pd.DataFrame({
        'Feature': features.columns,
        'Mean_Correct': mean_correct,
        'Mean_Misclassified': mean_misclassified,
        'Mean_Diff': mean_diff,
        'Variance_Correct': var_correct,
        'Variance_Misclassified': var_misclassified,
        'Var_Diff': var_diff,
        'P_Value': p_values
    })
    
    return analysis_df

# Run analysis for each test set
analysis_test_set_1 = get_feature_analysis(features_test_set_1, true_labels_test_set_1, pred_labels_test_set_1)
analysis_test_set_2 = get_feature_analysis(features_test_set_2, true_labels_test_set_2, pred_labels_test_set_2)

# Display significant features (p-value < 0.05)
significant_features_test_set_1 = analysis_test_set_1[analysis_test_set_1['P_Value'] < 0.05]
significant_features_test_set_2 = analysis_test_set_2[analysis_test_set_2['P_Value'] < 0.05]

# Clustering of mean differences
mean_diff_combined = pd.DataFrame({
    'Test_Set_1': analysis_test_set_1['Mean_Diff'],
    'Test_Set_2': analysis_test_set_2['Mean_Diff']
})

kmeans = KMeans(n_clusters=3)  # Adjust clusters based on data
mean_diff_combined['Cluster'] = kmeans.fit_predict(mean_diff_combined)

# Plot clustering of feature prominence differences
# plt.figure(figsize=(10, 6))
# plt.scatter(mean_diff_combined['Test_Set_1'], mean_diff_combined['Test_Set_2'], c=mean_diff_combined['Cluster'])
# plt.xlabel('Mean Difference (Test Set 1)')
# plt.ylabel('Mean Difference (Test Set 2)')
# plt.title('Clustering of Feature Mean Differences Between Test Sets')
# plt.show()

print("Significant Features in Test Set 1:")
significant_features_test_set_1.sort_values(by='P_Value').head(10)
print("\nSignificant Features in Test Set 2:")
significant_features_test_set_2.sort_values(by='P_Value').head(10)


# --------------------------------- Feature distribution in test sets and t-test ------------------------------------------------
# Load the data from both CSV files
features_test_set_1 = test_set_1_features
features_test_set_2 = test_set_2_features

# Initialize a dictionary to store t-test results
t_test_results = {}

# Perform t-test for each feature
for col in features_test_set_1.columns:
    # Perform the independent t-test
    t_stat, p_value = ttest_ind(features_test_set_1[col], features_test_set_2[col], equal_var=False)
    t_test_results[col] = (t_stat, p_value)

# Convert the results into a DataFrame
t_test_results_df = pd.DataFrame(t_test_results, index=["t_statistic", "p_value"]).T

# Filter for significant features with p-value < 0.05
significant_features = t_test_results_df[t_test_results_df["p_value"] < 0.05].sort_values(by="p_value")
print("Significant feature differences:\n", significant_features)


# Set the number of significant features to visualize
num_features_to_plot = 6
significant_features = t_test_results_df.sort_values("p_value").head(num_features_to_plot).index

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  # Flatten to easily iterate over

# Plot each significant feature in the 3x3 grid
for i, feature in enumerate(significant_features):
    sns.histplot(features_test_set_1[feature], color="blue", label="Test Set 1", kde=True, stat="density", bins=30, ax=axes[i])
    sns.histplot(features_test_set_2[feature], color="red", label="Test Set 2", kde=True, stat="density", bins=30, ax=axes[i])

    axes[i].set_title(f"Distribution of Feature: {feature}")
    axes[i].set_xlabel(f"Values of {feature}")
    axes[i].set_ylabel("Density")
    axes[i].legend()

# Turn off any empty subplots
for j in range(num_features_to_plot, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


# ---------------- CLUSTER ANALYSIS OF THE SIGNIFICANT FEATURES --------------------------
print("Feature space cluster analysis....")
# Concatenate significant features from both test sets
combined_features = pd.concat([
    significant_features_test_set_1[['Feature', 'Mean_Diff', 'Var_Diff']],
    significant_features_test_set_2[['Feature', 'Mean_Diff', 'Var_Diff']]
], ignore_index=True)

# Apply K-means clustering on the combined data
kmeans_combined = KMeans(n_clusters=3)  # Adjust number of clusters as needed
combined_features['Cluster'] = kmeans_combined.fit_predict(combined_features[['Mean_Diff', 'Var_Diff']])

# Split the combined data back into the two test sets
significant_features_test_set_1['Cluster'] = combined_features.iloc[:len(significant_features_test_set_1)]['Cluster'].values
significant_features_test_set_2['Cluster'] = combined_features.iloc[len(significant_features_test_set_1):]['Cluster'].values

# Plot clustering for significant features in Test Set 1
plt.figure(figsize=(10, 6))
plt.scatter(
    significant_features_test_set_1['Mean_Diff'], 
    significant_features_test_set_1['Var_Diff'], 
    c=significant_features_test_set_1['Cluster'], 
    cmap='viridis'
)
plt.xlabel('Mean Difference (Test Set 1)')
plt.ylabel('Variance Difference (Test Set 1)')
plt.title('Clustering of Significant Features - Test Set 1')
plt.colorbar(label='Cluster')
plt.show()

# Plot clustering for significant features in Test Set 2
plt.figure(figsize=(10, 6))
plt.scatter(
    significant_features_test_set_2['Mean_Diff'], 
    significant_features_test_set_2['Var_Diff'], 
    c=significant_features_test_set_2['Cluster'], 
    cmap='viridis'
)
plt.xlabel('Mean Difference (Test Set 2)')
plt.ylabel('Variance Difference (Test Set 2)')
plt.title('Clustering of Significant Features - Test Set 2')
plt.colorbar(label='Cluster')
plt.show()

