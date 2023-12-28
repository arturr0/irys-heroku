from flask import Flask, render_template, request
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the Iris dataset
iris = datasets.load_iris()
features = iris.data
labels = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)

# Initial number of neighbors
initial_neighbors = 1
initial_test_size = 0.5
# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=initial_neighbors)

# Train the model on the training data
knn.fit(X_train, y_train)

# Predict classes on the testing data
predictions = knn.predict(X_test)

# Evaluate accuracy
accuracy = metrics.accuracy_score(y_test, predictions)
def plot_classes(X, y, title):
    unique_labels = list(set(y))
    colors = plt.cm.Set1.colors[:len(unique_labels)]  # Use a colormap for distinct colors

    label_to_color = {label: color for label, color in zip(unique_labels, colors)}

    plt.scatter(X[:, 0], X[:, 1], c=[label_to_color[label] for label in y], edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_str = base64.b64encode(img_stream.read()).decode('utf-8')
    plt.close()
    
    return img_str
@app.route('/', methods=['GET', 'POST'])
def index():
    global knn, predictions, accuracy, X_train, X_test, y_train, y_test

    try:
        if request.method == 'POST':
            # Update the test size based on the form submission
            test_size = float(request.form['test_size'])
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)

            # Update the number of neighbors based on the form submission
            neighbors = int(request.form['neighbors'])
            knn = KNeighborsClassifier(n_neighbors=neighbors)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, predictions)

        # Return results and plot to be displayed in the browser
        result = {
            'accuracy': accuracy,
            'predictions': predictions.tolist(),
            'actual_labels': y_test.tolist(),
        }

        plot_img = plot_classes(X_test, predictions, 'Predicted Classes')
        
        return render_template('index.html', result=result, plot_img=plot_img, initial_test_size=initial_test_size, initial_neighbors=initial_neighbors)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
