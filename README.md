# Bayesian_Net: Neural Naive Bayes Classifier

<p align="center">
  <img src="https://github.com/jacobmcasey/bayesian_net/assets/71528526/9594ae14-770b-4004-981e-f29b4a0a1d3d" alt="Bayesian_net" width="200"/>
</p>
Bayesian_Net employs a fully connected, 3-layer neural network implemented in NumPy for predicting priors which are input into Bernoulli Naive Bayes 🌐

# Features
- 🤖Hybrid Model: Combines neural networks and Naive Bayes for intelligent probabilies.
- 🔢NumPy Based: Efficient matrix operations and computations.
- ⚙️Customisable: Easily adjust layers, learning rate, and iterations.
- 📊One-hot Encoding Utility: Convert integer lists to one-hot encoded numpy arrays.

## How To Use
### Import the Classifier.
from hybrid_classifier import Classifier

### Initialize the Classifier with desired hyper-parameters.
clf = Classifier(layers=[25,8,4], learning_rate=0.005, iterations=100)

### Train the model using data.
clf.fit(good_moves_data, target)

### Predict using a given state of Pacman.
best_move = clf.predict(state_data, legal)

## 📖 Documentation
For detailed documentation on each class and function, please refer to the project documentation's DOCSTRINGS using PEP8.

## 💾 Installation & Requirements
- Python 3.x
- NumPy

### Clone the repository and install using pip:
git clone https://github.com/your_username/bayesian_net.git
cd bayesian_net
pip install numpy

## 🤝 Contributing
Feel free to submit pull requests, enhancements, or report bugs. My email is jacobcasey.999@gmail.com for any questions!

## 📜 License
This project is licensed under the MIT License.
