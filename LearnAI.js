
class NeuralNetwork {
    constructor(layer1, layer2, ...otherLayers) { // Empty weights and biases
        this.layersizes = [layer1, layer2].concat(otherLayers);
        this.layerbiases = Array(this.layersizes.length - 1);
        for (let i=0; i < this.layerbiases.length; i++)
        {
            this.layerbiases[i] = Array(this.layersizes[i]);
        }
        this.layerweights = Array(this.layersizes.length - 1);
        for (let i = 0; i < this.layerweights.length; i++) {
            this.layerweights[i] = Array(this.layersizes[i]);
            for (let j = 0; j < this.layersizes[i]; j++) {
                this.layerweights[i][j] = Array(this.layersizes[i + 1]);
            }
        }
    }
    randomize() { // Random weights and biases
        for (let i=0; i < this.layerbiases.length; i++)
        {
            for (let j=0; j < this.layersizes[i]; j++)
            {
                this.layerbiases
            }
        }
        for (let i = 0; i < this.layerweights.length; i++) {
            for (let j = 0; j < this.layersizes[i]; j++) {
                for (let k = 0; k < this.layersizes[i + 1]; k++) {
                    this.layerweights[i][j][k] = Math.random();
                }
            }
        }
    }
    from() {
        ;
    }

    cost (target) {
        return DeepTrainer.sigmoid(run);
    }
}

class DeepTrainer {
    // Activation function
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of activation function
    static dSigmoid(y) {
        return y * (1 - y);
    }
}

/*class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
      // Initialize weights and biases for input-to-hidden and hidden-to-output layers
      this.weightsIH = new Matrix(hiddenSize, inputSize);
      this.weightsHO = new Matrix(outputSize, hiddenSize);
      this.weightsIH.randomize();
      this.weightsHO.randomize();
      this.biasH = new Matrix(hiddenSize, 1);
      this.biasO = new Matrix(outputSize, 1);
      this.biasH.randomize();
      this.biasO.randomize();
    }
  
    // Activation function
    static sigmoid(x) {
      return 1 / (1 + Math.exp(-x));
    }
  
    // Derivative of activation function
    static dSigmoid(y) {
      return y * (1 - y);
    }
  
    // Feedforward function
    predict(inputs) {
      // Generate hidden outputs
      let inputMatrix = Matrix.fromArray(inputs);
      let hidden = Matrix.multiply(this.weightsIH, inputMatrix);
      hidden.add(this.biasH);
      // Activation function
      hidden.map(NeuralNetwork.sigmoid);
      // Generate outputs
    let output = Matrix.multiply(this.weightsHO, hidden);
    output.add(this.biasO);
    // Activation function
    output.map(NeuralNetwork.sigmoid);

    // Convert Matrix object to an array
    return output.toArray();
  }

  // Training function
  train(inputs, targets) {
    // Generate hidden outputs
    let inputMatrix = Matrix.fromArray(inputs);
    let hidden = Matrix.multiply(this.weightsIH, inputMatrix);
    hidden.add(this.biasH);
    // Activation function
    hidden.map(NeuralNetwork.sigmoid);

    // Generate outputs
    let outputs = Matrix.multiply(this.weightsHO, hidden);
    outputs.add(this.biasO);
    // Activation function
    outputs.map(NeuralNetwork.sigmoid);

    // Convert targets to a matrix
    let targetsMatrix = Matrix.fromArray(targets);

    // Calculate error
    let outputErrors = Matrix.subtract(targetsMatrix, outputs);

    // Calculate gradient
    let gradients = Matrix.map(outputs, NeuralNetwork.dSigmoid);
    gradients.multiply(outputErrors);
    gradients.multiply(this.learningRate);

    // Calculate deltas
    let hiddenT = Matrix.transpose(hidden);
    let weightHO_deltas = Matrix.multiply(gradients, hiddenT);

    // Adjust weights by deltas
    this.weightsHO.add(weightHO_deltas);
    // Adjust biases by gradient
    this.biasO.add(gradients);

    // Calculate hidden layer errors
    let whoT = Matrix.transpose(this.weightsHO);
    let hiddenErrors = Matrix.multiply(whoT, outputErrors);
    
    // Calculate hidden gradient
    let hiddenGradient = Matrix.map(hidden, NeuralNetwork.dSigmoid);
    hiddenGradient.multiply(hiddenErrors);
    hiddenGradient.multiply(this.learningRate);

    // Calculate input-to-hidden deltas
    let inputsT = Matrix.transpose(inputMatrix);
    let weightIH_deltas = Matrix.multiply(hiddenGradient, inputsT);

    // Adjust weights by deltas
    this.weightsIH.add(weightIH_deltas);
    // Adjust biases by gradient
    this.biasH.add(hiddenGradient);
  }
}*/