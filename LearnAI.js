
/*const layercomputers = eval(`gpu.createKernel(function(a) {
                return a[this.thread.x];
            }).setOutput([10]);`);*/

class NeuralNetwork {
    constructor(layer1, layer2, ...otherLayers) { // Empty weights and biases
        //Gpu.js
        this.gpu = new GPU();
        // Layer sizes
        this.layersizes = [layer1, layer2].concat(otherLayers);
        // Layer biases
        this.layerbiases = Array(this.layersizes.length - 1);
        for (let i = 0; i < this.layerbiases.length; i++) {
            this.layerbiases[i] = Array(this.layersizes[i + 1]);
        }
        // Layer weights
        this.layerweights = Array(this.layersizes.length - 1);
        for (let i = 0; i < this.layerweights.length; i++) {
            this.layerweights[i] = Array(this.layersizes[i]);
            for (let j = 0; j < this.layersizes[i]; j++) {
                this.layerweights[i][j] = Array(this.layersizes[i + 1]);
            }
        }
        // Layer computation functions (Note to self: if you print this array, everythign will magically stop working)
        this.layercomputers = [];
        for (let i = 0; i < this.layerbiases.length; i++) {
            this.layercomputers.push(eval(`this.gpu.createKernel(function(inputs, biases, weights) {
                function ${NeuralNetwork.sigmoid}
                let sum = biases[this.thread.x];
                for (let i = 0; i < ${this.layersizes[i]}; i++) {
                    sum += inputs[i]*weights[i][this.thread.x];
                }
                return sigmoid(sum);
            }).setOutput([${this.layersizes[i + 1]}])`));
        }
    }
    randomize() { // Random weights and biases
        for (let i = 0; i < this.layerbiases.length; i++) {
            for (let j = 0; j < this.layersizes[i + 1]; j++) {
                this.layerbiases[i][j] = Math.random();
            }
        }
        for (let i = 0; i < this.layerweights.length; i++) {
            for (let j = 0; j < this.layersizes[i]; j++) {
                for (let k = 0; k < this.layersizes[i + 1]; k++) {
                    this.layerweights[i][j][k] = Math.random();
                }
            }
        }
        return this;
    }
    static fromFile(file) { // Loads AI from file, TODO
        let n = NeuralNetwork(1, 1); // We are deleting the network anyways, so make it really really small
        //let layersizes = file.
        return n;
    }

    print() { // Use console.log to print the contents of the class, this function explains the class so people can understand it
        console.log(`This is a placeholder description of the NeuralNetworks class
I hope I remember to fill this in before we submit the final copy!`);
    }

    nodeCost(output, expected) {
        let error = output - expected;
        return error * error;
    }
    calculateCost(data, targets) { // Takes an array of inputs and outputs and finds the cost of the neural network
        let cost = 0.0;
        // Cost is difference between expected output and actual output
        let output = this.runNetwork(data);
        for (let i = 0; i < output.length; i++) {
            cost += this.nodeCost(output[i], targets[i]);
        }
        return cost;
    }

    runNetwork(data) {
        let currentLayer = data;
        for (let i = 0; i < this.layercomputers.length; i++) {
            currentLayer = this.layercomputers[i](data, this.layerbiases[i], this.layerweights[i]);
        }
        return currentLayer;
    }
    // Activation function
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    // Derivative of activation function
    static dSigmoid(y) {
        return y * (1 - y);
    }
}

class DeepTrainer {
    constructor () {}

    // Activation function
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    // Derivative of activation function
    static dSigmoid(y) {
        return y * (1 - y);
    }

    train(n, dataset) {
        ;
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