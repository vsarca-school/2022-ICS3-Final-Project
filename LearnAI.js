
class NeuralNetwork {
    /* ********************************************************************************
        This function explains the class so students can understand what it does
        Example usage: NeuralNetwork.print();
    ******************************************************************************** */
    static print() { // Use console.log to print the contents of the class instead
        console.log(`This is a placeholder description of the NeuralNetworks class
I hope I remember to fill this in before we submit the final copy!`);
    }

    /* ********************************************************************************
      This constructor creates a neural network with undefined weights and biases
      Example usage: let nn = new NeuralNetwork(784, 100, 10);
    ******************************************************************************** */
    constructor(layer1, layer2, ...otherLayers) {
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
        // Layer computation functions (Note to self: if you print this array, everything will magically stop working)
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
        // Layer computers for backpropagation
        this.nodecomputers = [];
        for (let i = 0; i < this.layerbiases.length; i++) // TODO
        {
            this.nodecomputers.push(eval(`this.gpu.createKernel(function(inputs, biases, weights) {
                function ${NeuralNetwork.sigmoid}
                let sum = biases[this.thread.x];
                for (let i = 0; i < ${this.layersizes[i]}; i++) {
                    sum += inputs[i]*weights[i][this.thread.x];
                }
                return sigmoid(sum);
            }).setOutput([${this.layersizes[i + 1]}])`));
        }
    }
    /* ********************************************************************************
      This function will randomize all the weights in an existing network
      Example usage: let nn = new NeuralNetwork(784, 100, 10).randomize();
    ******************************************************************************** */
    randomize() {
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
    /* ********************************************************************************
      This function will load a neural network from a save file
      Example usage: let nn = new NeuralNetwork(784, 100, 10).fromFile("nn.txt");
    ******************************************************************************** */
    fromFile(file) {                                                                                                 // TODO
        //let layersizes = file.
        return this;
    }
    /* ********************************************************************************
      This function will save a neural network to a save file
      Example usage: nn.saveToFile("nn.txt");
    ******************************************************************************** */
    saveToFile(file) {                                                                                                 // TODO
        //let layersizes = file.
        return this;
    }

    /* ********************************************************************************
      The singleCost function calculates the cost of a single output of the network, can be set by user 
      to functions other than the default value, user must include derivative so that the network is trainable
      Both functions have 2 inputs and a return value
      Example usage: nn.setCostFunction(myfunc, myinverse);
    ******************************************************************************** */
    defaultCost(output, expected) {
        let error = output - expected;
        return error * error;
    }
    defaultCostDerivative(output, expected) {                                                                                                 // TODO
        let error = output - expected;
        return error * error;
    }
    singleCost = [this.defaultCost, this.defaultCostDerivative];
    setCostFunction(normal, derivative) {
        this.singleCost = [normal, derivative];
    }

    /* ********************************************************************************
      This function calculates the total cost of the network's output, not for user
      Example usage: N/A
    ******************************************************************************** */
    totalCost(output, expected) {
        let cost = [];
        for (let i=0; i<output.length; i++)
        {
            cost.push(this.singleCost[0](output[i], expected[i]));
        }
        return cost;
    }

    /* ********************************************************************************
      This function calculates the cost of the netowrk for a given datapoint or a set of datapoints
      Example usage: N/A
    ******************************************************************************** */                                                                                                 // TODO code needs to be nicer, cleaner, easier to use, faster, and just better
    /*Cost(data, targets, ...range) { // Takes an array of inputs and outputs and finds the cost of the neural network (average cost), range is [start,stop] for batchSize
        let cost = 0.0;
        // If its a dataset, calculate for each element, and then return average
        if (Array.isArray(data[0])) {
            // Find the start and stop of our data
            let start = 0;
            let end = data.length;
            if (range != undefined) {
                start = range[0];
                end = range[1];
            }
            // Find the average cost in that range
            for (let i = start; i < end; i++) {
                cost += this.Cost(data[i], targets[i]);
            }
            return cost / (end - start);
        }

        // Single input
        // Cost is difference between expected output and actual output
        let output = this.runNetwork(data);
        for (let i = 0; i < output.length; i++) {
            cost += this.nodeCost(output[i], targets[i]);
        }
        return cost;
    }*/

    /* ********************************************************************************
      These functions run the network and return its final outputs
      getNodeValues is used by the deep learner because it saves all intermittent values,
      which are necessary for training
      Example usage: N/A
    ******************************************************************************** */
    runNetwork(data) {
        let currentLayer = data;
        for (let i = 0; i < this.layercomputers.length; i++) {
            currentLayer = this.layercomputers[i](currentLayer, this.layerbiases[i], this.layerweights[i])[1];
        }
        return currentLayer;
    }
    getNodeValues(data) {
        let nodeValues = [[data]]; //this.layercomputers[0](data, this.layerbiases[0], this.layerweights[0])];
        for (let i=0; i<this.layercomputers.length; i++)
        {
            nodeValues.push(this.layercomputers[i](nodeValues[i], this.layerbiases[i], this.layerweights[i]));
        }
        return nodeValues;
    }


    /* ********************************************************************************
      The activation function is applied to a node's input to find the nodes output, can be set by user 
      to functions other than the default value, user must include derivative so that the network is trainable
      Both functions have 1 input and a return value
      Example usage: nn.setActivationFunction(myfunc, myinverse);
    ******************************************************************************** */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    sigmoidDerivative(y) {
        return y * (1 - y);
    }
    activation = [this.sigmoid, this.sigmoidDerivative];
    setActivationFunction(normal, derivative) {
        this.activation = [normal, derivative];
    }

    //                                                                                                 // TODO: put more default activation function here like SiLu and ReLu, which are cool, but then we also need to have an output activation which im not willing to implement just yet
}







class DeepLearner {
    // This function explains the class so students can understand what it does
    print() { // Use console.log to print the contents of the class instead
        console.log(`This is a placeholder description of the NeuralNetworks class
I hope I remember to fill this in before we submit the final copy!`);
    }

    // This class requires a network, training data, and some settings on how to train the network
    constructor(network, dataset, settings) {
        // Variables
        this.network = network;
        this.trainingset = [[], []];
        this.testingset = [[], []];
        // Fill the two sets using the batchSplit setting
        let trainamount = dataset[0].length * 0.8; // default setting
        if (settings.hasOwnProperty("batchSplit")) trainamount = dataset[0].length * settings.batchSplit;
        for (let i = 0; i < trainamount; i++) {
            this.trainingset[0].push(dataset[0][i]);
            this.trainingset[1].push(dataset[1][i]);
        }
        for (let i = trainamount; i < dataset[0].length; i++) {
            this.testingset[0].push(dataset[0][i]);
            this.testingset[1].push(dataset[1][i]);
        }
        // More variables
        this.settings = settings;
        if (!settings.hasOwnProperty("batchsize")) this.settings.batchsize = 100;
        this.batchindex = 0;
        // Weight gradients
        this.wgradients = Array(this.network.layerweights.length);
        for (let i = 0; i < this.network.layerweights.length; i++) {
            this.wgradients[i] = Array(this.network.layerweights[i].length);
            for (let j = 0; j < this.network.layerweights[i].length; j++) {
                this.wgradients[i][j] = Array(this.network.layerweights[i][j].length);
            }
        }
        // Bias gradients
        this.bgradients = Array(this.network.layerbiases.length);
        for (let i = 0; i < this.network.layerbiases.length; i++) {
            this.bgradients[i] = Array(this.network.layerbiases[i].length);
        }
    }
    // Pre-defined settings
    static defaultSettings = {
        learnRate: 1, // for gradient descent
        batchsize: 100, // give it 100 samples at a time
        batchsplit: 0.8 // 80% used, 20% saved for testing its learning on never before seen data
    }

    // Train the network contsantly
    train(milliseconds) {
        // Loop
        let trainer = setInterval(this.trainOnce.bind(this), milliseconds);
        return trainer; // In case the user wants to edit this interval
    }
    trainOnce() {
        // Get the average gradient of all data points
        this.updateGradients();
    }
    updateGradients() {
        this.clearGradients();
        // Run batchsize samples of the training data and calculate gradients
        for (let i = 0; i < this.settings.batchsize; i++) {
            // Get the node values of each input
            let nodevalues = this.network.getNodeValues(this.trainingset[0][this.batchindex]);
            // Calculate the gradients for all weights
            for (let j = 0; j < this.wgradients.length; j++)
            {
                for (let k=0; k< this.wgradients[j].length; k++)
                {
                    for (let l=0; l<this.wgradients[j][k].length; l++)
                    {
                        this.wgradients[j][k][l] += ;
                    }
                }
            }
            // Find the cost
            let cost = this.network.Cost(nodevalues[nodevalues.length-1], this.trainingset[1]);
            // Find gradient of last layer nodes
            

            this.batchindex = (this.batchindex + 1) % this.trainingset.length;
        }
    }
    clearGradients() {
        // Clear gradients
        for (let i = 0; i < this.wgradients.length; i++) {
            for (let j = 0; j < this.wgradients[i].length; j++) {
                this.wgradients[i][j].fill(0);
            }
        }
        for (let i = 0; i < this.bgradients.length; i++) {
            this.bgradients[i].fill(0);
        }
    }
    //updateGradient(datapoint) {

        /*
        let cost;
        let batchend = this.batchindex + this.settings.batchsize;
        if (batchend > this.trainingset[0].length) {
            // around the end, two calls to test data
            cost = this.network.Cost(this.trainingset[0], this.trainingset[1], this.batchindex, this.trainingset[0].length) + this.network.Cost(this.trainingset[0], this.trainingset[1], 0, batchend - this.trainingset[0].length);
        }
        else {
            cost = this.network.Cost(this.trainingset[0], this.trainingset[1], this.batchindex, batchend);
        }
        // increase batchindex
        this.batchindex = (this.batchindex + this.settings.batchsize) % this.trainingset[0].length;
        // return cost
        return cost;*/
    //}
}
