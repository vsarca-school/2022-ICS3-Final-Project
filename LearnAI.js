

/* ********************************************************************************
    This class is responsible for storing the weights and biases of a neural network, 
    as well as the gpu programs and other function needed to run the network and
    evaluate it
    Example usage: let nn = new NeuralNetwork(784, 100, 10).randomize();
******************************************************************************** */
class NeuralNetwork {
    /* ********************************************************************************
        This function explains the class so students can understand what it does
        Example usage: NeuralNetwork.print();
    ******************************************************************************** */
    static print() { // Use console.log to print the contents of the class instead
        console.log(`This is a placeholder description of the NeuralNetworks class
I hope I remember to fill this in before we submit the final copy!`);                                                                                                 // TODO
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

        // "The debug zone"
        console.log(`this.gpu.createKernel(function(inputs, biases, weights) {
            let sum = biases[this.thread.x];
            for (let i = 0; i < ${this.layersizes[0]}; i++) {
                sum += inputs[i]*weights[i][this.thread.x];
            }
            return sum;
        }).setOutput([${this.layersizes[0 + 1]}])`);

        // Layer computation functions ( Note to self: if you print this array, everything will magically stop working )
        this.layercomputers = [[], []];
        for (let i = 0; i < this.layerbiases.length; i++) {
            this.layercomputers[0].push(eval(`this.gpu.createKernel(function(inputs, biases, weights) {
                let sum = biases[this.thread.x];
                for (let i = 0; i < ${this.layersizes[i]}; i++) {
                    sum += inputs[i]*weights[i][this.thread.x];
                }
                return sum;
            }).setOutput([${this.layersizes[i + 1]}])`));
            this.layercomputers[1].push(eval(`this.gpu.createKernel(function(weighted_inputs) {
                function ${this.activation[0]}
                return ${this.activation[0].name}(weighted_inputs[this.thread.x]);
            }).setOutput([${this.layersizes[i + 1]}])`));
        }
        // Node computation functions (for backpropagation and gradient descent)
        this.outputnodescomputer = eval(`this.gpu.createKernel(function(weighted_inputs, outputs, expected_outputs) {
            function ${this.singleCost[1]}
            function ${this.activation[1]}
            return ${this.singleCost[1].name}(outputs[this.thread.x],expected_outputs[this.thread.x]) 
                * ${this.activation[1].name}(weighted_inputs[this.thread.x]);
        }).setOutput([${this.layersizes[this.layersizes.length - 1]}])`);
        this.hiddennodescomputers = [];
        for (let i = 1; i < this.layerbiases.length - 1; i++) {
            this.layercomputers[0].push(eval(`this.gpu.createKernel(function(weights, weighted_inputs, outputNodes) {
                function ${this.activation[1]}
                let value = 0;
                for (int i=0; i<${this.layersizes[i + 1]}; i++)
                {
                    value += weights[this.thread.x][i] * outputNodes[i];
                }
                return value * ${this.activation[1].name}(weighted_inputs[this.thread.x]);
            }).setOutput([${this.layersizes[i]}])`));
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
        for (let i = 0; i < output.length; i++) {
            cost.push(this.singleCost[0](output[i], expected[i]));
        }
        return cost;
    }

    /* ********************************************************************************
      This function calculates the cost of the network for a given datapoint or a set of datapoints
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
        console.log("Begin runNetwork");
        let currentLayer = data;
        for (let i = 0; i < this.layercomputers[0].length; i++) {
            console.log(currentLayer);
            currentLayer = this.layercomputers[0][i](currentLayer, this.layerbiases[i], this.layerweights[i]);
            console.log(currentLayer);
            currentLayer = this.layercomputers[1][i](currentLayer);
        }
        console.log("End runNetwork");
        return currentLayer;
    }
    getAllValues(data) {
        console.log("Begin getAllValues");
        // nodeValues with store a bunch of pairs, one for every layer. The first item in the pair is the weighted inputs, and the second is the activations/outputs/inputs to next layer. Netowrk input has no weighted input
        let nodeValues = [[null, data]];
        for (let i = 0; i < this.layercomputers[0].length; i++) {
            nodeValues.push([null, null]);
            console.log(nodeValues);
            nodeValues[i + 1][0] = this.layercomputers[0][i](nodeValues[i][1], this.layerbiases[i], this.layerweights[i]);
            console.log(nodeValues);
            nodeValues[i + 1][1] = this.layercomputers[1][i](nodeValues[i + 1][0]);
        }
        console.log("End getAllValues");
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




/* ********************************************************************************
    This class is responsible for training the neural network it is assigned to
    as well as the gpu programs and other function needed to run the network and
    evaluate it
    Example usage: let nn = new NeuralNetwork(784, 100, 10).randomize();
******************************************************************************** */
class DeepLearner {
    /* ********************************************************************************
        This function explains the class so students can understand what it does
        Example usage: DeepLearner.print();
    ******************************************************************************** */
    static print() { // Use console.log to print the contents of the class instead
        console.log(`This is a placeholder description of the DeepLearner class
I hope I remember to fill this in before we submit the final copy!`);                                                                                                 // TODO
    }

    /* ********************************************************************************
      This constructor generates all the arrays and data required to be able to train
      a given neural netowrk. The network cannot be changed after initialization
      See defaultSettings for a list of possible settings, or read this messy constructor
      Example usage: let dl = new DeepLearner(nn, dataset, DeepLearner.defaultSettings);
    ******************************************************************************** */
    constructor(network, dataset, settings) {
        // Variables
        this.network = network;
        this.trainingset = [[], []];
        this.testingset = [[], []];

        // Fill the two sets using the batchSplit setting
        let trainamount = dataset[0].length * 0.8; // default setting
        if (settings.hasOwnProperty("batchsplit")) trainamount = dataset[0].length * settings.batchsplit;

        for (let i = 0; i < trainamount; i++) {
            this.trainingset[0].push(dataset[0][i]);                                                                    // TODO: there must be a more efficient way to split arrays
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

    /* ********************************************************************************
      This object stores all the settings for the learner, which are explained in the object's comments
      Example usage: let dl = new DeepLearner(nn, dataset, DeepLearner.defaultSettings);
    ******************************************************************************** */
    static defaultSettings = {
        learnrate: 1, // for gradient descent
        batchsize: 100, // give it 100 samples at a time                                                                                                 // TODO: Make these comments
        batchsplit: 0.8 // 80% used, 20% saved for testing its learning on never before seen data
    }

    /* ********************************************************************************
      This function creates an interval which will call trainOnce at a user defined frequency (in milliseconds)
      This function is here for training until done
      Example usage: let trainer = dl.train(2000);
    ******************************************************************************** */
    train(milliseconds) {
        // Loop
        let trainer = setInterval(this.trainOnce.bind(this), milliseconds);                                                                                                 // TODO: store this in the function to automatically turn off when fully trained maybe? maybe its a setting
        return trainer; // In case the user wants to edit this interval
    }

    /* ********************************************************************************
      Most things below this point are unfinished                                                                                                 // TODO: Finish everything below this point, it really doesnt do much yet, and I have no idea how im going to do this yet
    ******************************************************************************** */


    trainOnce() {
        // Get the average gradient of all data points
        this.updateGradients();
        this.applyGradients();
    }


    /* ********************************************************************************
      This function will clear the gradient arrays and then calculate and add the gradient for a lot of datapoints
      Together with the gpu programs, this is half the magic of deep learning (the other half is actually using the gradients)
      Example usage: N/A
    ******************************************************************************** */
    updateGradients() {
        this.clearGradients();
        // For settings.batchsize amount of datapoints, calculate the gradients and add them (we will take the average later)
        for (let i = 0; i < this.settings.batchsize; i++) {
            // Get the node values of each input
            console.log("Data point is", this.trainingset);
            let layerdata = this.network.getAllValues(this.trainingset[0][this.batchindex]);
            console.log("Layer data is", layerdata);

            // Backpropagation
            let index = layerdata.length - 2; // Layer index

            // Calculate output layer gradients
            let nodeValues = this.network.outputnodescomputer(layerdata[index + 1][0], layerdata[index + 1][1], this.trainingset[1][this.batchindex]);
            for (let j = 0; j < nodeValues.length; j++) {
                // Calculate the gradients for weights
                for (let k = 0; k < layerdata[index][1].length; k++) {
                    this.wgradients[index][j][k] += layerdata[index][1][k] * nodeValues[j];
                }
                // Calculate the gradient for bias
                this.bgradients[index][j] += nodeValues[j];
            }

            // Calculate hidden layer gradients
            for (index--; index >= 0; index--) {
                nodeValues = this.network.hiddennodescomputers(this.network.layerweights[index], layerdata[index + 1][0], nodeValues);
                for (let j = 0; j < nodeValues.length; j++) {
                    // Calculate the gradients for weights
                    for (let k = 0; k < layerdata[index][1].length; k++) {
                        this.wgradients[index][j][k] += layerdata[index][1][k] * nodeValues[j];
                    }
                    // Calculate the gradient for bias
                    this.bgradients[index][j] += nodeValues[j];
                }
            }

            // Increment the index
            this.batchindex = (this.batchindex + 1) % this.trainingset.length;
        }
    }
    /* ********************************************************************************
      This function will clear the gradient arrays
      Example usage: N/A
    ******************************************************************************** */
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

    /* ********************************************************************************
      This function will use the gradients to train the network, using the gradient descent method
      Using the gradients as a slope of a graph, our goal is to slide down the slope and find the lowest point on the graph
      Example usage: N/A
    ******************************************************************************** */
    applyGradients() {
        // We use a little trick here, instead of taking the average of our gradients we use the sum and instead divide our learn rate
        let learnrate = this.settings.learnrate / this.settings.trainamount;
        let weightDecay = (1 - learnrate);

        // Apply weights
        for (let i = 0; i < this.wgradients.length; i++) {
            for (let j = 0; j < this.wgradients[i].length; j++) {
                for (let k = 0; k < this.wgradients[i][j].length; k++) {
                    this.network.layerweights[i][j][k] = this.network.layerweights[i][j][k] * weightDecay + this.wgradients[i][j][k] * learnrate;
                }
            }
        }

        // Apply biases
        for (let i = 0; i < this.bgradients.length; i++) {
            for (let j = 0; j < this.bgradients[i].length; j++) {
                this.network.layerbiases[i][j] += this.bgradients[i][j] * learnrate;
            }
        }
    }
}
