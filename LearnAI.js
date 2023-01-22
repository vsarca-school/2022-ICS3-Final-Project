
function defaultCost(output, expected) {
    let error = output - expected;
    return error * error;
}
function defaultCostDerivative(output, expected) {
    return 2 * (output - expected);
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
function sigmoidDerivative(x) {
    let y = 1 / (1 + Math.exp(-x));
    return y * (1 - y);
}

function TanH(x) {
    let e2 = Exp(2 * x);
    return (e2 - 1) / (e2 + 1);
}
function TanHDerivative(x) {
    let e2 = Exp(2 * x);
    let y = (e2 - 1) / (e2 + 1);
    return 1 - y * y;
}

function ReLU(x) {
    return Math.max(0, x);
}
function ReLUDerivative(x) {
    return x > 0 ? 1 : 0
}

function SiLU(x) {
    return x / (1 + Math.exp(-x));
}
function SiLUDerivative(x) {
    let y = 1 / (1 + Math.exp(-x));
    return x * y * (1 - y) + y;
}

// System reqrite is needed in order to be able to use these
// function Softmax(x)
// function SoftmaxDerivative(x)

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
        this.totallayers = this.layersizes.length - 1; // Number of layers

        // Layer weights
        this.layerweights = Array(this.totallayers);
        for (let i = 0; i < this.totallayers; i++) {
            this.layerweights[i] = Array(this.layersizes[i]);
            for (let j = 0; j < this.layersizes[i]; j++) {
                this.layerweights[i][j] = Array(this.layersizes[i + 1]);
            }
        }

        // Layer biases
        this.layerbiases = Array(this.totallayers);
        for (let i = 0; i < this.totallayers; i++) {
            this.layerbiases[i] = Array(this.layersizes[i + 1]);
        }
    }
    /* ********************************************************************************
      This function will be called to create the gpu kernels once the user has changed the
      activation/cost functions to their satisfaction
      Example usage: let nn = new NeuralNetwork(784, 100, 10).generateGPU();
    ******************************************************************************** */
    generateGPU() {
        // Layer computation functions ( Note to self: if you print this array, everything will magically stop working )
        this.layercomputers = [[], []];
        for (let i = 0; i < this.totallayers; i++) {
            // inputs => weighted_inputs
            this.layercomputers[0].push(eval(`this.gpu.createKernel(function(inputs, biases, weights) {
                let sum = biases[this.thread.x];
                for (let ii = 0; ii < ${this.layersizes[i]}; ii++) {
                    sum += inputs[ii]*weights[ii][this.thread.x];
                }
                return sum;
            }).setOutput([${this.layersizes[i + 1]}])`));
            // weighted_inputs => outputs
            this.layercomputers[1].push(eval(`this.gpu.createKernel(function(weighted_inputs) {
                ${this.activation[0]}
                return ${this.activation[0].name}(weighted_inputs[this.thread.x]);
            }).setOutput([${this.layersizes[i + 1]}])`));
        }
        this.outputlayercomputer = eval(`this.gpu.createKernel(function(weighted_inputs) {
            ${this.outputactivation[0]}
            return ${this.outputactivation[0].name}(weighted_inputs[this.thread.x]);
        }).setOutput([${this.layersizes[this.totallayers]}])`);

        // Node computation functions (for backpropagation and gradient descent)
        // These calculate the derivative of the cost with respect to weighted input
        // cost => weighted_inputs
        this.outputnodescomputer = eval(`this.gpu.createKernel(function(weighted_inputs, outputs, expected_outputs) {
            ${this.singleCost[1]}
            ${this.outputactivation[1]}
            return ${this.singleCost[1].name}(outputs[this.thread.x],expected_outputs[this.thread.x])
                * ${this.outputactivation[1].name}(weighted_inputs[this.thread.x]);
        }).setOutput([${this.layersizes[this.layersizes.length - 1]}])`);
        // weighted_inputs => weighted_inputs
        this.hiddennodescomputers = [];
        for (let i = 1; i < this.totallayers; i++) {
            this.hiddennodescomputers.push(eval(`this.gpu.createKernel(function(weighted_inputs, weights, nodevalues) {
                ${this.activation[1]}
                let value = 0;
                for (let ii = 0; ii<${this.layersizes[i + 1]}; ii++)
                {
                    value += weights[this.thread.x][ii] * nodevalues[ii];
                }
                return value * ${this.activation[1].name}(weighted_inputs[this.thread.x]);
            }).setOutput([${this.layersizes[i]}])`));
        }

        return this;
    }
    /* ********************************************************************************
      This function will randomize all the weights in an existing network
      Example usage: let nn = new NeuralNetwork(784, 100, 10).randomize();
    ******************************************************************************** */
    randomize() {
        for (let i = 0; i < this.totallayers; i++) {
            for (let j = 0; j < this.layersizes[i]; j++) {
                let sqrt = Math.sqrt(this.layersizes[i]);
                for (let k = 0; k < this.layersizes[i + 1]; k++) {
                    this.layerweights[i][j][k] = (Math.random()*2 - 1) / sqrt;
                }
            }
        }
        for (let i = 0; i < this.totallayers; i++) {
            for (let j = 0; j < this.layersizes[i + 1]; j++) {
                this.layerbiases[i][j] = 0;
            }
        }
        return this;
    }
    /* ********************************************************************************
      This function will load a neural network from a save file
      Example usage: let nn = new NeuralNetwork(784, 100, 10).fromFile("nn.txt");
    ******************************************************************************** */
    fromFile(file) {                                                                                                 // TODO actual ui maybe?
        this.layersizes = file.layersizes;
        this.totallayers = this.layersizes.length - 1;
        this.layerweights = file.weights;
        this.layerbiases = file.biases;
        // Cant do activation and cost yet because I didnt save them properly, I should stringify the function
        
        console.log("Successfully loaded from file");

        return this;
    }
    /* ********************************************************************************
      This function will save a neural network to a save file
      Example usage: nn.saveToFile("nn.txt");
    ******************************************************************************** */
    saveToFile(filename) {                                                                                                 // TODO
        let a = document.createElement("a");
        let file = new Blob([JSON.stringify({
            activation: [JSON.stringify(this.activation[0]), JSON.stringify(this.activation[1])], 
            cost: [JSON.stringify(this.singleCost[0]), JSON.stringify(this.singleCost[1])], 
            layersizes: this.layersizes, 
            weights: this.layerweights, 
            biases: this.layerbiases})], {type: 'text:plain'});
        a.href = URL.createObjectURL(file);
        a.download = filename;
        a.click();
        a.remove();
    }

    /* ********************************************************************************
      The singleCost function calculates the cost of a single output of the network, can be set by user 
      to functions other than the default value, user must include derivative so that the network is trainable
      Both functions have 2 inputs and a return value
      Example usage: nn.setCostFunction(myfunc, myinverse);
    ******************************************************************************** */
    singleCost = [defaultCost, defaultCostDerivative];
    setCostFunction(normal, derivative) {
        this.singleCost = [normal, derivative];
        return cost;
    }

    /* ********************************************************************************
      This function calculates the total cost of the network's output, not for user
      Example usage: N/A
    ******************************************************************************** */
    averageCost(output, expected) {
        /*let cost = [];
        for (let i = 0; i < output.length; i++) {
            cost.push(this.singleCost[0](output[i], expected[i]));
        }
        return cost;*/
        let cost = 0;
        for (let i = 0; i < output.length; i++) {
            cost += this.singleCost[0](output[i], expected[i]);
        }
        return cost; // / output.length;
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
        let currentLayer = data;
        for (let i = 0; i < this.totallayers - 1; i++) {
            currentLayer = this.layercomputers[0][i](currentLayer, this.layerbiases[i], this.layerweights[i]);
            currentLayer = this.layercomputers[1][i](currentLayer);
        }
        currentLayer = this.layercomputers[0][this.totallayers - 1](currentLayer, this.layerbiases[this.totallayers - 1], this.layerweights[this.totallayers - 1]);
        currentLayer = this.outputlayercomputer(currentLayer);
        return currentLayer;
    }
    getAllValues(data) {
        // nodeValues with store a bunch of pairs, one for every layer. The first item in the pair is the weighted inputs, and the second is the activations/outputs/inputs to next layer. Netowrk input has no weighted input
        let nodeValues = [[null, data]];
        for (let i = 0; i < this.totallayers - 1; i++) {
            nodeValues.push([null, null]); // For clarity
            nodeValues[i + 1][0] = this.layercomputers[0][i](nodeValues[i][1], this.layerbiases[i], this.layerweights[i]);
            nodeValues[i + 1][1] = this.layercomputers[1][i](nodeValues[i + 1][0]);
        }
        nodeValues.push([null, null]); // For clarity
        nodeValues[this.totallayers][0] = this.layercomputers[0][this.totallayers - 1](nodeValues[this.totallayers - 1][1], this.layerbiases[this.totallayers - 1], this.layerweights[this.totallayers - 1]);
        nodeValues[this.totallayers][1] = this.outputlayercomputer(nodeValues[this.totallayers][0]);
        return nodeValues;
    }


    /* ********************************************************************************
      The activation function is applied to a node's input to find the nodes output, can be set by user 
      to functions other than the default value, user must include derivative so that the network is trainable
      Both functions have 1 input and a return value
      The gpu really doesn't like it when your function creates a variable, so only math in there ok?
      Example usage: nn.setActivationFunction(myfunc, myinverse);
    ******************************************************************************** */
    activation = [sigmoid, sigmoidDerivative];
    setActivationFunction(normal, derivative) {
        this.activation = [normal, derivative];
        return this;
    }
    outputactivation = [sigmoid, sigmoidDerivative];
    setOutputActivationFunction(normal, derivative) {
        this.activation = [normal, derivative];
        return this;
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
        this.settings = settings;
        this.totallayers = this.network.totallayers;
        this.totaldata = dataset[0].length;
        this.trainingset = [[], []];
        this.testingset = [[], []];
        this.batchindex = 0;
        this.totalcorrect = 0; // To track total correct every epoch
        this.incorrectguessesprinted = 0; // Every epoch we print the first 10 mistakes
        this.epochscompleted = 0;

        // Settings
        if (!this.settings.hasOwnProperty("learnrate")) this.settings.learnrate = 0.1;
        this.trainamount = this.totaldata * 0.8; // default setting
        if (this.settings.hasOwnProperty("batchsplit")) this.trainamount = this.totaldata * this.settings.batchsplit;
        if (!this.settings.hasOwnProperty("batchsize")) this.settings.batchsize = 100;
        if (!this.settings.hasOwnProperty("maxIncorrectGuessesToPrint")) this.settings.maxIncorrectGuessesToPrint = 10;

        // Fill the two sets using the batchsplit setting
        for (let i = 0; i < this.trainamount; i++) {
            this.trainingset[0].push(dataset[0][i]);                                                                    // TODO: there must be a more efficient way to split arrays
            this.trainingset[1].push(dataset[1][i]);
        }
        for (let i = this.trainamount; i < this.totaldata; i++) {
            this.testingset[0].push(dataset[0][i]);
            this.testingset[1].push(dataset[1][i]);
        }

        // Weight gradients and velocities
        this.wgradients = Array(this.network.layerweights.length);
        this.wvelocities = Array(this.network.layerweights.length);
        for (let i = 0; i < this.network.layerweights.length; i++) {
            this.wgradients[i] = Array(this.network.layerweights[i].length);
            this.wvelocities[i] = Array(this.network.layerweights[i].length);
            for (let j = 0; j < this.network.layerweights[i].length; j++) {
                this.wgradients[i][j] = Array(this.network.layerweights[i][j].length).fill(0);
                this.wvelocities[i][j] = Array(this.network.layerweights[i][j].length).fill(0);
            }
        }

        // Bias gradients
        this.bgradients = Array(this.network.layerbiases.length);
        this.bvelocities = Array(this.network.layerbiases.length);
        for (let i = 0; i < this.network.layerbiases.length; i++) {
            this.bgradients[i] = Array(this.network.layerbiases[i].length).fill(0);
            this.bvelocities[i] = Array(this.network.layerbiases[i].length).fill(0);
        }
    }

    /* ********************************************************************************
      This object stores all the settings for the learner, which are explained in the object's comments
      Example usage: let dl = new DeepLearner(nn, dataset, DeepLearner.defaultSettings);
    ******************************************************************************** */
    static defaultSettings = {
        learnrate: 0.1, // for gradient descent
        batchsize: 100, // give it 100 samples at a time                                                                                                 // TODO: Make these comments
        batchsplit: 0.8, // 80% used, 20% saved for testing its learning on never before seen data
        maxIncorrectGuessesToPrint: 1, // Print the first X incorrect guesses
        regularization: 0.1, // Used for weight decay, too big and your network forgets what it learned
        momentum: 0.9 // Weights remember how fast they were moving previously and accelerate down a slope, this is how much of the previous speed is retained (so 0.9 is losing 10% of the speed)
    }

    /*                                                                                                                          TODO: write this
    */
    test() {
        let totalcorrect = 0;
        for (let i = 0; i < this.testingset[0].length; i++) {
            let outputs = this.network.runNetwork(this.testingset[0][i]);
            let largest = 0;
            for (let j = 1; j < outputs[1].length; j++) {
                if (outputs[1][j] > outputs[1][largest])
                    largest = j;
            }
            if (this.testingset[1][i][largest] == 1) totalcorrect++;
        }
        return "The network gets " + totalcorrect + "/" + this.testingset[0].length + " on testing data";
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
      Most things below this point are untested                                                                                               // TODO: Finish everything below this point, it really doesnt do much yet, and I have no idea how im going to do this yet
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
        // Track cost for user because we can
        let cost = 0;

        // For settings.batchsize amount of datapoints, calculate the gradients and add them (we will take the average later)
        for (let i = 0; i < this.settings.batchsize; i++) {
            // Run the data through the network and save all layers
            let layerdata = this.network.getAllValues(this.trainingset[0][this.batchindex]); // Length is this.totallayers + 1

            // Track cost and see if the network is correct, for statistics
            cost += this.network.averageCost(layerdata[layerdata.length - 1][1], this.trainingset[1][this.batchindex]);
            let largest = 0;
            let index = this.totallayers - 1; // Layer index, used later for backpropagation
            for (let i = 1; i < layerdata[index + 1][1].length; i++) {
                if (layerdata[index + 1][1][i] > layerdata[index + 1][1][largest])
                    largest = i;
            }
            this.debugnow = false;
            if (this.trainingset[1][this.batchindex][largest] == 1) this.totalcorrect++;
            else if (this.incorrectguessesprinted < this.settings.maxIncorrectGuessesToPrint) {
                this.incorrectguessesprinted++;
                console.log("Incorrect guess", this.incorrectguessesprinted + ", network guessed", layerdata[index + 1][1], "for", this.trainingset[1][this.batchindex] + ", biggest is", largest);
                if (this.debug) {
                    this.debugnow = true;
                }
            }

            // Backpropagation begins
            // Calculate output layer gradients
            let nodevalues = this.network.outputnodescomputer(layerdata[index + 1][0], layerdata[index + 1][1], this.trainingset[1][this.batchindex]);
            for (let j = 0; j < nodevalues.length; j++) {
                // Calculate the gradients for weights
                for (let k = 0; k < layerdata[index][1].length; k++) {
                    this.wgradients[index][k][j] += layerdata[index][1][k] * nodevalues[j];
                }
                // Calculate the gradient for bias
                this.bgradients[index][j] += nodevalues[j];
            }

            if (this.debugnow) console.log("Output layer nodeValues", nodevalues);

            // Calculate hidden layer gradients
            for (index--; index >= 0; index--) {
                nodevalues = this.network.hiddennodescomputers[index](layerdata[index + 1][0], this.network.layerweights[index + 1], nodevalues);
                for (let j = 0; j < nodevalues.length; j++) {
                    // Calculate the gradients for weights
                    for (let k = 0; k < layerdata[index][1].length; k++) {
                        this.wgradients[index][k][j] += layerdata[index][1][k] * nodevalues[j];
                    }
                    // Calculate the gradient for bias
                    this.bgradients[index][j] += nodevalues[j];
                }

                if (this.debugnow) {
                    console.log("Hidden layer nodevalues", nodevalues, "other multiply thingy", layerdata[index][1]);
                    //console.log(nodevalues.length, layerdata[index][1].length);
                }
            }

            if (this.debugnow) {
                console.log("Gradients are w", this.wgradients, "b", this.bgradients);
                console.log("Weights are w", this.network.layerweights, "b", this.network.layerbiases);
                console.log("Fuck it, heres the layerdata", layerdata);
            }

            // Increment the index
            this.batchindex++;
            if (this.batchindex == this.trainingset[0].length) {
                this.epochscompleted++;
                console.log("Epoch", this.epochscompleted, "complete, network guessed", this.totalcorrect, "/", this.trainingset[0].length, "correct");
                this.batchindex = 0;
                this.totalcorrect = 0;
                this.incorrectguessesprinted = 0;
            }
        }

        console.log("Average cost on data is", cost / this.settings.batchsize);
    }

    /* ********************************************************************************
      This function will use the gradients to train the network, using the gradient descent method
      Using the gradients as a slope of a graph, our goal is to slide down the slope and find the lowest point on the graph
      Example usage: N/A
    ******************************************************************************** */
    applyGradients() {                                                                                                                              // TODO: this method screws up the weights because the gradients are screwed up
        // We use a little trick here, instead of taking the average of our gradients we use the sum and instead divide our learn rate
        let learnrate = this.settings.learnrate / this.trainamount;
        let weightDecay = (1 - this.settings.regularization * learnrate);

        // Apply weights
        for (let i = 0; i < this.wgradients.length; i++) {
            for (let j = 0; j < this.wgradients[i].length; j++) {
                for (let k = 0; k < this.wgradients[i][j].length; k++) {
                    let velocity = this.wvelocities[i][j][k] * this.settings.momentum - this.wgradients[i][j][k] * learnrate;
                    this.wvelocities[i][j][k] = velocity;
                    this.network.layerweights[i][j][k] = this.network.layerweights[i][j][k] * weightDecay + velocity;
                }
                this.wgradients[i][j].fill(0);
            }
        }

        // Apply biases
        for (let i = 0; i < this.bgradients.length; i++) {
            for (let j = 0; j < this.bgradients[i].length; j++) {
                let velocity = this.bvelocities[i][j] * this.settings.momentum - this.bgradients[i][j] * learnrate;
                this.bvelocities[i][j] = velocity;
                this.network.layerbiases[i][j] += velocity;
            }
            this.bgradients[i].fill(0);
        }
    }
}
