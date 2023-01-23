{
    let content = document.currentScript.parentElement;
    let textarea = content.getElementsByTagName("textarea")[0];
    //let a = content.getElementsByTagName("a")[0];
    let button = content.getElementsByTagName("button")[0];
    button.onclick = function () {
        eval(textarea.value);
    }
    
    /*a.onclick = function(e) {
        e.preventDefault();
        input.click();
    };
    let input = document.createElement("input");
    input.type = "file";
    input.onchange = function(e) {
        let reader = new FileReader();
        reader.onload = function () {
            nn = new NeuralNetwork(1,1).fromFile(JSON.parse(reader.result));
        }
        reader.readAsText(input.files[0]);
    }*/
}