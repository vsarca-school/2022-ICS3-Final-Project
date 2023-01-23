{
    let content = document.currentScript.parentElement;
    let textarea = content.getElementsByTagName("textarea")[0];
    
    //textarea.rows = window.innerHeight / 20;
    window.onresize = function () {
        textarea.rows = Math.floor(window.innerHeight / 22);
    }

    let button = content.getElementsByTagName("button")[0];
    button.onclick = function () {
        eval(textarea.value);
    }
}