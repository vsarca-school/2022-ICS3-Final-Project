{
    let content = document.currentScript.parentElement;
    let textarea = content.getElementsByTagName("textarea")[0];
    //let a = content.getElementsByTagName("a")[0];
    let button = content.getElementsByTagName("button")[0];
    button.onclick = function () {
        eval(textarea.value);
    }
}