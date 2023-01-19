const select = document.getElementById("selection");
const td = document.currentScript.parentElement;
select.addEventListener('change', (event) => {
    // Selected thing is event.target.value
    let content;
    switch(event.target.value) {
    case "empty":
        console.log("empty");
        content = td.getElementsByClassName("content")[0];
        td.replaceChild(empty, content);
        break;
    case "Image Editor":
        console.log("image editor");
        console.log(content, image_editor);
        content = td.getElementsByClassName("content")[0];
        td.replaceChild(image_editor, content);
        break;
    case "Code Editor":
        console.log("code editor");
        content = td.getElementsByClassName("content")[0];
        td.replaceChild(code_editor, content);
        break;
    }
});