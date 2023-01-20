{
    let td = document.currentScript.parentElement;
    let select = td.getElementsByClassName("selection")[0].getElementsByClassName("selection")[0];
    console.log("new object");
    select.addEventListener('change', (event) => {
        // Selected thing is event.target.value
        let content;
        switch(event.target.value) {
        case "empty":
            console.log("empty");
            content = td.getElementsByClassName("content")[0];
            td.replaceChild(empty.cloneNode(true), content);
            break;
        case "Image Editor":
            console.log("image editor");
            content = td.getElementsByClassName("content")[0];
            console.log(content, image_editor);
            td.replaceChild(image_editor.cloneNode(true), content);
            break;
        case "Code Editor":
            console.log("code editor");
            content = td.getElementsByClassName("content")[0];
            td.replaceChild(code_editor.cloneNode(true), content);
            break;
        }
    });
}