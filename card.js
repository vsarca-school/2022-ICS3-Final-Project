{
    let td = document.currentScript.parentElement;
    let buttons = td.getElementsByClassName("menu")[0].getElementsByClassName("dropdown")[0].children;
    console.log("New card created");

    for (let i=0; i<buttons.length; i++)
    {
        buttons[i].onclick = function () {
            content = td.getElementsByClassName("content")[0];
            td.replaceChild(card_types[i].cloneNode(true), content);
        }
    }

    /*select.addEventListener('change', (event) => {
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
    });*/
}