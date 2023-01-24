{
    let arr = new Array(28);
    for (let i = 0; i < 28; i++) {
        arr[i] = new Array(28).fill(0);
    }
    let testingdata = new Array(784).fill(0);
    // When true, moving the mouse draws on the canvas
    let scale = 20;
    let drawing = false;
    let erasing = false;

    let content = document.currentScript.parentElement;
    let canvas = content.getElementsByTagName("canvas")[0];
    let submit = content.getElementsByClassName("submitButton")[0];
    let clear = content.getElementsByClassName("clearButton")[0];
    let guess = content.getElementsByClassName("guess")[0];
    submit.onclick = submitImage;
    clear.onclick = clearArea;

    let ctx = canvas.getContext('2d');
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // event.offsetX, event.offsetY gives the (x,y) offset from the edge of the canvas.
    function drawPoint(e) {
        if (!drawing && !erasing) return;
        let x = e.offsetX / scale + 0.5;
        let y = e.offsetY / scale + 0.5;
        let x1 = x % 1;
        let y1 = y % 1;
        let x2 = 1 - x1;
        let y2 = 1 - y1;
        x = Math.floor(x);
        y = Math.floor(y);
        let color = 255 * (1 - x1 * x1 - y1 * y1);
        if (!drawing) color = 0;
        if (x > 0 && x < 28 && y > 0 && y < 28 && (drawing && arr[x - 1][y - 1] < color || erasing && arr[x - 1][y - 1] > color)) {
            arr[x - 1][y - 1] = color;
            testingdata[(x-1) + 28*(y-1)] = color/255;
            ctx.fillStyle = `rgb(${color}, ${color}, ${color})`;
            ctx.fillRect((x - 1) * scale, (y - 1) * scale, scale, scale);
        }
        color = 255 * (1 - x2 * x2 - y1 * y1);
        if (!drawing) color = 0;
        if (x >= 0 && x < 28 && y > 0 && y < 28 && (drawing && arr[x][y - 1] < color || erasing && arr[x][y - 1] > color)) {
            arr[x][y - 1] = color;
            testingdata[(x) + 28*(y-1)] = color/255;
            ctx.fillStyle = `rgb(${color}, ${color}, ${color})`;
            ctx.fillRect((x) * scale, (y - 1) * scale, scale, scale);
        }
        color = 255 * (1 - x1 * x1 - y2 * y2);
        if (!drawing) color = 0;
        if (x > 0 && x < 28 && y >= 0 && y < 28 && (drawing && arr[x - 1][y] < color || erasing && arr[x - 1][y] > color)) {
            arr[x - 1][y] = color;
            testingdata[(x-1) + 28*(y)] = color/255;
            ctx.fillStyle = `rgb(${color}, ${color}, ${color})`;
            ctx.fillRect((x - 1) * scale, (y) * scale, scale, scale);
        }
        color = 255 * (1 - x2 * x2 - y2 * y2);
        if (!drawing) color = 0;
        testingdata[(x) + 28*(y)] = color/255;
        if (x >= 0 && x < 28 && y >= 0 && y < 28 && (drawing && arr[x][y] < color || erasing && arr[x][y] > color)) {
            arr[x][y] = color;
            ctx.fillStyle = `rgb(${color}, ${color}, ${color})`;
            ctx.fillRect((x) * scale, (y) * scale, scale, scale);
        }
    }

    // Add the event listeners for mousedown, mousemove, and mouseup
    canvas.addEventListener('mousedown', (e) => {
        if (e.button == 0) drawing = true;
        else if (e.button == 1) erasing = true;
        drawPoint(e);
    });

    canvas.addEventListener('mousemove', drawPoint);

    canvas.addEventListener('mouseup', (e) => {
        if (e.button == 0) drawing = false;
        else if (e.button == 1) erasing = false;
    });

    function drawLine(ctx, x1, y1, x2, y2) {
        ctx.beginPath();
        ctx.strokeStyle = "white";
        ctx.lineWidth = document.getElementById('selWidth').value;
        ctx.lineJoin = "round";
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.closePath();
        ctx.stroke();
    }

    function clearArea() {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < 28; i++) {
            arr[i].fill(0);
        }
    }

    function submitImage() {
        let output = nn.runNetwork(testingdata.slice(0,784));
        let largest = 0;
        for (let i=1; i<output.length; i++)
        {
            if (output[i] > output[largest]) largest = i;
        }
        guess.innerHTML = "Network guesses "+largest;
        console.log("Network output for user-drawn image: ", output);
    }
}