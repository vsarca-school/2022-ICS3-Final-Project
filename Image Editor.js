{
    let arr = new Array(28);
    for (let i = 0; i < 28; i++) {
        arr[i] = new Array(28).fill(0);
    }
    // When true, moving the mouse draws on the canvas2
    let scale = 20;
    let drawing = false;
    let erasing = false;
    let x = 0;
    let y = 0;
    let dist = 0;
    let max = Math.sqrt(2) * (scale / 2);
    let slope = 255 / max;

    const canvas2 = document.getElementById('canvas2');
    const ctx2 = canvas2.getContext('2d');
    ctx2.fillStyle = "black";
    ctx2.fillRect(0, 0, canvas2.width, canvas2.height);

    // event.offsetX, event.offsetY gives the (x,y) offset from the edge of the canvas2.
    function drawPoint(e) {
        console.log("drawing point");
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
            ctx2.fillStyle = `rgb(${color}, ${color}, ${color})`;
            ctx2.fillRect((x - 1) * scale, (y - 1) * scale, scale, scale);
        }
        color = 255 * (1 - x2 * x2 - y1 * y1);
        if (!drawing) color = 0;
        if (x >= 0 && x < 28 && y > 0 && y < 28 && (drawing && arr[x][y - 1] < color || erasing && arr[x][y - 1] > color)) {
            arr[x][y - 1] = color;
            ctx2.fillStyle = `rgb(${color}, ${color}, ${color})`;
            ctx2.fillRect((x) * scale, (y - 1) * scale, scale, scale);
        }
        color = 255 * (1 - x1 * x1 - y2 * y2);
        if (!drawing) color = 0;
        if (x > 0 && x < 28 && y >= 0 && y < 28 && (drawing && arr[x - 1][y] < color || erasing && arr[x - 1][y] > color)) {
            arr[x - 1][y] = color;
            ctx2.fillStyle = `rgb(${color}, ${color}, ${color})`;
            ctx2.fillRect((x - 1) * scale, (y) * scale, scale, scale);
        }
        color = 255 * (1 - x2 * x2 - y2 * y2);
        if (!drawing) color = 0;
        if (x >= 0 && x < 28 && y >= 0 && y < 28 && (drawing && arr[x][y] < color || erasing && arr[x][y] > color)) {
            arr[x][y] = color;
            ctx2.fillStyle = `rgb(${color}, ${color}, ${color})`;
            ctx2.fillRect((x) * scale, (y) * scale, scale, scale);
        }
    }

    // Add the event listeners for mousedown, mousemove, and mouseup
    canvas2.addEventListener('mousedown', (e) => {
        if (e.button == 0) drawing = true;
        else if (e.button == 1) erasing = true;
        drawPoint(e);
    });

    canvas2.addEventListener('mousemove', drawPoint);

    canvas2.addEventListener('mouseup', (e) => {
        if (e.button == 0) drawing = false;
        else if (e.button == 1) erasing = false;
    });

    function drawLine(ctx2, x1, y1, x2, y2) {
        ctx2.beginPath();
        ctx2.strokeStyle = "white";
        ctx2.lineWidth = document.getElementById('selWidth').value;
        ctx2.lineJoin = "round";
        ctx2.moveTo(x1, y1);
        ctx2.lineTo(x2, y2);
        ctx2.closePath();
        ctx2.stroke();
    }

    function clearArea() {
        ctx2.fillStyle = "black";
        ctx2.fillRect(0, 0, canvas2.width, canvas2.height);
        for (let i = 0; i < 28; i++) {
            arr[i].fill(0);
        }
    }
}