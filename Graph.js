{
    let content = document.currentScript.parentElement;
    let canvas = content.getElementsByTagName("canvas")[0];
    let start = content.getElementsByClassName("startButton")[0];
    start.onclick = startButton;
    let clear = content.getElementsByClassName("clearButton")[0];
    clear.onclick = clearArea;
    let accuracy = content.getElementsByClassName("accuracy")[0];
    let cost = content.getElementsByClassName("cost")[0];

    let ctx = canvas.getContext("2d");
    let points = new Array();

    function addPoint(num) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        points.push(num);
        if (points.length === 1) {
            ctx.beginPath();
            ctx.arc(0, (canvas.height * (1 - num)), 2, 0, 2 * Math.PI);
            ctx.fillStyle = "#7D9DDF";
            ctx.fill();
            ctx.strokeStyle = "#7D9DDF";
            ctx.stroke();
        }
        else {
            /* for (let i = 0; i < points.length; i++) {
                let x = canvas.width / (points.length - 1) * i;
                let y = (canvas.height * (1 - points[i]));
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.fillStyle = "#7D9DDF";
                ctx.fill();
                ctx.strokeStyle = "#7D9DDF";
                ctx.stroke();
            }*/
            ctx.beginPath();
            for (let i = 0; i < points.length; i++) {
                let x = canvas.width / (points.length - 1) * i;
                let y = (canvas.height * (1 - points[i]));
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.strokeStyle = "#7D9DDF";
            ctx.stroke();
        }


    }

    function clearArea() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        points = new Array();
    }

    let timer;
    function startButton() {
        timer = setInterval(function () {
            if (dl.batchindex == 0) return;
            addPoint(dl.totalcorrect / dl.batchindex);
            
        }, 2000);
        start.innerHTML = "Stop Graphing";
        start.onclick = stopButton;
        dl.onEpoch = function (correct, total) {
            accuracy.innerHTML = "The network got "+correct+" out of"+total+" in the most recent epoch."
        }
        dl.onCost = function (value) {
            cost.innerHTML = "Cost: "+value;
        }
    }
    function stopButton() {
        clearInterval(timer);
        start.innerHTML = "Start Graphing";
        start.onclick = startButton;
    }
}