{
    let content = document.currentScript.parentElement;
    let canvas = content.getElementsByTagName("canvas")[0];

    let ctx = canvas.getContext("2d");
    let points = new Array();

    function addPoint() {
        clearArea();
        let val = document.getElementById("num").value;
        points.push(val);
        if (points.length === 1) {
            ctx.beginPath();
            ctx.arc(0, (canvas.height * (1 - val)), 2, 0, 2 * Math.PI);
            ctx.fillStyle = "#7D9DDF";
            ctx.fill();
            ctx.strokeStyle = "#7D9DDF";
            ctx.stroke();
        }
        else {
            for (let i = 0; i < points.length; i++) {
                let x = canvas.width / (points.length - 1) * i;
                let y = (canvas.height * (1 - points[i]));
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.fillStyle = "#7D9DDF";
                ctx.fill();
                ctx.strokeStyle = "#7D9DDF";
                ctx.stroke();
            }
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
    }
}