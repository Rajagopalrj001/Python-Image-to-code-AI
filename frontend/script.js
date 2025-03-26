document.getElementById("imageUpload").addEventListener("change", function(event) {
  let img = new Image();
  img.src = URL.createObjectURL(event.target.files[0]);
  img.onload = function () {
      let canvas = document.getElementById("canvas");
      let ctx = canvas.getContext("2d");
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
  };
});

document.getElementById("processBtn").addEventListener("click", function() {
  let canvas = document.getElementById("canvas");
  let dataUrl = canvas.toDataURL("image/png");

  fetch("http://127.0.0.1:5000/process", {
      method: "POST",
      body: JSON.stringify({ image: dataUrl }),
      headers: { "Content-Type": "application/json" }
  })
  .then(response => response.json())
  .then(data => {
      document.getElementById("output").innerHTML = `<pre>${data.html}</pre>`;
  });
});
