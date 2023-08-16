const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");


navigator.mediaDevices.getUserMedia({video: true}).then((stream) => {
    video.srcObject = stream;
    video.play();
}).catch((error) => {
    console.error("An error has occurred" + error)
})

const loadModel = (async () => {
    const model = tf.loadGraphModel("model/model.json");
    return model;
})

const preprocess = (source, modelWidth, modelHeight) => {
    let xRatio, yRatio; // ratios for boxes
  
    const input = tf.tidy(() => {
      const img = tf.browser.fromPixels(source);
  
      // padding image to square => [n, m] to [n, n], n > m
      const [h, w] = img.shape.slice(0, 2); // get source width and height
      const maxSize = Math.max(w, h); // get max size
      const imgPadded = img.pad([
        [0, maxSize - h], // padding y [bottom only]
        [0, maxSize - w], // padding x [right only]
        [0, 0],
      ]);
  
      xRatio = maxSize / w; // update xRatio
      yRatio = maxSize / h; // update yRatio
  
      return tf.image
        .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
        .div(255.0) // normalize
        .expandDims(0); // add batch
    });
  
    return [input, xRatio, yRatio];
  };





let output_boxes = null;
let output_scores = null;
let selected_classes = null;

video.addEventListener("timeupdate", async () => {
  canvas.offsetHeight = video.offsetHeight + "px";
  canvas.offsetLeft = video.offsetLeft + "px";
  
  let model = await loadModel();
  let [pixels, xRatio, yRatio] = await preprocess(video, 640, 640)
  let res = model.predict(pixels);
  res = res.transpose([0, 2, 1]);
  const rawScores = res.slice([0, 0, 4], [-1, -1, -1]).squeeze(0); 

  let {values, indices} = tf.topk(rawScores.max(1), 20);  // filter our 20 highest probs

  let selected_detections = tf.gather(res, indices, axis = 1);
  
  let selected_boxes = selected_detections.slice([0, 0, 0], [-1, -1, 4]).squeeze(0);
  let selected_scores = selected_detections.slice([0, 0, 4], [-1, -1, -1]).max(-1).squeeze(0);
  let selected_classes = selected_detections.slice([0, 0, 4], [-1, -1, -1]).argMax(-1).squeeze(0);
  let nms = await tf.image.nonMaxSuppressionAsync(selected_boxes, selected_scores, 4, 0.6, 0.4);

  output_boxes = selected_boxes.gather(nms, 0).dataSync();
  output_scores = selected_scores.gather(nms, 0).dataSync();
  output_classes = selected_classes.gather(nms, 0).dataSync();


  console.log(output_boxes.length);
  console.log(output_scores.length);
  console.log(output_classes.length);

})

function drawVideoOnCanvas(){
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  // if(output_boxes != null)
  // {

  //     let [xc, yc, w, h] = output_boxes;
  //     let x_l = parseInt(xc-w/2);
  //     let y_l = parseInt(yc - h/2);
  //     ctx.strokeStyle = "green";
  //     ctx.lineWidth = "3";
  //     ctx.strokeRect(x_l, y_l, w, h);
  //     ctx.stroke();

  // }


  requestAnimationFrame(drawVideoOnCanvas);
}
  video.addEventListener("play", () => {
  drawVideoOnCanvas();
}); 



// Hello Triet





