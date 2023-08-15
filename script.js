const video = document.getElementById("video");

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


let model = loadModel();
video.addEventListener("timeupdate", async () => {

  let model = await loadModel();
  let [pixels, xRatio, yRatio] = await preprocess(video, 640, 640)
  let res = model.predict(pixels);
  console.log(res)

})


