import PythonKit
import TensorFlow
import Foundation

let plt = Python.import("matplotlib.pyplot")

public func saveResultImageWithGT(image: Tensor<Float>, landmarks: Tensor<Float>, groundTruth: Tensor<Float>, url: URL) {
    let dpi: Float = 300
    let figure = plt.figure(figsize: [Float(image.shape[0]) / dpi, Float(image.shape[1]) / dpi], dpi: dpi)
    let img = plt.subplot(1, 1, 1)
    img.axis("off")
    let x = image.makeNumpyArray()
    img.imshow(x)
           
    let lmx = (0..<68).map { landmarks.scalars[$0 * 2 + 0] }
    let lmy = (0..<68).map { landmarks.scalars[$0 * 2 + 1] }
    img.scatter(lmx.makeNumpyArray(), lmy.makeNumpyArray(), s: 0.1)
    
    let gtlmx = (0..<68).map { groundTruth.scalars[$0 * 2 + 0] }
    let gtlmy = (0..<68).map { groundTruth.scalars[$0 * 2 + 1] }
    img.scatter(gtlmx.makeNumpyArray(), gtlmy.makeNumpyArray(), s: 0.1, c: "#F00")
    
    plt.savefig(url.path, bbox_inches: "tight", pad_inches: 0, dpi: dpi)
    plt.close(figure)
}
