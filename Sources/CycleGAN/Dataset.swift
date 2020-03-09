import TensorFlow
import Files
import Foundation

public class Images {
    struct Elements: TensorGroup {
        var image: Tensor<Float>
    }

    let dataset: Dataset<Elements>
    let count: Int

    public init(folder: Folder) throws {
        let imageFiles = folder.files(extensions: ["jpg"])

        var sourceData: [Float] = []

        var elements = 0

        for imageFile in imageFiles {
            let imageTensor = Image(jpeg: imageFile.url).tensor

            sourceData.append(contentsOf: imageTensor.scalars)

            elements += 1
        }

        let source = Tensor<Float>(shape: [elements, 256, 256, 3], scalars: sourceData) / 127.5 - 1.0
        self.dataset = .init(elements: Elements(image: source))
        self.count = elements
    }
}
