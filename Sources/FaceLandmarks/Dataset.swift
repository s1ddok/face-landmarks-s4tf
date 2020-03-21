import TensorFlow
import Files
import Foundation

public class LabeledImages {
    struct Elements: TensorGroup {
        var image: Tensor<Float>
        var landmarks: Tensor<Float>
    }

    let dataset: Dataset<Elements>
    let count: Int

    public init(folder: Folder, imageSize: (Int, Int)) throws {
        let imageFiles = folder.files(extensions: ["jpg"])

        var imageArray: [Float] = []
        var landmarksArray: [Float] = []

        var elements = 0
        
        let decoder = JSONDecoder()

        for imageFile in imageFiles {
            let imageTensor = Image(jpeg: imageFile.url).resized(to: imageSize).tensor

            imageArray.append(contentsOf: imageTensor.scalars)
            
            let landmarksPath = imageFile.url.path.replacingOccurrences(of: ".jpg", with: "_pts.landmarks")
            let landmarksData = try Data(contentsOf: URL(fileURLWithPath: landmarksPath))
            let landmarks = try decoder.decode(Tensorf.self, from: landmarksData)
            
            landmarksArray.append(contentsOf: landmarks.scalars)

            elements += 1
        }

        let source = Tensorf(shape: [elements, imageSize.0, imageSize.1, 3], scalars: imageArray) / 127.5 - 1.0
        let landmarks = Tensorf(shape: [elements, 68 * 2], scalars: landmarksArray)
        self.dataset = .init(elements: Elements(image: source, landmarks: landmarks))
        
        self.count = elements
    }
}
