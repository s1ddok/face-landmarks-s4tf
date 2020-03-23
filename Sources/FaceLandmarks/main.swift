import TensorFlow
import Files
import Foundation
import TensorBoardX
import PythonKit

let options = Options.parseOrExit()
let logDirURL = URL(fileURLWithPath: options.tensorboardLogdir, isDirectory: true)
let runId = currentRunId(logDir: logDirURL)
let writerURL = logDirURL.appendingPathComponent(String(runId), isDirectory: true)
let writer = SummaryWriter(logdir: writerURL)

print("Starting with run id: \(runId)")

let imageSize = 260

let datasetFolder = try Folder(path: options.datasetPath)

let trainDataset = try LabeledImages(folder: datasetFolder, imageSize: (imageSize, imageSize))
let validationDataset = trainDataset.dataset

var model = EfficientNet(width: 1.1, depth: 1.2, resolution: imageSize, dropout: 0.3)
let optimizer = Adam(for: model, learningRate: 0.0005)

let epochs = options.epochs
let batchSize = options.batchSize

var sampleImage: Tensorf = .zero
var sampleLandmarks: Tensorf = .zero
    
for batch in trainDataset.dataset.batched(1) {
    sampleImage = batch.image
    sampleLandmarks = batch.landmarks
    break
}

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
    img.scatter(lmx.makeNumpyArray(), lmy.makeNumpyArray(), s: 0.2)
    
    let gtlmx = (0..<68).map { groundTruth.scalars[$0 * 2 + 0] }
    let gtlmy = (0..<68).map { groundTruth.scalars[$0 * 2 + 1] }
    img.scatter(gtlmx.makeNumpyArray(), gtlmy.makeNumpyArray(), s: 0.2, c: "#F00")
    
    plt.savefig(url.path, bbox_inches: "tight", pad_inches: 0, dpi: dpi)
    plt.close(figure)
}

var step = 0

for epoch in 0..<epochs {
    if epoch == 1 {
        optimizer.learningRate = 0.0001
    }
    
    print("Epoch \(epoch) started at: \(Date())")

    let trainingAShuffled = trainDataset.dataset
                                        .shuffled(sampleCount: trainDataset.count,
                                                  randomSeed: Int64(epoch))

    for batch in trainingAShuffled.batched(batchSize) {
        Context.local.learningPhase = .training
        
        let images = batch.image
        let landmarks = batch.landmarks
        
        let (loss, 𝛁model) = valueWithGradient(at: model) { model -> Tensorf in
            let predictedLandmarks = model(images)
            
            let loss = wingLoss(predicted: predictedLandmarks * Float(imageSize),
                                expected: landmarks * Float(imageSize))
            
            return loss
        }
        
        optimizer.update(&model, along: 𝛁model)
        
        writer.addScalar(tag: "train_loss",
                         scalar: loss.scalars[0],
                         globalStep: step)
        
        if step % options.sampleLogPeriod == 0 {
            Context.local.learningPhase = .inference
            
            let predictedLandmarks = model(sampleImage)[0]
            
            saveResultImageWithGT(image: sampleImage[0] * 0.5 + 0.5,
                                  landmarks: predictedLandmarks * Float(imageSize),
                                  groundTruth: sampleLandmarks * Float(imageSize),
                                  url: Folder.current.url.appendingPathComponent("intermediate\(step).jpg"))
            
            var totalMetric = Tensorf.zero
            var totalCount = Float.zero
            for batch in validationDataset.batched(1) {
                let predicted = model(batch.image)[0]
                let gt = batch.landmarks[0]
                
                let nme = normalizedMeanError(predicted: predicted, expected: gt)
                
                totalMetric += nme
                totalCount += 1
                
                if totalCount == 64 { break }
            }
            
            let metric = totalMetric / totalCount
            print("Metric: \(metric.scalar!), step: \(step)")
            writer.addScalar(tag: "metric", scalar: metric.scalar!, globalStep: step)
        }
        
        writer.flush()
        step += 1
    }
}

// MARK: Final test
Context.local.learningPhase = .inference

let resultsFolder = try Folder.current.createSubfolderIfNeeded(at: "results")

var testStep = 0
for testBatch in trainDataset.dataset.batched(1) {
    let images = testBatch.image
    let gtLandmarks = testBatch.landmarks
    let predictedLandmarks = model(images)

    saveResultImageWithGT(image: images[0] * 0.5 + 0.5,
                          landmarks: predictedLandmarks[0] * Float(imageSize),
                          groundTruth: gtLandmarks[0] * Float(imageSize),
                          url: resultsFolder.url.appendingPathComponent("\(testStep).jpg"))
    testStep += 1
    
    if testStep == 100 {
        break
    }
}
