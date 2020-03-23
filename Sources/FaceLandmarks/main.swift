import TensorFlow
import Files
import Foundation
import TensorBoardX

let options = Options.parseOrExit()
let logDirURL = URL(fileURLWithPath: options.tensorboardLogdir, isDirectory: true)
let runId = currentRunId(logDir: logDirURL)
let writerURL = logDirURL.appendingPathComponent(String(runId), isDirectory: true)
let writer = SummaryWriter(logdir: writerURL)

print("Starting with run id: \(runId)")

let imageSize = 260

let datasetFolder = try Folder(path: options.datasetPath)

let trainDataset = try LabeledImages(folder: datasetFolder.subfolder(named: "train"), imageSize: (imageSize, imageSize))
let validationDataset = try LabeledImages(folder: datasetFolder.subfolder(named: "val"), imageSize: (imageSize, imageSize))

var model = EfficientNet(width: 1.1, depth: 1.0, resolution: imageSize, dropout: 0.3)
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

var step = 0

for epoch in 0..<epochs {
    if epoch == 10 {
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
            
            var totalMetric = Tensorf.zero
            var totalCount = Float.zero
            for batch in validationDataset.dataset.batched(1) {
                let predicted = model(batch.image)[0]
                let gt = batch.landmarks[0]
                
                let nme = normalizedMeanError(predicted: predicted, expected: gt)
                
                totalMetric += nme
                totalCount += 1
                
                if epoch < epochs / 2 && totalCount > 32 { break }
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

let start = Date()
var testStep = 0
for testBatch in validationDataset.dataset.batched(1) {
    let images = testBatch.image
    let gtLandmarks = testBatch.landmarks
    let predictedLandmarks = model(images)

    saveResultImageWithGT(image: images[0] * 0.5 + 0.5,
                          landmarks: predictedLandmarks[0] * Float(imageSize),
                          groundTruth: gtLandmarks[0] * Float(imageSize),
                          url: resultsFolder.url.appendingPathComponent("\(testStep).jpg"))
    testStep += 1
}
let end = Date()

print("Inference time: \(end.timeIntervalSince(start) / Double(testStep))")
