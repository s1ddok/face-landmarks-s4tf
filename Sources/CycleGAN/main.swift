import TensorFlow
import Files
import Foundation
import TensorBoardX

let options = Options.parseOrExit()

let logdir = URL(fileURLWithPath: options.tensorboardLogdir).appendingPathComponent(String(Int(Date().timeIntervalSince1970)))
//try? FileManager.default.removeItem(at: logdir)
let writer = SummaryWriter(logdir: logdir)

let facadesFolder = try Folder(path: options.datasetPath)
let trainFolderA = try facadesFolder.subfolder(named: "trainA")
let trainFolderB = try facadesFolder.subfolder(named: "trainB")
let trainDatasetA = try Images(folder: trainFolderA)
let trainDatasetB = try Images(folder: trainFolderB)

var generatorG = NetG(inputChannels: 3, outputChannels: 3, ngf: 64, normalization: InstanceNorm2D.self, useDropout: false)
var generatorF = NetG(inputChannels: 3, outputChannels: 3, ngf: 64, normalization: InstanceNorm2D.self, useDropout: false)
var discriminatorX = NetD(inChannels: 3, lastConvFilters: 64)
var discriminatorY = NetD(inChannels: 3, lastConvFilters: 64)

let optimizerGF = Adam(for: generatorG, learningRate: 0.0002, beta1: 0.5)
let optimizerGG = Adam(for: generatorF, learningRate: 0.0002, beta1: 0.5)
let optimizerDX = Adam(for: discriminatorX, learningRate: 0.0002, beta1: 0.5)
let optimizerDY = Adam(for: discriminatorY, learningRate: 0.0002, beta1: 0.5)

let epochs = options.epochs
let batchSize = 1
let lambdaL1 = Tensorf(10)
let zeros = Tensorf(0)
let ones = Tensorf.one
let gpuIndex = options.gpuIndex

for epoch in 0..<epochs {
    print("Epoch \(epoch) started at: \(Date())")
    Context.local.learningPhase = .training
    
    let trainingAShuffled = trainDatasetA.dataset
                                         .shuffled(sampleCount: trainDatasetA.count,
                                                   randomSeed: Int64(epoch))
    let trainingBShuffled = trainDatasetB.dataset
                                         .shuffled(sampleCount: trainDatasetB.count,
                                                   randomSeed: Int64(epoch))
    let zippedAB = zip(trainingAShuffled, trainingBShuffled)

    var ganGLossTotal = Tensorf(0)
    var ganFLossTotal = Tensorf(0)
    
    var discXLossTotal = Tensorf(0)
    var discYLossTotal = Tensorf(0)
    
    var totalLossCount: Float = 0
    
    for batch in zippedAB.batched(batchSize) {
        Context.local.learningPhase = .training
        let realX = batch.first.image
        let realY = batch.second.image
    
        // we do it outside of GPU scope so that dataset shuffling happens on CPU side
        let concatanatedImages = realX.concatenated(with: realY)
        
        withDevice(.gpu, gpuIndex) {
            let scaledImages = _Raw.resizeNearestNeighbor(images: concatanatedImages, size: [286, 286])
            var croppedImages = scaledImages.slice(lowerBounds: Tensor<Int32>([0, Int32(random() % 30), Int32(random() % 30), 0]),
                                                   sizes: [2, 256, 256, 3])
            if random() % 2 == 0 {
                croppedImages = _Raw.reverse(croppedImages, dims: [false, false, true, false])
            }
        
            let realX = croppedImages[0].expandingShape(at: 0)
            let realY = croppedImages[1].expandingShape(at: 0)
            
            let onesd = ones.broadcasted(to: [1, 30, 30, 1])
            let zerosd = zeros.broadcasted(to: [1, 30, 30, 1])
            
            let 𝛁generatorG = TensorFlow.gradient(at: generatorG) { g -> Tensorf in
                let fakeY = g(realX)
                let cycledX = generatorF(fakeY)
                let fakeX = generatorF(realY)
                let cycledY = g(fakeX)
                
                let cycleConsistencyLoss = (abs(realX - cycledX).mean() +
                                            abs(realY - cycledY).mean()) * lambdaL1

                let discFakeY = discriminatorY(fakeY)
                let generatorLoss = sigmoidCrossEntropy(logits: discFakeY, labels: onesd)
                
                let sameY = g(realY)
                let identityLoss = abs(sameY - realY).mean() * lambdaL1 * 0.5
                
                let totalLoss = cycleConsistencyLoss + generatorLoss + identityLoss
                ganGLossTotal += totalLoss
                                                                    
                return totalLoss
            }
            
            let 𝛁generatorF = TensorFlow.gradient(at: generatorF) { g -> Tensorf in
                let fakeX = g(realY)
                let cycledY = generatorG(fakeX)
                let fakeY = generatorG(realX)
                let cycledX = g(fakeY)
                
                let cycleConsistencyLoss = (abs(realY - cycledY).mean()
                                            + abs(realX - cycledX).mean()) * lambdaL1

                let discFakeX = discriminatorX(fakeX)
                let generatorLoss = sigmoidCrossEntropy(logits: discFakeX, labels: onesd)
                
                let sameX = g(realX)
                let identityLoss = abs(sameX - realX).mean() * lambdaL1 * 0.5
                
                let totalLoss = cycleConsistencyLoss + generatorLoss + identityLoss
                ganFLossTotal += totalLoss
                return totalLoss
            }
            
            let 𝛁discriminatorX = TensorFlow.gradient(at: discriminatorX) { d -> Tensorf in
                let fakeX = generatorG(realX)
                let discFakeX = d(fakeX)
                let discRealX = d(realX)
                
                let totalLoss = 0.5 * (sigmoidCrossEntropy(logits: discFakeX, labels: zerosd)
                                       + sigmoidCrossEntropy(logits: discRealX, labels: onesd))
                discXLossTotal += totalLoss
                return totalLoss
            }
            
            let 𝛁discriminatorY = TensorFlow.gradient(at: discriminatorY) { d -> Tensorf in
                let fakeY = generatorF(realY)
                let discFakeY = d(fakeY)
                let discRealY = d(realY)
                
                let totalLoss = 0.5 * (sigmoidCrossEntropy(logits: discFakeY, labels: zerosd)
                                       + sigmoidCrossEntropy(logits: discRealY, labels: onesd))
                discYLossTotal += totalLoss
                return totalLoss
            }
            
            optimizerGG.update(&generatorG, along: 𝛁generatorG)
            optimizerGF.update(&generatorF, along: 𝛁generatorF)
            optimizerDX.update(&discriminatorX, along: 𝛁discriminatorX)
            optimizerDY.update(&discriminatorY, along: 𝛁discriminatorY)
            
            totalLossCount += 1
        }
        
        if Int(totalLossCount) % 50 == 0 {
            Context.local.learningPhase = .inference
            for testBatch in trainDatasetA.dataset.batched(1) {
                let result = generatorG(testBatch.image)
                let images = result * 0.5 + 0.5
                
                let image = Image(tensor: images[0] * 255)
                
                let currentURL = Folder.current.url.appendingPathComponent("\(epoch).jpg")
                
                image.save(to: currentURL, format: .rgb)
                
                break
            }
        }
    }
    
    let generatorLossG = ganGLossTotal / totalLossCount
    let generatorLossF = ganFLossTotal / totalLossCount
    let discriminatorLossX = discXLossTotal / totalLossCount
    let discriminatorLossY = discYLossTotal / totalLossCount
    
    writer.addScalars(mainTag: "train_loss",
                      taggedScalars: [
                        "GeneratorG": generatorLossG.scalars[0],
                        "GeneratorF": generatorLossF.scalars[0],
                        "DiscriminatorX": discriminatorLossX.scalars[0],
                        "DiscriminatorY": discriminatorLossY.scalars[0]
                      ],
                      globalStep: epoch)
    
    print("GeneratorG \(generatorLossG)")
    print("GeneratorF \(generatorLossF)")
    print("DiscriminatorX \(discriminatorLossX)")
    print("DiscriminatorY \(discriminatorLossY)")
}
