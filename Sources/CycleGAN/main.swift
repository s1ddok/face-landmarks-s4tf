import TensorFlow
import Files
import Foundation
import TensorBoardX

let options = Options.parseOrExit()

let logdir = URL(fileURLWithPath: options.tensorboardLogdir).appendingPathComponent(String(Int(Date().timeIntervalSince1970)))
//try? FileManager.default.removeItem(at: logdir)
//let writer = SummaryWriter(logdir: logdir)

let facadesFolder = try Folder(path: options.datasetPath)
let trainFolderA = try facadesFolder.subfolder(named: "trainA")
let trainFolderB = try facadesFolder.subfolder(named: "trainB")
let trainDatasetA = try Images(folder: trainFolderA)
let trainDatasetB = try Images(folder: trainFolderB)

var generatorG = NetG(inputChannels: 3, outputChannels: 3, ngf: 64, normalization: InstanceNorm2D.self, useDropout: false)
var generatorF = NetG(inputChannels: 3, outputChannels: 3, ngf: 64, normalization: InstanceNorm2D.self, useDropout: false)
var discriminatorX = NetD(inChannels: 6, lastConvFilters: 64)
var discriminatorY = NetD(inChannels: 6, lastConvFilters: 64)

let optimizerGF = Adam(for: generatorG, learningRate: 0.0002, beta1: 0.5)
let optimizerGG = Adam(for: generatorF, learningRate: 0.0002, beta1: 0.5)
let optimizerDX = Adam(for: discriminatorX, learningRate: 0.0002, beta1: 0.5)
let optimizerDY = Adam(for: discriminatorY, learningRate: 0.0002, beta1: 0.5)

let epochs = options.epochs
let batchSize = 1
let lambdaL1 = Tensorf(100)
let zeros = Tensorf(0)
let ones = Tensorf(1)
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
    var ganGLossCount: Float = 0
    
    for batch in zippedAB.batched(batchSize) {
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
        
            let sourceImages = croppedImages[0].expandingShape(at: 0)
            let targetImages = croppedImages[1].expandingShape(at: 0)
            
            let (fakeY, fakeYBackprop) = generatorG.appliedForBackpropagation(to: realX)
            let (cycledX, cycledXBackprop) = generatorF.appliedForBackpropagation(to: fakeY)
            
            let (fakeX, fakeXBackprop) = generatorF.appliedForBackpropagation(to: realY)
            let (cycledY, cycledYBackprop) = generatorG.appliedForBackpropagation(to: fakeX)
            
            // sameX and sameY are used for identity loss.
            let (sameX, sameXBackprop) = generatorF.appliedForBackpropagation(to: realX)
            let (sameY, sameYBackprop) = generatorG.appliedForBackpropagation(to: realY)
            
            let (discRealX, discRealXBackprop) = discriminatorX.appliedForBackpropagation(to: realX)
            let (discRealY, discRealYBackprop) = discriminatorY.appliedForBackpropagation(to: realY)
            
            let (discFakeX, discFakeXBackprop) = discriminatorX.appliedForBackpropagation(to: fakeX)
            let (discFakeY, discFakeYBackprop) = discriminatorY.appliedForBackpropagation(to: fakeY)
            
            let ones = Tensorf.one.broadcasted(like: discFakeY)
            let (genGLoss, genGGradient) = TensorFlow.valueWithGradient(at: discFakeY) {
                sigmoidCrossEntropy(logits: $0, labels: ones)
            }
            let (genFLoss, genFGradient) = TensorFlow.valueWithGradient(at: discFakeX) {
                sigmoidCrossEntropy(logits: $0, labels: ones)
            }

            let (cycleLossX, cycleLossXGradient) = TensorFlow.valueWithGradient(at: realX, cycledX) {
                abs($0 - $1).mean() * lambdaL1
            }
            let (cycleLossY, cycleLossYGradient) = TensorFlow.valueWithGradient(at: realY, cycledY) {
                abs($0 - $1).mean() * lambdaL1
            }
            
            let (identityLossX, identityLossXGradient) = TensorFlow.valueWithGradient(at: realX, sameX) {
                abs($0 - $1).mean() * lambdaL1 * 0.5
            }
            let (identityLossY, identityLossYGradient) = TensorFlow.valueWithGradient(at: realY, sameY) {
                abs($0 - $1).mean() * lambdaL1 * 0.5
            }
            
            let totalCycleLoss = cycleLossX + cycleLossY
            let totalCycleLossGradient = cycleLossXGradient.0 + cycleLossYGradient.0 + cycleLossXGradient.1 + cycleLossYGradient.1
            
            let (𝛁generatorG, _) = fakeYBackprop(totalCycleLossGradient + genGGradient + identityLossYGradient.0 + identityLossYGradient.1)
            optimizerGG.update(&generatorG, along: 𝛁generatorG)
            
            let (𝛁generatorF, _) = fakeXBackprop(totalCycleLossGradient + genFGradient + identityLossXGradient.0 + identityLossXGradient.1)
            optimizerGF.update(&generatorF, along: 𝛁generatorF)
            
            ganGLossTotal += totalCycleLoss + identityLossY + genGLoss
            ganGLossCount += 1
            
            let onesLikeDiscRealX = Tensorf.one.broadcasted(like: discRealX)
            let zerosLikeDiscFakeX = Tensorf(0).broadcasted(like: discRealX)
            let (discXLoss, discXLossGradient) = TensorFlow.valueWithGradient(at: discRealX, discFakeX) {
                (sigmoidCrossEntropy(logits: $0, labels: onesLikeDiscRealX) + sigmoidCrossEntropy(logits: $1, labels: zerosLikeDiscFakeX)) * 0.5
            }
            
            let (discYLoss, discYLossGradient) = TensorFlow.valueWithGradient(at: discRealY, discFakeY) {
                (sigmoidCrossEntropy(logits: $0, labels: onesLikeDiscRealX) + sigmoidCrossEntropy(logits: $1, labels: zerosLikeDiscFakeX)) * 0.5
            }
            
            let (𝛁discriminatorX, _) = discRealXBackprop(discXLossGradient.0 + discXLossGradient.1)
            optimizerDX.update(&discriminatorX, along: 𝛁discriminatorX)
            let (𝛁discriminatorY, _) = discRealYBackprop(discYLossGradient.0 + discYLossGradient.1)
            optimizerDY.update(&discriminatorX, along: 𝛁discriminatorY)
        }
    }
    
    print("Gan G loss: \(ganGLossTotal / ganGLossCount)")
}
