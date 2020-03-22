import TensorFlow

fileprivate var depthCoefficient: Float = 1.0
fileprivate func roundBlockDepthUp(blockCount: Int) -> Int {
    /// Multiply + round up the number of blocks based on global depth multiplier
    var newFilterCount = depthCoefficient * Float(blockCount)
    newFilterCount.round(.up)
    return Int(newFilterCount)
}

fileprivate var widthCoefficient: Float = 1.0
fileprivate func roundFilterCountDown(filter: Int, depthDivisor: Float = 8.0) -> Int {
    /// Multiply + round down the number of filters based on global width multiplier
    let filterMult = Float(filter) * widthCoefficient
    let filterAdd = Float(filterMult) + (depthDivisor / 2.0)
    var div = filterAdd / depthDivisor
    div.round(.down)
    div = div * Float(depthDivisor)
    var newFilterCount = max(1, Int(div))
    if newFilterCount < Int(0.9 * Float(filter)) {
        newFilterCount += Int(depthDivisor)
    }
    return Int(newFilterCount)
}

fileprivate func roundFilterPair(filters: (Int, Int)) -> (Int, Int) {
    return (roundFilterCountDown(filter: filters.0), roundFilterCountDown(filter: filters.1))
}

public struct InitialMBConvBlock: Layer {
    public var dConv: DepthwiseConv2D<Float>
    public var seReduceConv: Conv2D<Float>
    public var seExpandConv: Conv2D<Float>
    public var conv2: Conv2D<Float>

    public init(filters: (Int, Int)) {
        let filterMult = roundFilterPair(filters: filters)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, filterMult.0, 1),
            strides: (1, 1),
            padding: .same)
        seReduceConv = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, roundFilterCountDown(filter: 8)),
            strides: (1, 1),
            padding: .same)
        seExpandConv = Conv2D<Float>(
            filterShape: (1, 1, roundFilterCountDown(filter: 8), filterMult.0),
            strides: (1, 1),
            padding: .same)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, filterMult.1),
            strides: (1, 1),
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let dw = swish(dConv(input))
        let se = sigmoid(seExpandConv(swish(seReduceConv(dw))))
        return conv2(se)
    }
}

public struct MBConvBlock: Layer {
    @noDerivative public var addResLayer: Bool
    @noDerivative public var strides: (Int, Int)
    @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))

    public var conv1: Conv2D<Float>
    public var dConv: DepthwiseConv2D<Float>
    public var seReduceConv: Conv2D<Float>
    public var seExpandConv: Conv2D<Float>
    public var conv2: Conv2D<Float>

    public init(
        filters: (Int, Int),
        depthMultiplier: Int = 6,
        strides: (Int, Int) = (1, 1),
        kernel: (Int, Int) = (3, 3)
    ) {
        self.strides = strides
        self.addResLayer = filters.0 == filters.1 && strides == (1, 1)

        let filterMult = roundFilterPair(filters: filters)
        let hiddenDimension = filterMult.0 * depthMultiplier
        let reducedDimension = max(1, Int(filterMult.0 / 4))
        conv1 = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, hiddenDimension),
            strides: (1, 1),
            padding: .same)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (kernel.0, kernel.1, hiddenDimension, 1),
            strides: strides,
            padding: strides == (1, 1) ? .same : .valid)
        seReduceConv = Conv2D<Float>(
            filterShape: (1, 1, hiddenDimension, reducedDimension),
            strides: (1, 1),
            padding: .same)
        seExpandConv = Conv2D<Float>(
            filterShape: (1, 1, reducedDimension, hiddenDimension),
            strides: (1, 1),
            padding: .same)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, hiddenDimension, filterMult.1),
            strides: (1, 1),
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let pw = swish(conv1(input))
        var dw: Tensor<Float>
        if self.strides == (1, 1) {
            dw = swish(dConv(pw))
        } else {
            dw = swish(zeroPad(dConv(pw)))
        }
        let se = sigmoid(seExpandConv(swish(seReduceConv(dw))))
        let pwLinear = conv2(se)

        if self.addResLayer {
            return input + pwLinear
        } else {
            return pwLinear
        }
    }
}

public struct MBConvBlockStack: Layer {
    var blocks: [MBConvBlock] = []

    public init(
        filters: (Int, Int),
        initialStrides: (Int, Int) = (2, 2),
        kernel: (Int, Int) = (3, 3),
        blockCount: Int
    ) {
        let blockMult = roundBlockDepthUp(blockCount: blockCount)
        self.blocks = [MBConvBlock(filters: (filters.0, filters.1),
            strides: initialStrides, kernel: kernel)]
        for _ in 1..<blockMult {
            self.blocks.append(MBConvBlock(filters: (filters.1, filters.1), kernel: kernel))
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return blocks.differentiableReduce(input) { $1($0) }
    }
}

public struct EfficientNet: Layer {
    public var inputConv: Conv2D<Float>
    public var initialMBConv: InitialMBConvBlock

    public var residualBlockStack1: MBConvBlockStack
    public var residualBlockStack2: MBConvBlockStack
    public var residualBlockStack3: MBConvBlockStack
    public var residualBlockStack4: MBConvBlockStack
    public var residualBlockStack5: MBConvBlockStack
    public var residualBlockStack6: MBConvBlockStack

    public var finalConv: Conv2D<Float>
    public var avgPool = GlobalAvgPool2D<Float>()

    /// default settings are efficientnetB0 (baseline) network
    /// resolution is here to show what the network can take as input, it doesn't set anything!
    public init(
        width: Float = 1.0,
        depth: Float = 1.0,
        resolution: Int = 224,
        dropout: Double = 0.2
    ) {
        depthCoefficient = depth
        widthCoefficient = width

        inputConv = Conv2D<Float>(
            filterShape: (3, 3, 3, roundFilterCountDown(filter: 32)),
            strides: (2, 2),
            padding: .same)

        // global filter resizing happens at block layer
        initialMBConv = InitialMBConvBlock(filters: (32, 16))

        // global block resizing happens at stack layer
        residualBlockStack1 = MBConvBlockStack(filters: (16, 24), blockCount: 2)
        residualBlockStack2 = MBConvBlockStack(filters: (24, 40), kernel: (5, 5),
            blockCount: 2)
        residualBlockStack3 = MBConvBlockStack(filters: (40, 80), blockCount: 3)
        residualBlockStack4 = MBConvBlockStack(filters: (80, 112), initialStrides: (1, 1),
            kernel: (5, 5), blockCount: 3)
        residualBlockStack5 = MBConvBlockStack(filters: (112, 192), kernel: (5, 5),
            blockCount: 4)
        residualBlockStack6 = MBConvBlockStack(filters: (192, 320), initialStrides: (1, 1),
            blockCount: 1)

        finalConv = Conv2D<Float>(
            filterShape: (1, 1, roundFilterCountDown(filter: 320), 136),
            strides: (1, 1),
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: inputConv, initialMBConv)
        let backbone = convolved.sequenced(through: residualBlockStack1, residualBlockStack2,
            residualBlockStack3, residualBlockStack4, residualBlockStack5, residualBlockStack6)
        return backbone.sequenced(through: finalConv, avgPool)
    }
}
