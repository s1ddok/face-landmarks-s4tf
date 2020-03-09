import TensorFlow

public typealias Tensorf = Tensor<Float>

#if os(macOS)
func random() -> UInt32 {
    arc4random()
}
#endif

public protocol FeatureChannelInitializable: Layer {
    init(featureCount: Int)
}

extension BatchNorm: FeatureChannelInitializable {
    public init(featureCount: Int) {
        self.init(featureCount: featureCount, axis: -1, momentum: 0.99, epsilon: 0.001)
    }
}

extension InstanceNorm2D: FeatureChannelInitializable {
    public init(featureCount: Int) {
        self.init(featureCount: featureCount, epsilon: Tensor(1e-5))
    }
}
