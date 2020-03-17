import TensorFlow

public struct NetG<NT: FeatureChannelInitializable>: Layer where NT.TangentVector.VectorSpaceScalar == Float, NT.Input == Tensorf, NT.Output == Tensorf {
    
    var module: UNetSkipConnectionOutermost<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnection<UNetSkipConnectionInnermost<NT>, NT>, NT>, NT>, NT>, NT>, NT>>


    public init(inputChannels: Int,
                outputChannels: Int,
                ngf: Int,
                normalization: NT.Type,
                useDropout: Bool = false) {
        let firstBlock = UNetSkipConnectionInnermost(inChannels: ngf * 8,
                                                     innerChannels: ngf * 8,
                                                     outChannels: ngf * 8,
                                                     normalization: normalization)
        
        let module1 = UNetSkipConnection(inChannels: ngf * 8,
                                         innerChannels: ngf * 8,
                                         outChannels: ngf * 8,
                                         submodule: firstBlock,
                                         normalization: normalization,
                                         useDropOut: useDropout)
        let module2 = UNetSkipConnection(inChannels: ngf * 8,
                                         innerChannels: ngf * 8,
                                         outChannels: ngf * 8,
                                         submodule: module1,
                                         normalization: normalization,
                                         useDropOut: useDropout)
        let module3 = UNetSkipConnection(inChannels: ngf * 8,
                                         innerChannels: ngf * 8,
                                         outChannels: ngf * 8,
                                         submodule: module2,
                                         normalization: normalization,
                                         useDropOut: useDropout)

        let module4 = UNetSkipConnection(inChannels: ngf * 4,
                                         innerChannels: ngf * 8, outChannels: ngf * 4,
                                         submodule: module3,
                                         normalization: normalization,
                                         useDropOut: useDropout)
        let module5 = UNetSkipConnection(inChannels: ngf * 2,
                                         innerChannels: ngf * 4,
                                         outChannels: ngf * 2,
                                         submodule: module4,
                                         normalization: normalization,
                                         useDropOut: useDropout)
        let module6 = UNetSkipConnection(inChannels: ngf,
                                         innerChannels: ngf * 2,
                                         outChannels: ngf,
                                         submodule: module5,
                                         normalization: normalization,
                                         useDropOut: useDropout)

        self.module = UNetSkipConnectionOutermost(inChannels: inputChannels, innerChannels: ngf, outChannels: outputChannels,
                                                  submodule: module6)
    }

    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        return self.module(input)
    }
}

public struct ResNetGenerator<NT: FeatureChannelInitializable>: Layer where NT.TangentVector.VectorSpaceScalar == Float, NT.Input == Tensorf, NT.Output == Tensorf {
    
    var conv1: Conv2D<Float>
    var norm1: NT
    
    var conv2: Conv2D<Float>
    var norm2: NT
    
    var conv3: Conv2D<Float>
    var norm3: NT
    
    var resblocks: [ResnetBlock<NT>]
    
    var upConv1: TransposedConv2D<Float>
    var upNorm1: NT
    
    var upConv2: TransposedConv2D<Float>
    var upNorm2: NT
    
    var lastConv: Conv2D<Float>
    
    public init(inputChannels: Int,
                outputChannels: Int,
                blocks: Int,
                ngf: Int,
                normalization: NT.Type,
                useDropout: Bool = false) {
        self.norm1 = .init(featureCount: ngf)
        let useBias = self.norm1 is InstanceNorm2D<Float>
        
        let filterInit: (TensorShape) -> Tensorf = { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) }
        let biasInit: (TensorShape) -> Tensorf = useBias ? filterInit : zeros()
        
        self.conv1 = .init(filterShape: (7, 7,
                                         inputChannels, ngf),
                           strides: (1, 1),
                           filterInitializer: filterInit,
                           biasInitializer: biasInit)
        
        var mult = 1
        
        self.conv2 = .init(filterShape: (3, 3,
                                         ngf * mult, ngf * mult * 2),
                           strides: (2, 2),
                           padding: .same,
                           filterInitializer: filterInit,
                           biasInitializer: biasInit)
        self.norm2 = .init(featureCount: ngf * mult * 2)
        
        mult = 2
        
        self.conv3 = .init(filterShape: (3, 3,
                                         ngf * mult, ngf * mult * 2),
                           strides: (2, 2),
                           padding: .same,
                           filterInitializer: filterInit,
                           biasInitializer: biasInit)
        self.norm3 = .init(featureCount: ngf * mult * 2)
        
        mult = 4
        
        self.resblocks = (0..<blocks).map { _ in
            ResnetBlock(channels: ngf * mult,
                        paddingMode: .reflect,
                        normalization: normalization,
                        useDropOut: useDropout,
                        filterInit: filterInit,
                        biasInit: biasInit)
        }
        
        mult = 4
        
        self.upConv1 = .init(filterShape: (3, 3, ngf * mult / 2, ngf * mult),
                             strides: (2, 2),
                             padding: .same,
                             filterInitializer: filterInit,
                             biasInitializer: biasInit)
        self.upNorm1 = .init(featureCount: ngf * mult / 2)
        
        mult = 2
        
        self.upConv2 = .init(filterShape: (3, 3, ngf * mult / 2, ngf * mult),
                             strides: (2, 2),
                             padding: .same,
                             filterInitializer: filterInit,
                             biasInitializer: biasInit)
        self.upNorm2 = .init(featureCount: ngf * mult / 2)
        
        self.lastConv = .init(filterShape: (7, 7, ngf, outputChannels),
                              padding: .same,
                              filterInitializer: filterInit,
                              biasInitializer: biasInit)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        var x = input.padded(forSizes: [(0, 0), (3, 3), (3, 3), (0, 0)], mode: .reflect)
        x = x.sequenced(through: conv1, norm1)
        x = relu(x)
        x = x.sequenced(through: conv2, norm2)
        x = relu(x)
        x = x.sequenced(through: conv3, norm3)
        x = relu(x)
        
        x = resblocks(x)
        
        x = x.sequenced(through: upConv1, upNorm1)
        x = relu(x)
        x = x.sequenced(through: upConv2, upNorm2)
        x = relu(x)
        
        x = lastConv(x)
        x = tanh(x)
        
        return x
    }
}

