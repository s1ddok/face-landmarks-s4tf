import TensorFlow

/**
 ''Calculates normalized mean error (NME)
     Parameters:
         Predicted    (tensor of shape [N, 2]): predicted facial landmarks in cropped space.
         Expected    (tensor of shape [N, 2]): ground truth facial landmarks in cropped space.
     Returns:
         nme (float): normalized mean error
 */
func normalizedMeanError(predicted: Tensorf, expected: Tensorf) -> Tensorf {
    let gtx = (0..<68).map { expected.scalars[$0 + 0] }
    let gty = (0..<68).map { expected.scalars[$0 + 0] }
    
    let minX = gtx.min()!
    let maxX = gtx.max()!
    let minY = gty.min()!
    let maxY = gty.max()!
    
    let length = sqrt((maxX - minX) * (maxY - minY))
    
    var dis = predicted - expected
    dis = sqrt(pow(dis, 2).sum())
    
    return (dis / length).mean() * 100
}
