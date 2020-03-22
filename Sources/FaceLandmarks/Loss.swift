import TensorFlow

@differentiable(wrt: predicted)
func wingLoss(predicted: Tensorf, expected: Tensorf, w: Float = 10, eps: Float = 2) -> Tensorf {
    let x = abs(predicted - expected)
    let center = w * log(1 + x / eps)
    let c = w - w * log(1 + w / eps)
    let wings = x - c
    
    // There is a more elegant way but it is not differentiable as of now
    // Tensorf(x .< w) * center + Tensorf(x .>= w) * wings
    var loss = max(sign(w - x), 0) * center
    loss = loss + max(sign(x - w), 0) * wings
    return loss.mean()
}

