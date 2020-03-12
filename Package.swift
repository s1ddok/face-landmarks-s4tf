// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CycleGAN",
    platforms: [.macOS(SupportedPlatform.MacOSVersion.v10_13)],
    products: [
        .executable(name: "CycleGAN", targets: ["CycleGAN"]),
    ],
    dependencies: [
        .package(url: "https://github.com/JohnSundell/Files", from: "4.0.0"),
        .package(name: "TensorBoardX", url: "https://github.com/t-ae/tensorboardx-s4tf.git", from: "0.1.2"),
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMinor(from: "0.0.1")),
    ],
    targets: [
        .target(
            name: "CycleGAN",
            dependencies: ["Files", "TensorBoardX", .product(name: "ArgumentParser", package: "swift-argument-parser")])
    ]
)
