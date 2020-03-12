import Foundation

func currentRunId(logDir: URL, runIdFileName: String = ".run") -> Int {
    var runId = 0
    
    let runIdURL = logDir.appendingPathComponent(runIdFileName, isDirectory: false)
    if let data = try? Data(contentsOf: runIdURL) {
        runId = data.withUnsafeBytes { $0.baseAddress?.assumingMemoryBound(to: Int.self).pointee ?? 0 }
    }
    
    runId += 1
    
    let data = Data(bytes: &runId, count: MemoryLayout<Int>.stride)
    
    try? data.write(to: runIdURL)
    
    return runId
}
