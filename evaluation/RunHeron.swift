import Foundation
import CoreML

// Native Core ML runner. Inputs are little-endian Float32 NCHW tensors prepared
// by evaluate_ibem.py. Indexed output reads honor MLMultiArray padding/strides.
@main
struct RunHeron {
    static func deviceName(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu: return "CPU"
        case .gpu: return "GPU"
        case .neuralEngine: return "ANE"
        @unknown default: return "unknown"
        }
    }

    static func writeJSON(_ object: Any, to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    static func predict(_ model: MLModel, _ provider: MLDictionaryFeatureProvider) throws -> any MLFeatureProvider {
        try model.prediction(from: provider)
    }

    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 5 else {
            fatalError("Usage: RunHeron model.mlmodelc input-directory output-directory cpu-ne|all|cpu")
        }
        let modelURL = URL(fileURLWithPath: args[1])
        let inputURL = URL(fileURLWithPath: args[2])
        let outputURL = URL(fileURLWithPath: args[3])
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
        let config = MLModelConfiguration()
        switch args[4] {
        case "cpu-ne": config.computeUnits = .cpuAndNeuralEngine
        case "all": config.computeUnits = .all
        case "cpu": config.computeUnits = .cpuOnly
        default: fatalError("Unknown compute unit mode")
        }
        let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: config)
        var operations = [[String: Any]]()
        func visit(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                if op.operatorName != "const" {
                    let usage = plan.deviceUsage(for: op)
                    operations.append([
                        "operator": op.operatorName,
                        "outputs": op.outputs.map(\.name),
                        "preferred_device": usage.map { deviceName($0.preferred) } ?? "unknown",
                        "supported_devices": usage.map { $0.supported.map(deviceName) } ?? [],
                        "estimated_cost_weight": plan.estimatedCost(of: op)?.weight ?? 0
                    ])
                }
                for child in op.blocks { visit(child) }
            }
        }
        if case .program(let program) = plan.modelStructure {
            for key in program.functions.keys.sorted() { visit(program.functions[key]!.block) }
        }
        var counts = [String: Int]()
        var weights = [String: Double]()
        for op in operations {
            let device = op["preferred_device"] as! String
            counts[device, default: 0] += 1
            weights[device, default: 0] += op["estimated_cost_weight"] as! Double
        }
        try writeJSON([
            "compute_units": args[4], "nonconstant_operation_counts": counts,
            "estimated_cost_weights": weights, "operations": operations,
            "note": "Compiler preferred-device assignments, not a runtime hardware trace. Cost weights are estimates, not measured time."
        ], to: outputURL.appendingPathComponent("compute_plan.json"))
        print("Compute plan: \(counts)")
        let loadStart = ProcessInfo.processInfo.systemUptime
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        let loadMS = (ProcessInfo.processInfo.systemUptime - loadStart) * 1000
        let files = try FileManager.default.contentsOfDirectory(at: inputURL, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "bin" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        var pages = [[String: Any]]()
        var warmupMS = [Double]()
        for (pageIndex, file) in files.enumerated() {
            let data = try Data(contentsOf: file)
            precondition(data.count == 3 * 640 * 640 * 4)
            let tensor = try MLMultiArray(shape: [1, 3, 640, 640], dataType: .float32)
            precondition(tensor.strides.map(\.intValue) == [1228800, 409600, 640, 1])
            data.withUnsafeBytes { bytes in tensor.dataPointer.copyMemory(from: bytes.baseAddress!, byteCount: data.count) }
            let provider = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": tensor])
            if pageIndex == 0 {
                for _ in 0..<3 {
                    let start = ProcessInfo.processInfo.systemUptime
                    _ = try predict(model, provider)
                    warmupMS.append((ProcessInfo.processInfo.systemUptime - start) * 1000)
                }
            }
            var durations = [Double]()
            var firstOutput: (any MLFeatureProvider)?
            for repeatIndex in 0..<5 {
                let start = ProcessInfo.processInfo.systemUptime
                let prediction = try predict(model, provider)
                durations.append((ProcessInfo.processInfo.systemUptime - start) * 1000)
                if repeatIndex == 0 { firstOutput = prediction }
            }
            var page: [String: Any] = ["page_id": file.deletingPathExtension().lastPathComponent, "prediction_ms": durations]
            for (name, columns) in [("logits", 17), ("pred_boxes", 4)] {
                let array = firstOutput!.featureValue(for: name)!.multiArrayValue!
                precondition(array.shape.map(\.intValue) == [1, 300, columns])
                var values = [[Double]]()
                for query in 0..<300 {
                    var row = [Double]()
                    for column in 0..<columns {
                        let value = array[[0, NSNumber(value: query), NSNumber(value: column)]].doubleValue
                        precondition(value.isFinite)
                        row.append(value)
                    }
                    values.append(row)
                }
                page[name] = values
                page[name + "_shape"] = array.shape.map(\.intValue)
                page[name + "_strides"] = array.strides.map(\.intValue)
                page[name + "_dtype"] = String(describing: array.dataType)
            }
            pages.append(page)
            print("\(file.lastPathComponent): \(durations) ms")
        }
        try writeJSON([
            "compute_units": args[4], "model_path": modelURL.path,
            "created_at": ISO8601DateFormatter().string(from: Date()),
            "model_load_ms_after_plan": loadMS, "warmup_ms": warmupMS, "pages": pages,
            "timing_boundary": "Synchronous MLModel.prediction call only, 3 warmups then 5 repeats per page; excludes preprocessing, tensor construction, postprocessing and file IO."
        ], to: outputURL.appendingPathComponent("raw_outputs.json"))
    }
}
