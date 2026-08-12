import Foundation
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "models")

/// Unified model manager tracking download state for all 3 models (LLM, STT, TTS).
/// All downloads use the vendored ModelDownloader for resume support and integrity checks.
@MainActor
final class UnifiedModelManager: ObservableObject {
    @Published var modelStates: [ModelRegistry.ModelType: ModelState] = [:]
    @Published var error: String?

    /// Known LLM file size (Q4_K_S variant).
    private static let llmExpectedSize: Int64 = 1_261_854_880

    enum ModelState: Equatable {
        case notDownloaded
        case downloading(progress: Double)
        case downloaded
        case error(String)

        var isReady: Bool {
            if case .downloaded = self { return true }
            return false
        }

        var progress: Double {
            if case .downloading(let p) = self { return p }
            if case .downloaded = self { return 1.0 }
            return 0
        }
    }

    var allModelsReady: Bool {
        ModelRegistry.ModelType.allCases.allSatisfy { modelStates[$0]?.isReady == true }
    }

    var llmModelPath: String? {
        ModelRegistry.llmModelPath
    }

    init() {
        checkExistingModels()
    }

    // MARK: - Integrity checks (fast path for app launch)

    private func checkExistingModels() {
        // LLM: check file exists with expected size (±1% tolerance).
        // Full SHA256 verification (slow, ~1.26 GB read) is done post-download only.
        if let path = ModelRegistry.llmModelPath {
            let url = URL(fileURLWithPath: path)
            let attrs = try? FileManager.default.attributesOfItem(atPath: path)
            let size = (attrs?[.size] as? Int64) ?? 0
            if abs(size - Self.llmExpectedSize) < Self.llmExpectedSize / 100 {
                modelStates[.llm] = .downloaded
            } else {
                try? FileManager.default.removeItem(at: url)
                modelStates[.llm] = .notDownloaded
            }
        } else {
            modelStates[.llm] = .notDownloaded
        }

        // STT: check key model file exists with non-trivial size
        let sttDir = ModelRegistry.modelsDirectory
            .appendingPathComponent("parakeet-eou-streaming/320ms")
        let sttEncoderBin = sttDir.appendingPathComponent(
            "streaming_encoder.mlmodelc/coremldata.bin")
        if fileHasContent(sttEncoderBin) {
            modelStates[.stt] = .downloaded
        } else {
            try? FileManager.default.removeItem(at: sttDir)
            modelStates[.stt] = .notDownloaded
        }

        // TTS: check key model file exists with non-trivial size.
        // Must match path that FluidAudio's PocketTtsManager expects:
        // ~/Library/Caches/fluidaudio/Models/pocket-tts/
        let ttsDir = ModelRegistry.ttsModelsDirectory
            .appendingPathComponent("pocket-tts")
        let ttsCondBin = ttsDir.appendingPathComponent("cond_step.mlmodelc/coremldata.bin")
        if fileHasContent(ttsCondBin) {
            modelStates[.tts] = .downloaded
        } else {
            try? FileManager.default.removeItem(at: ttsDir)
            modelStates[.tts] = .notDownloaded
        }
    }

    private func fileHasContent(_ url: URL) -> Bool {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
            let size = attrs[.size] as? Int64, size > 1024
        else {
            return false
        }
        return true
    }

    // MARK: - Download

    func downloadAllModels() async {
        await withTaskGroup(of: Void.self) { group in
            if modelStates[.llm]?.isReady != true {
                group.addTask { await self.downloadLLM() }
            }
            if modelStates[.stt]?.isReady != true {
                group.addTask { await self.downloadSTT() }
            }
            if modelStates[.tts]?.isReady != true {
                group.addTask { await self.downloadTTS() }
            }
        }
    }

    func retryModel(_ type: ModelRegistry.ModelType) async {
        // Clean up any partial files before retry
        switch type {
        case .llm:
            let dest = ModelRegistry.modelsDirectory
                .appendingPathComponent(ModelRegistry.llmFilename)
            try? FileManager.default.removeItem(at: dest)
        case .stt:
            let sttDir = ModelRegistry.modelsDirectory
                .appendingPathComponent("parakeet-eou-streaming/320ms")
            try? FileManager.default.removeItem(at: sttDir)
        case .tts:
            let ttsDir = ModelRegistry.ttsModelsDirectory
                .appendingPathComponent("pocket-tts")
            try? FileManager.default.removeItem(at: ttsDir)
        }

        modelStates[type] = .notDownloaded
        error = nil

        switch type {
        case .llm: await downloadLLM()
        case .stt: await downloadSTT()
        case .tts: await downloadTTS()
        }
    }

    // MARK: - LLM Download

    private func downloadLLM() async {
        modelStates[.llm] = .downloading(progress: 0)

        let destination = ModelRegistry.modelsDirectory
            .appendingPathComponent(ModelRegistry.llmFilename)

        guard let url = URL(string: ModelRegistry.llmDownloadURL) else {
            modelStates[.llm] = .error("Invalid URL")
            return
        }

        do {
            try await ModelDownloader.shared.downloadLLM(
                from: url, to: destination,
                expectedSize: Self.llmExpectedSize
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.modelStates[.llm] = .downloading(progress: progress)
                }
            }
            modelStates[.llm] = .downloaded
            logger.info("LLM downloaded successfully")
        } catch {
            modelStates[.llm] = .error(error.localizedDescription)
            self.error = "LLM download failed: \(error.localizedDescription)"
            logger.error("LLM download failed: \(error.localizedDescription)")
        }
    }

    // MARK: - STT Download

    private func downloadSTT() async {
        modelStates[.stt] = .downloading(progress: 0)

        do {
            try await ModelDownloader.shared.downloadSTT(
                to: ModelRegistry.modelsDirectory
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.modelStates[.stt] = .downloading(progress: progress)
                }
            }
            modelStates[.stt] = .downloaded
            logger.info("STT models downloaded successfully")
        } catch {
            modelStates[.stt] = .error(error.localizedDescription)
            self.error = "STT download failed: \(error.localizedDescription)"
            logger.error("STT download failed: \(error.localizedDescription)")
        }
    }

    // MARK: - TTS Download

    private func downloadTTS() async {
        modelStates[.tts] = .downloading(progress: 0)

        do {
            try await ModelDownloader.shared.downloadTTS(
                to: ModelRegistry.ttsModelsDirectory
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.modelStates[.tts] = .downloading(progress: progress)
                }
            }
            modelStates[.tts] = .downloaded
            logger.info("TTS models downloaded successfully")
        } catch {
            modelStates[.tts] = .error(error.localizedDescription)
            self.error = "TTS download failed: \(error.localizedDescription)"
            logger.error("TTS download failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Helpers

    func deleteAllModels() {
        try? FileManager.default.removeItem(at: ModelRegistry.modelsDirectory)
        try? FileManager.default.createDirectory(
            at: ModelRegistry.modelsDirectory, withIntermediateDirectories: true)
        // Also clear legacy PocketTTS cache from FluidAudio if it exists
        if let cachesDir = FileManager.default.urls(
            for: .cachesDirectory, in: .userDomainMask
        ).first {
            let legacyCache = cachesDir.appendingPathComponent("fluidaudio")
            try? FileManager.default.removeItem(at: legacyCache)
        }
        checkExistingModels()
    }
}
