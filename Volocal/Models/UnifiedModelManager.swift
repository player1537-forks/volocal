import Foundation
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "models")

/// Unified model manager tracking download state for all 3 models (LLM, STT, TTS).
/// All downloads use the vendored ModelDownloader for resume support and integrity checks.
@MainActor
final class UnifiedModelManager: ObservableObject {
    @Published var modelStates: [ModelRegistry.ModelType: ModelState] = [:]
    @Published var error: String?

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
        // Fast path: if we already have a cached manifest, resolve model states
        // synchronously so the UI doesn't flash the download screen on a normal
        // launch. We then refresh from the network in the background.
        if let manifest = ModelDownloader.shared.loadCachedManifest() {
            applyCompleteness(manifest)
        }
        Task {
            await self.refreshModelStates()
        }
    }

    // MARK: - Integrity checks (fast path for app launch)

    /// Re-fetches the manifest and re-evaluates which models are complete.
    /// Falls back to the cached manifest when the network is unavailable.
    func refreshModelStates() async {
        do {
            let manifest = try await ModelDownloader.shared.fetchManifest()
            applyCompleteness(manifest)
        } catch {
            if let manifest = ModelDownloader.shared.loadCachedManifest() {
                applyCompleteness(manifest)
            } else {
                for type in ModelRegistry.ModelType.allCases {
                    modelStates[type] = .notDownloaded
                }
            }
        }
    }

    /// Marks each model downloaded (or not) by checking every file listed in the
    /// manifest for presence and size. This replaces the old hardcoded per-model
    /// sentinel checks and adapts automatically if the model layout changes.
    private func applyCompleteness(_ manifest: ModelManifest) {
        for type in ModelRegistry.ModelType.allCases {
            // Don't clobber an in-flight download.
            if case .downloading = modelStates[type] {
                continue
            }
            modelStates[type] =
                ModelDownloader.shared.isComplete(type, manifest: manifest)
                ? .downloaded
                : .notDownloaded
        }
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

        // Also clear any partially-downloaded chunks for this model
        try? FileManager.default.removeItem(at: ModelDownloader.chunksDirectory(for: type))

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

        do {
            try await ModelDownloader.shared.download(.llm) { [weak self] progress in
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
            try await ModelDownloader.shared.download(.stt) { [weak self] progress in
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
            try await ModelDownloader.shared.download(.tts) { [weak self] progress in
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
        for type in ModelRegistry.ModelType.allCases {
            modelStates[type] = .notDownloaded
        }
        error = nil
    }
}
