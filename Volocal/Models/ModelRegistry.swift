import Foundation

/// Central registry of all models used by Volocal.
/// Defines metadata, download sources, and local paths.
enum ModelRegistry {
    /// All model types used in the app
    enum ModelType: String, CaseIterable, Identifiable {
        case llm
        case stt
        case tts

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .llm: return "Language Model"
            case .stt: return "Speech Recognition"
            case .tts: return "Text-to-Speech"
            }
        }

        var icon: String {
            switch self {
            case .llm: return "brain"
            case .stt: return "mic.fill"
            case .tts: return "speaker.wave.3.fill"
            }
        }

        var sizeDescription: String {
            switch self {
            case .llm: return "~1.2 GB"
            case .stt: return "~450 MB"
            case .tts: return "~600 MB"
            }
        }

        var detail: String {
            switch self {
            case .llm: return "Qwen3.5-2B Q4_K_S"
            case .stt: return "Parakeet EOU 320"
            case .tts: return "PocketTTS"
            }
        }
    }

    // MARK: - LLM

    static let llmFilename = "Qwen3.5-2B-Q4_K_S.gguf"

    /// Expected size (bytes) of the Unsloth Q4_K_S GGUF. Used for a cheap
    /// launch-time completeness check before full SHA256 verification.
    static let llmExpectedSize: Int64 = 1_217_757_440

    // MARK: - Chunked model repo (single repo containing all 3 models)

    static let modelRepo = "player1537/volocal-models"
    static let manifestFilename = "manifest.json"

    static var manifestURL: String {
        "https://huggingface.co/\(modelRepo)/resolve/main/\(manifestFilename)"
    }

    // MARK: - Paths

    static var modelsDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static var llmModelPath: String? {
        let path = modelsDirectory.appendingPathComponent(llmFilename).path
        guard FileManager.default.fileExists(atPath: path),
            let attrs = try? FileManager.default.attributesOfItem(atPath: path),
            let size = attrs[.size] as? UInt64,
            size > 1024
        else { return nil }
        return path
    }

    /// Directory where PocketTTS models are stored.
    /// Must match the path FluidAudio's PocketTtsManager expects at runtime:
    /// ~/Library/Caches/fluidaudio/Models/
    static var ttsModelsDirectory: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let dir =
            caches
            .appendingPathComponent("fluidaudio")
            .appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Directory where Parakeet EOU (STT) models are stored.
    /// Same as general modelsDirectory.
    static var sttModelsDirectory: URL {
        modelsDirectory
    }
}
