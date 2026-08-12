import CryptoKit
import Foundation
import OSLog

private let logger = Logger(subsystem: "com.volocal.app", category: "downloader")

// MARK: - HuggingFace URL Helpers

private enum HF {
    static let baseURL = "https://huggingface.co"

    static func resolveFile(repo: String, filePath: String) throws -> URL {
        let urlString = "\(baseURL)/\(repo)/resolve/main/\(filePath)"
        guard let url = URL(string: urlString) else {
            throw DownloadError.invalidURL(urlString)
        }
        return url
    }
}

// MARK: - Errors

enum DownloadError: LocalizedError {
    case invalidURL(String)
    case httpError(statusCode: Int, path: String)
    case fileNotFound(path: String)
    case sizeMismatch(path: String, expected: Int64, got: Int64)
    case checksumMismatch(path: String)

    var errorDescription: String? {
        switch self {
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        case .httpError(let code, let path):
            return "Server error (\(code)) for \(path)"
        case .fileNotFound(let path):
            return "File not found after download: \(path)"
        case .sizeMismatch(let path, let expected, let got):
            return "Size mismatch for \(path): expected \(expected), got \(got)"
        case .checksumMismatch(let path):
            return "Checksum verification failed for \(path)"
        }
    }
}

// MARK: - Manifest

/// The on-disk manifest (`manifest.json`) published in the chunked model repo.
/// Maps each final model file to an ordered list of independently-downloadable chunks.
struct ModelManifest: Codable {
    let version: Int
    let chunkSize: Int64
    let models: [String: [File]]

    struct File: Codable {
        /// Path of the reassembled file, relative to the model's local root.
        let path: String
        let size: Int64
        let sha256: String
        let chunks: [Chunk]
    }

    struct Chunk: Codable {
        /// Repo-relative path of the chunk (e.g. "llm/Qwen…gguf.part000").
        let name: String
        let size: Int64
        let sha256: String
    }
}

// MARK: - SHA256 (streaming, via CryptoKit)

private enum SHA256 {
    static func hex(_ data: Data) -> String {
        CryptoKit.SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Stream the file so we never load a multi-hundred-MB file into memory.
    static func hex(fileURL url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = CryptoKit.SHA256()
        while let chunk = try? handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Chunk Downloader

/// Downloads a single chunk directly to disk, streaming via `URLSessionDataDelegate`.
///
/// Unlike `URLSessionDownloadTask` (whose partial bytes live in an opaque temp file),
/// partial progress here always lives at `destinationURL`. Resume is therefore a plain
/// `Range` header — deterministic and independent of `NSURLSessionDownloadTaskResumeData`.
private final class ChunkDownloader: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let remoteURL: URL
    private let destinationURL: URL
    private let expectedSize: Int64
    private let expectedSHA256: String

    private var fileHandle: FileHandle?
    private var received: Int64 = 0
    private var requestedRange = false
    private var continuation: CheckedContinuation<Void, Error>?
    private var progressHandler: ((Double) -> Void)?
    private var session: URLSession?
    private var finished = false

    init(remoteURL: URL, destinationURL: URL, expectedSize: Int64, expectedSHA256: String) {
        self.remoteURL = remoteURL
        self.destinationURL = destinationURL
        self.expectedSize = expectedSize
        self.expectedSHA256 = expectedSHA256
        super.init()
    }

    func download(progress: @escaping (Double) -> Void) async throws {
        self.progressHandler = progress

        // Already complete and verified.
        if existingSize() == expectedSize, SHA256.hex(fileURL: destinationURL) == expectedSHA256 {
            progress(1.0)
            return
        }

        return try await withCheckedThrowingContinuation { cont in
            self.continuation = cont

            try? FileManager.default.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)

            var offset = existingSize()
            if offset >= expectedSize {
                // Equal-or-larger but failed verification: start over.
                try? FileManager.default.removeItem(at: destinationURL)
                offset = 0
            }
            if !FileManager.default.fileExists(atPath: destinationURL.path) {
                FileManager.default.createFile(atPath: destinationURL.path, contents: nil)
            }

            do {
                let handle = try FileHandle(forWritingTo: destinationURL)
                try handle.seekToEnd()
                self.fileHandle = handle
            } catch {
                finish(with: error)
                return
            }
            self.received = offset

            let config = URLSessionConfiguration.default
            config.timeoutIntervalForResource = 3600
            let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
            self.session = session

            var request = URLRequest(url: remoteURL, timeoutInterval: 3600)
            if offset > 0 && offset < expectedSize {
                request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
                self.requestedRange = true
            }
            session.dataTask(with: request).resume()
        }
    }

    // MARK: - URLSessionDataDelegate

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        if let http = response as? HTTPURLResponse {
            // 206 = server honored Range (append). 200 = ignored Range. 416 = stale range.
            let restart: Bool
            switch http.statusCode {
            case 200:
                restart = requestedRange
            case 416:
                restart = true
            default:
                restart = false
            }
            if restart {
                received = 0
                if let handle = fileHandle {
                    try? handle.truncate(atOffset: 0)
                    try? handle.seek(toOffset: 0)
                }
            }
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let handle = fileHandle else { return }
        do {
            try handle.write(contentsOf: data)
            received += Int64(data.count)
            progressHandler?(Double(received) / Double(expectedSize))
        } catch {
            finish(with: error)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?)
    {
        try? fileHandle?.close()
        fileHandle = nil

        if let error {
            finish(with: error)
        } else if received == expectedSize, SHA256.hex(fileURL: destinationURL) == expectedSHA256 {
            finish(with: nil)
        } else {
            finish(
                with: DownloadError.sizeMismatch(
                    path: destinationURL.lastPathComponent, expected: expectedSize, got: received))
        }
    }

    // MARK: - Helpers

    private func existingSize() -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: destinationURL.path),
            let size = attrs[.size] as? Int64
        else { return 0 }
        return size
    }

    private func finish(with error: Error?) {
        guard !finished else { return }
        finished = true
        session?.invalidateAndCancel()
        session = nil
        fileHandle = nil
        if let error {
            continuation?.resume(throwing: error)
        } else {
            continuation?.resume()
        }
        continuation = nil
    }
}

// MARK: - Model Downloader

/// Downloads all three models from the consolidated, chunked HuggingFace repo.
/// Each model file is fetched as independently-resumable chunks, reassembled, and
/// verified against `manifest.json` (per-chunk and whole-file SHA256).
final class ModelDownloader: @unchecked Sendable {
    static let shared = ModelDownloader()

    /// Bound concurrent chunk downloads to keep memory/connection usage in check.
    private let maxConcurrentChunks = 3

    private init() {}

    // MARK: - Public

    func download(
        _ type: ModelRegistry.ModelType, onProgress: @escaping (Double) -> Void
    ) async throws {
        let manifest = try await fetchManifest()
        guard let files = manifest.models[type.rawValue], !files.isEmpty else {
            throw DownloadError.fileNotFound(path: type.rawValue)
        }

        let baseDir = Self.baseDirectory(for: type)
        let chunksRoot = baseDir.appendingPathComponent(".chunks", isDirectory: true)

        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }
        var doneBytes: Int64 = 0

        for file in files {
            let dest = baseDir.appendingPathComponent(file.path)
            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)

            // Already present and verified? Skip.
            if fileSize(dest) == file.size, SHA256.hex(fileURL: dest) == file.sha256 {
                doneBytes += file.size
                onProgress(totalBytes > 0 ? Double(doneBytes) / Double(totalBytes) : 1.0)
                continue
            }

            try await downloadChunks(file.chunks, into: chunksRoot)
            try reassemble(file.chunks, from: chunksRoot, to: dest)

            guard SHA256.hex(fileURL: dest) == file.sha256 else {
                throw DownloadError.checksumMismatch(path: file.path)
            }

            doneBytes += file.size
            onProgress(totalBytes > 0 ? Double(doneBytes) / Double(totalBytes) : 1.0)
        }

        // Free the chunk files now that every file in this model is reassembled + verified.
        try? FileManager.default.removeItem(at: Self.chunksDirectory(for: type))
    }

    /// Directory where a model's chunk files are stored (used for retry cleanup).
    static func chunksDirectory(for type: ModelRegistry.ModelType) -> URL {
        baseDirectory(for: type)
            .appendingPathComponent(".chunks", isDirectory: true)
            .appendingPathComponent(type.rawValue, isDirectory: true)
    }

    // MARK: - Manifest

    private func fetchManifest() async throws -> ModelManifest {
        guard let url = URL(string: ModelRegistry.manifestURL) else {
            throw DownloadError.invalidURL(ModelRegistry.manifestURL)
        }
        var request = URLRequest(url: url, timeoutInterval: 30)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw DownloadError.httpError(
                statusCode: http.statusCode, path: ModelRegistry.manifestFilename)
        }
        return try JSONDecoder().decode(ModelManifest.self, from: data)
    }

    // MARK: - Chunks

    private func downloadChunks(_ chunks: [ModelManifest.Chunk], into root: URL) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            var iterator = chunks.makeIterator()

            func spawn() {
                guard let chunk = iterator.next() else { return }
                group.addTask {
                    try await self.downloadChunk(chunk, into: root)
                }
            }

            for _ in 0..<min(maxConcurrentChunks, chunks.count) {
                spawn()
            }
            while try await group.next() != nil {
                spawn()
            }
        }
    }

    private func downloadChunk(_ chunk: ModelManifest.Chunk, into root: URL) async throws {
        let remote = try HF.resolveFile(repo: ModelRegistry.modelRepo, filePath: chunk.name)
        let local = root.appendingPathComponent(chunk.name)
        try FileManager.default.createDirectory(
            at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
        let downloader = ChunkDownloader(
            remoteURL: remote,
            destinationURL: local,
            expectedSize: chunk.size,
            expectedSHA256: chunk.sha256)
        try await downloader.download { _ in }
    }

    private func reassemble(_ chunks: [ModelManifest.Chunk], from root: URL, to dest: URL) throws {
        try? FileManager.default.removeItem(at: dest)
        FileManager.default.createFile(atPath: dest.path, contents: nil)
        let output = try FileHandle(forWritingTo: dest)
        defer { try? output.close() }

        for chunk in chunks {
            let part = root.appendingPathComponent(chunk.name)
            let input = try FileHandle(forReadingFrom: part)
            while let data = try input.read(upToCount: 1 << 20), !data.isEmpty {
                try output.write(contentsOf: data)
            }
            try? input.close()
        }
    }

    // MARK: - Helpers

    private func fileSize(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64)
    }

    private static func baseDirectory(for type: ModelRegistry.ModelType) -> URL {
        switch type {
        case .llm, .stt:
            return ModelRegistry.modelsDirectory
        case .tts:
            return ModelRegistry.ttsModelsDirectory
        }
    }
}
