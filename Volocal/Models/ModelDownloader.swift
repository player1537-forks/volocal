import CryptoKit
import Foundation
import OSLog

private let logger = Logger(subsystem: "com.volocal.app", category: "downloader")

// MARK: - HuggingFace URL Helpers

private enum HF {
    static let baseURL = "https://huggingface.co"

    static func apiListFiles(repo: String, subpath: String = "") throws -> URL {
        let apiPath = subpath.isEmpty ? "tree/main" : "tree/main/\(subpath)"
        let urlString = "\(baseURL)/api/models/\(repo)/\(apiPath)"
        guard let url = URL(string: urlString) else {
            throw DownloadError.invalidURL(urlString)
        }
        return url
    }

    static func resolveFile(repo: String, filePath: String) throws -> URL {
        let urlString = "\(baseURL)/\(repo)/resolve/main/\(filePath)"
        guard let url = URL(string: urlString) else {
            throw DownloadError.invalidURL(urlString)
        }
        return url
    }
}

// MARK: - Model Definitions (vendored from FluidAudio)

/// Minimal repo definition for the models Volocal needs.
enum ModelRepo {
    case parakeetEou320
    case pocketTts

    /// HuggingFace repo path (owner/name)
    var remotePath: String {
        switch self {
        case .parakeetEou320: return "FluidInference/parakeet-realtime-eou-120m-coreml"
        case .pocketTts: return "FluidInference/pocket-tts-coreml"
        }
    }

    /// Subdirectory within the repo (nil if using whole repo)
    var subPath: String? {
        switch self {
        case .parakeetEou320: return "320ms"
        case .pocketTts: return nil
        }
    }

    /// Local folder name
    var folderName: String {
        switch self {
        case .parakeetEou320: return "parakeet-eou-streaming/320ms"
        case .pocketTts: return "pocket-tts"
        }
    }

    /// Required top-level model directories/files that must exist after download.
    var requiredModels: Set<String> {
        switch self {
        case .parakeetEou320:
            return [
                "streaming_encoder.mlmodelc", "decoder.mlmodelc", "joint_decision.mlmodelc",
                "vocab.json",
            ]
        case .pocketTts:
            return [
                "cond_step.mlmodelc", "flowlm_step.mlmodelc", "flow_decoder.mlmodelc",
                "mimi_decoder_v2.mlmodelc", "constants_bin",
            ]
        }
    }
}

// MARK: - Error Types

enum DownloadError: LocalizedError {
    case invalidURL(String)
    case httpError(statusCode: Int, path: String)
    case fileNotFound(path: String)
    case sizeMismatch(path: String, expected: Int64, got: Int64)
    case checksumMismatch(path: String)
    case cancelled

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
        case .cancelled:
            return "Download cancelled"
        }
    }
}

// MARK: - Resumable Single-File Downloader

/// Downloads a single file with resume support and integrity verification.
private final class ResumableFileDownloader: NSObject, URLSessionDownloadDelegate,
    @unchecked Sendable
{
    private let remoteURL: URL
    private let destinationURL: URL
    private let expectedSize: Int64?
    private let checksumsFile: URL
    private let resumeDataFile: URL
    private var session: URLSession?
    private var downloadTask: URLSessionDownloadTask?

    // Continuation state
    private var continuation: CheckedContinuation<Void, any Error>?
    private var progressHandler: ((Double) -> Void)?
    private var hasCompleted = false

    init(
        remoteURL: URL,
        destinationURL: URL,
        expectedSize: Int64?,
        checksumsDir: URL
    ) {
        self.remoteURL = remoteURL
        self.destinationURL = destinationURL
        self.expectedSize = expectedSize
        self.checksumsFile = checksumsDir.appendingPathComponent(".checksums.json")
        self.resumeDataFile = checksumsDir.appendingPathComponent(
            ".resume-\(remoteURL.hashValue).data")
        super.init()
    }

    /// Download the file, resuming if possible.
    func download(progress: @escaping (Double) -> Void) async throws {
        self.progressHandler = progress

        return try await withCheckedThrowingContinuation { cont in
            self.continuation = cont

            let config = URLSessionConfiguration.default
            config.timeoutIntervalForResource = 3600
            self.session = URLSession(configuration: config, delegate: self, delegateQueue: nil)

            // Try to resume if we have saved resume data
            if let resumeData = try? Data(contentsOf: resumeDataFile), !resumeData.isEmpty {
                logger.info(
                    "Resuming download from saved state (\(resumeData.count) bytes resume data)")
                self.downloadTask = self.session?.downloadTask(withResumeData: resumeData)
            } else {
                var request = URLRequest(url: remoteURL, timeoutInterval: 3600)
                // Request resume support from server
                if FileManager.default.fileExists(atPath: destinationURL.path) {
                    let existingSize =
                        (try? FileManager.default.attributesOfItem(atPath: destinationURL.path)[
                            .size] as? Int64) ?? 0
                    if existingSize > 0 {
                        request.setValue("bytes=\(existingSize)-", forHTTPHeaderField: "Range")
                        logger.info("Requesting range restart from byte \(existingSize)")
                    }
                }
                self.downloadTask = self.session?.downloadTask(with: request)
            }

            self.downloadTask?.resume()
        }
    }

    func cancel() {
        downloadTask?.cancel { resumeData in
            if let data = resumeData {
                try? data.write(to: self.resumeDataFile, options: .atomic)
                logger.info("Saved resume data (\(data.count) bytes)")
            }
        }
        finish(with: DownloadError.cancelled)
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let total =
            totalBytesExpectedToWrite > 0
            ? totalBytesExpectedToWrite : (expectedSize ?? totalBytesWritten)
        let fraction = total > 0 ? Double(totalBytesWritten) / Double(total) : 0
        progressHandler?(min(fraction, 1.0))
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let tempCopy = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.moveItem(at: location, to: tempCopy)
            try FileManager.default.moveItem(at: tempCopy, to: destinationURL)

            // Clear resume data on success
            try? FileManager.default.removeItem(at: resumeDataFile)

            // Verify size
            let actualSize =
                (try? FileManager.default.attributesOfItem(atPath: destinationURL.path)[.size]
                    as? Int64) ?? 0
            if let expected = expectedSize, actualSize != expected {
                finish(
                    with: DownloadError.sizeMismatch(
                        path: destinationURL.lastPathComponent,
                        expected: expected, got: actualSize))
                return
            }

            // Compute and store SHA256
            computeAndStoreChecksum()
            finish(with: nil)
        } catch {
            finish(with: error)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?)
    {
        guard let error = error else { return }

        // Save resume data for later
        let userInfo = (error as NSError).userInfo
        if let resumeData = userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
            try? resumeData.write(to: resumeDataFile, options: .atomic)
            logger.info("Saved resume data after failure (\(resumeData.count) bytes)")
        }

        finish(with: error)
    }

    // MARK: - Checksums

    private func computeAndStoreChecksum() {
        let destURL = self.destinationURL
        let checksumsURL = self.checksumsFile
        guard let data = try? Data(contentsOf: destURL) else { return }
        let sha256 = SHA256.hex(data)

        var checksums: [String: String] = [:]
        if let existing = try? Data(contentsOf: checksumsURL),
            let decoded = try? JSONDecoder().decode([String: String].self, from: existing)
        {
            checksums = decoded
        }
        checksums[destURL.lastPathComponent] = sha256

        if let encoded = try? JSONEncoder().encode(checksums) {
            try? encoded.write(to: checksumsURL, options: .atomic)
        }
        logger.debug("SHA256 for \(destURL.lastPathComponent): \(sha256.prefix(16))...")
    }

    /// Verify the file against a stored checksum. Returns true if verified or no stored checksum exists.
    func verifyChecksum() -> Bool {
        guard let data = try? Data(contentsOf: checksumsFile),
            let checksums = try? JSONDecoder().decode([String: String].self, from: data),
            let expected = checksums[destinationURL.lastPathComponent],
            let fileData = try? Data(contentsOf: destinationURL)
        else {
            // No stored checksum = first download, trust it
            return true
        }
        let actual = SHA256.hex(fileData)
        return actual == expected
    }

    private func finish(with error: Error?) {
        guard !hasCompleted else { return }
        hasCompleted = true
        session?.finishTasksAndInvalidate()
        session = nil
        downloadTask = nil
        if let error {
            continuation?.resume(throwing: error)
        } else {
            continuation?.resume()
        }
        continuation = nil
    }
}

// MARK: - SHA256 (via CryptoKit)

private enum SHA256 {
    static func hex(_ data: Data) -> String {
        CryptoKit.SHA256.hash(data: data).compactMap { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - HuggingFace Repo Downloader

/// Downloads an entire model repository from HuggingFace with resume, progress, and integrity checks.
/// Replaces FluidAudio's DownloadUtils + PocketTtsResourceDownloader.
final class HuggingFaceDownloader {
    /// Progress information for UI reporting.
    struct Progress {
        let fractionCompleted: Double
        let filesDownloaded: Int
        let totalFiles: Int
        let currentFile: String
    }

    /// Download a complete repo, returning when all required files are present and verified.
    /// - Parameters:
    ///   - repo: The model repo to download
    ///   - directory: Base directory where the model folder will be created
    ///   - onProgress: Called with progress updates (on background queue)
    func download(
        repo: ModelRepo, to directory: URL,
        onProgress: @escaping (Progress) -> Void
    ) async throws {
        let repoDir = directory.appendingPathComponent(repo.folderName)
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)

        // Quick path: all files present and checksums verify
        if verifyExistingFiles(repo: repo, repoDir: repoDir) {
            onProgress(
                Progress(fractionCompleted: 1.0, filesDownloaded: 0, totalFiles: 0, currentFile: "")
            )
            logger.info("All \(repo.folderName) files verified locally")
            return
        }

        // List files from HuggingFace API
        onProgress(
            Progress(
                fractionCompleted: 0, filesDownloaded: 0, totalFiles: 0, currentFile: "listing..."))
        let files = try await listFiles(repo: repo)
        logger.info("Found \(files.count) files to download for \(repo.folderName)")

        let totalBytes = files.reduce(Int64(0)) { $0 + Int64(max(0, $1.size)) }
        var downloadedBytes: Int64 = 0

        // Download each file with resume support
        for (index, file) in files.enumerated() {
            let destPath = repoDir.appendingPathComponent(file.path)
            let downloadURL = try HF.resolveFile(repo: repo.remotePath, filePath: file.rawPath)

            if FileManager.default.fileExists(atPath: destPath.path) {
                // File exists — verify integrity
                let existingSize =
                    (try? FileManager.default.attributesOfItem(atPath: destPath.path)[.size]
                        as? Int64) ?? 0
                if existingSize == file.size || file.size <= 0 || existingSize > 1024 {
                    // Create a lightweight verifier just for checksum check
                    let checksumOK = verifyFileChecksum(at: destPath, in: repoDir)
                    if checksumOK {
                        downloadedBytes += Int64(max(0, file.size))
                        onProgress(
                            Progress(
                                fractionCompleted: totalBytes > 0
                                    ? Double(downloadedBytes) / Double(totalBytes)
                                    : Double(index + 1) / Double(files.count),
                                filesDownloaded: index + 1, totalFiles: files.count,
                                currentFile: file.path
                            ))
                        continue
                    }
                    // Checksum mismatch — delete and re-download
                    try? FileManager.default.removeItem(at: destPath)
                }
            }

            // Download
            try FileManager.default.createDirectory(
                at: destPath.deletingLastPathComponent(),
                withIntermediateDirectories: true)

            let downloader = ResumableFileDownloader(
                remoteURL: downloadURL,
                destinationURL: destPath,
                expectedSize: file.size > 0 ? Int64(file.size) : nil,
                checksumsDir: repoDir
            )

            let fileIndex = index
            let fileCount = files.count
            let totalBytesSnapshot = totalBytes
            let baseDownloaded = downloadedBytes

            try await downloader.download { fileProgress in
                let currentBytes = baseDownloaded + Int64(fileProgress * Double(max(0, file.size)))
                let fraction =
                    totalBytesSnapshot > 0
                    ? Double(currentBytes) / Double(totalBytesSnapshot)
                    : Double(fileIndex) / Double(fileCount)
                onProgress(
                    Progress(
                        fractionCompleted: min(fraction, 1.0),
                        filesDownloaded: fileIndex, totalFiles: fileCount, currentFile: file.path
                    ))
            }

            downloadedBytes += Int64(max(0, file.size))
            logger.info("Downloaded \(index + 1)/\(files.count): \(file.path)")
        }

        // Final verification
        guard verifyRequiredModels(repo: repo, repoDir: repoDir) else {
            throw DownloadError.fileNotFound(path: repo.folderName)
        }

        onProgress(
            Progress(
                fractionCompleted: 1.0, filesDownloaded: files.count, totalFiles: files.count,
                currentFile: ""))
        logger.info("Successfully downloaded \(repo.folderName)")
    }

    // MARK: - File listing

    private func listFiles(repo: ModelRepo) async throws -> [(
        path: String, rawPath: String, size: Int
    )] {
        var allFiles: [(path: String, rawPath: String, size: Int)] = []
        try await listDirectory(
            repo: repo, path: repo.subPath ?? "", rawPrefix: repo.subPath ?? "", into: &allFiles)
        return allFiles
    }

    private func listDirectory(
        repo: ModelRepo, path: String, rawPrefix: String,
        into files: inout [(path: String, rawPath: String, size: Int)]
    ) async throws {
        let url = try HF.apiListFiles(repo: repo.remotePath, subpath: path)
        let request = URLRequest(url: url, timeoutInterval: 30)
        let (data, response) = try await URLSession.shared.data(for: request)

        if let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 429 || httpResponse.statusCode == 503
        {
            throw DownloadError.httpError(statusCode: httpResponse.statusCode, path: path)
        }

        guard let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return
        }

        for item in items {
            guard let itemPath = item["path"] as? String,
                let itemType = item["type"] as? String
            else { continue }

            if itemType == "directory" {
                // Only recurse into directories that could contain model files
                let shouldRecurse: Bool = {
                    if let subPath = repo.subPath {
                        return itemPath.hasPrefix("\(subPath)/") || itemPath == subPath
                    }
                    return true
                }()
                if shouldRecurse {
                    try await listDirectory(
                        repo: repo, path: itemPath, rawPrefix: rawPrefix, into: &files)
                }
            } else if itemType == "file" {
                let shouldInclude: Bool = {
                    if let subPath = repo.subPath {
                        return itemPath.hasPrefix("\(subPath)/")
                    }
                    // Include model files + metadata
                    return itemPath.hasSuffix(".mlmodelc") || itemPath.hasSuffix(".bin")
                        || itemPath.hasSuffix(".json") || itemPath.hasSuffix(".model")
                        || itemPath.hasSuffix(".mil") || itemPath.hasSuffix(".txt")
                        || itemPath.hasPrefix("constants_bin/")
                }()
                if shouldInclude {
                    let localPath: String = {
                        if let subPath = repo.subPath, itemPath.hasPrefix("\(subPath)/") {
                            return String(itemPath.dropFirst(subPath.count + 1))
                        }
                        return itemPath
                    }()
                    let fileSize = item["size"] as? Int ?? -1
                    files.append((path: localPath, rawPath: itemPath, size: fileSize))
                }
            }
        }
    }

    // MARK: - Verification

    func verifyExistingFiles(repo: ModelRepo, repoDir: URL) -> Bool {
        guard verifyRequiredModels(repo: repo, repoDir: repoDir) else { return false }

        // Spot-check: verify checksum of at least one file
        let checksumsFile = repoDir.appendingPathComponent(".checksums.json")
        guard FileManager.default.fileExists(atPath: checksumsFile.path) else { return false }

        // Pick the first required model file and verify its checksum
        for modelName in repo.requiredModels {
            let modelPath = repoDir.appendingPathComponent(modelName)
            if modelName.hasSuffix(".mlmodelc") || modelName.hasSuffix(".json") {
                if FileManager.default.fileExists(atPath: modelPath.path) {
                    return verifyFileChecksum(at: modelPath, in: repoDir)
                }
            } else {
                // Directory — check for coremldata.bin inside
                let binPath = modelPath.appendingPathComponent("coremldata.bin")
                if FileManager.default.fileExists(atPath: binPath.path) {
                    return verifyFileChecksum(at: binPath, in: repoDir)
                }
            }
        }
        return false
    }

    private func verifyRequiredModels(repo: ModelRepo, repoDir: URL) -> Bool {
        for model in repo.requiredModels {
            let modelPath = repoDir.appendingPathComponent(model)
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: modelPath.path, isDirectory: &isDir) {
                if isDir.boolValue {
                    // Must contain coremldata.bin
                    let binPath = modelPath.appendingPathComponent("coremldata.bin")
                    guard FileManager.default.fileExists(atPath: binPath.path),
                        let attrs = try? FileManager.default.attributesOfItem(atPath: binPath.path),
                        let size = attrs[.size] as? Int64, size > 1024
                    else {
                        logger.warning("Missing or empty coremldata.bin in \(model)")
                        return false
                    }
                } else {
                    // Must have non-trivial size
                    guard
                        let attrs = try? FileManager.default.attributesOfItem(
                            atPath: modelPath.path),
                        let size = attrs[.size] as? Int64, size > 4
                    else {
                        logger.warning("Missing or empty file: \(model)")
                        return false
                    }
                }
            } else {
                return false
            }
        }
        return true
    }

    private func verifyFileChecksum(at path: URL, in repoDir: URL) -> Bool {
        let checksumsFile = repoDir.appendingPathComponent(".checksums.json")
        let existingChecksums: [String: String]
        if let data = try? Data(contentsOf: checksumsFile),
            let decoded = try? JSONDecoder().decode([String: String].self, from: data)
        {
            existingChecksums = decoded
        } else {
            existingChecksums = [:]
        }

        if let expected = existingChecksums[path.lastPathComponent] {
            // Have stored checksum — verify file matches
            guard let fileData = try? Data(contentsOf: path) else { return false }
            return SHA256.hex(fileData) == expected
        } else {
            // No checksum stored yet — compute and store
            if let fileData = try? Data(contentsOf: path) {
                let sha = SHA256.hex(fileData)
                var c = existingChecksums
                c[path.lastPathComponent] = sha
                if let encoded = try? JSONEncoder().encode(c) {
                    try? encoded.write(to: checksumsFile, options: .atomic)
                }
                return true
            }
            return false
        }
    }
}

// MARK: - Public API: ModelDownloader

/// High-level downloader that replaces FluidAudio downloads for STT/TTS
/// and replaces the custom LLM download in UnifiedModelManager.
/// All three models benefit from resume, progress, and SHA256 integrity.
@MainActor
final class ModelDownloader {
    /// Shared instance
    static let shared = ModelDownloader()

    private let hf = HuggingFaceDownloader()

    private init() {}

    // MARK: - STT (Parakeet EOU 320)

    func downloadSTT(
        to directory: URL,
        onProgress: @escaping (Double) -> Void
    ) async throws {
        try await hf.download(repo: .parakeetEou320, to: directory) { hfProgress in
            onProgress(hfProgress.fractionCompleted)
        }
    }

    // MARK: - TTS (PocketTTS)

    func downloadTTS(
        to directory: URL,
        onProgress: @escaping (Double) -> Void
    ) async throws {
        try await hf.download(repo: .pocketTts, to: directory) { hfProgress in
            onProgress(hfProgress.fractionCompleted)
        }
    }

    // MARK: - LLM (single GGUF file with resume)

    /// Known expected size for Q4_K_S variant.
    private static let llmExpectedBytes: Int64 = 1_261_854_880
    private static let llmChecksumFile = "Qwen_Qwen3.5-2B-Q4_K_S.gguf"

    func downloadLLM(
        from url: URL, to destination: URL,
        expectedSize: Int64 = ModelDownloader.llmExpectedBytes,
        onProgress: @escaping (Double) -> Void
    ) async throws {
        let checksumsDir = destination.deletingLastPathComponent()

        let downloader = ResumableFileDownloader(
            remoteURL: url,
            destinationURL: destination,
            expectedSize: expectedSize,
            checksumsDir: checksumsDir
        )

        try await downloader.download(progress: onProgress)
    }

    // MARK: - Integrity helpers

    /// Verify that the LLM file passes checksum verification.
    func verifyLLM(at path: URL) -> Bool {
        let checksumsDir = path.deletingLastPathComponent()
        let checksumsFile = checksumsDir.appendingPathComponent(".checksums.json")
        guard let data = try? Data(contentsOf: checksumsFile),
            let checksums = try? JSONDecoder().decode([String: String].self, from: data),
            let expected = checksums[Self.llmChecksumFile],
            let fileData = try? Data(contentsOf: path)
        else { return false }
        return SHA256.hex(fileData) == expected
    }

    /// Verify STT models pass integrity check.
    func verifySTT(in directory: URL) -> Bool {
        let repoDir = directory.appendingPathComponent(ModelRepo.parakeetEou320.folderName)
        return HuggingFaceDownloader().verifyExistingFiles(repo: .parakeetEou320, repoDir: repoDir)
    }

    /// Verify TTS models pass integrity check.
    func verifyTTS(in directory: URL) -> Bool {
        let repoDir = directory.appendingPathComponent(ModelRepo.pocketTts.folderName)
        return HuggingFaceDownloader().verifyExistingFiles(repo: .pocketTts, repoDir: repoDir)
    }
}
