// A visible, stable macOS application identity for the external Roblox learner.
// No private APIs, TCC database writes, injected game code, or sandbox bypasses.
// Permission prompts occur only when a visible button or explicit request asks.
import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

struct LauncherConfiguration: Decodable {
    let python: String
    let repository: String
    let controlDirectory: String
}

final class Launcher: NSObject, NSApplicationDelegate {
    private var config: LauncherConfiguration!
    private var control: URL!
    private var window: NSWindow!
    private let statusLabel = NSTextField(wrappingLabelWithString: "Starting…")
    private let logView = NSTextView()
    private var timer: Timer?
    private var child: Process?
    private var stdoutHandle: FileHandle?
    private var stderrHandle: FileHandle?
    private var stdoutURL: URL?
    private var stderrURL: URL?
    private var requestID = ""
    private var lastRequestID = ""
    private var phase = "ready"
    private var detail = "No child process running."
    private var lastExitCode: Int32?
    private var quitAfterChild = false
    private let allowedModules: Set<String> = [
        "roblox_learner.play", "roblox_learner.record", "roblox_learner.collect",
        "roblox_learner.train", "roblox_learner.evaluate", "roblox_learner.benchmark"
    ]

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            guard let resource = Bundle.main.url(forResource: "launcher", withExtension: "json") else {
                throw NSError(domain: "Launcher", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing launcher.json"])
            }
            config = try JSONDecoder().decode(LauncherConfiguration.self, from: Data(contentsOf: resource))
            control = URL(fileURLWithPath: config.controlDirectory, isDirectory: true)
            try FileManager.default.createDirectory(at: control, withIntermediateDirectories: true)
            lastRequestID = (try? String(contentsOf: control.appendingPathComponent("agent-last-request.txt"), encoding: .utf8)) ?? ""
            makeWindow()
            writeStatus()
            timer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { [weak self] _ in
                self?.pollRequest()
                self?.writeStatus()
                self?.refreshLog()
            }
        } catch {
            let alert = NSAlert()
            alert.messageText = "Roblox Learner could not start"
            alert.informativeText = error.localizedDescription
            alert.runModal()
            NSApp.terminate(nil)
        }
    }

    private func makeWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 740, height: 490),
            styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "Roblox Learner"
        window.center()
        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 14
        root.edgeInsets = NSEdgeInsets(top: 20, left: 20, bottom: 20, right: 20)
        root.translatesAutoresizingMaskIntoConstraints = false
        window.contentView!.addSubview(root)
        NSLayoutConstraint.activate([
            root.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor),
            root.trailingAnchor.constraint(equalTo: window.contentView!.trailingAnchor),
            root.topAnchor.constraint(equalTo: window.contentView!.topAnchor),
            root.bottomAnchor.constraint(equalTo: window.contentView!.bottomAnchor)
        ])
        let title = NSTextField(labelWithString: "Roblox Learner · external visual agent")
        title.font = NSFont.systemFont(ofSize: 21, weight: .semibold)
        root.addArrangedSubview(title)
        let explanation = NSTextField(wrappingLabelWithString: "This local app owns the agent’s macOS permissions. It captures the Roblox window and uses ordinary keyboard and mouse events. Escape or a focus change stops live controls.")
        explanation.maximumNumberOfLines = 3
        root.addArrangedSubview(explanation)
        let permissions = NSStackView()
        permissions.orientation = .horizontal
        permissions.spacing = 8
        for (title, action) in [
            ("Grant Accessibility", #selector(requestAccessibility)),
            ("Grant Input Monitoring", #selector(requestInputMonitoring)),
            ("Grant Screen Recording", #selector(requestScreenRecording))
        ] {
            permissions.addArrangedSubview(NSButton(title: title, target: self, action: action))
        }
        root.addArrangedSubview(permissions)
        statusLabel.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        statusLabel.maximumNumberOfLines = 5
        root.addArrangedSubview(statusLabel)
        let controls = NSStackView()
        controls.orientation = .horizontal
        controls.spacing = 10
        controls.addArrangedSubview(NSButton(title: "Diagnose Python child", target: self, action: #selector(diagnose)))
        controls.addArrangedSubview(NSButton(title: "Stop agent", target: self, action: #selector(stopAgent)))
        root.addArrangedSubview(controls)
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        logView.isEditable = false
        logView.isSelectable = true
        logView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        logView.autoresizingMask = [.width]
        scroll.documentView = logView
        root.addArrangedSubview(scroll)
        NSLayoutConstraint.activate([
            scroll.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -40),
            scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 170),
            explanation.widthAnchor.constraint(equalTo: scroll.widthAnchor),
            statusLabel.widthAnchor.constraint(equalTo: scroll.widthAnchor)
        ])
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func permissionState() -> [String: Bool] {
        ["screen_recording": CGPreflightScreenCaptureAccess(),
         "accessibility": AXIsProcessTrusted(),
         "post_events": CGPreflightPostEventAccess(),
         "input_monitoring": CGPreflightListenEventAccess()]
    }

    private func writeStatus() {
        guard control != nil else { return }
        let permissions = permissionState()
        let status: [String: Any] = [
            "timestamp": Date().timeIntervalSince1970,
            "bundle_id": Bundle.main.bundleIdentifier ?? "",
            "app_pid": ProcessInfo.processInfo.processIdentifier,
            "permissions": permissions,
            "phase": phase,
            "detail": detail,
            "request_id": requestID,
            "child_pid": child?.isRunning == true ? Int(child!.processIdentifier) : NSNull(),
            "child_exit_code": lastExitCode.map { Int($0) as Any } ?? NSNull(),
            "stdout": stdoutURL?.path ?? "",
            "stderr": stderrURL?.path ?? "",
            "repository": config.repository,
            "python": config.python
        ]
        if let data = try? JSONSerialization.data(withJSONObject: status, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: control.appendingPathComponent("agent-status.json"), options: .atomic)
        }
        let check: (String) -> String = { permissions[$0] == true ? "granted" : "not granted" }
        statusLabel.stringValue = "Screen: \(check("screen_recording"))   Accessibility: \(check("accessibility"))   Input: \(check("input_monitoring"))\n\(phase): \(detail)"
    }

    private func pollRequest() {
        let path = control.appendingPathComponent("agent-request.json")
        guard let data = try? Data(contentsOf: path), data.count <= 65536,
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let identifier = object["id"] as? String,
              identifier.range(of: "^[A-Za-z0-9_-]{1,80}$", options: .regularExpression) != nil,
              identifier != lastRequestID else { return }
        lastRequestID = identifier
        try? identifier.write(to: control.appendingPathComponent("agent-last-request.txt"), atomically: true, encoding: .utf8)
        requestID = identifier
        if let command = object["command"] as? String {
            switch command {
            case "request_accessibility": requestAccessibility()
            case "request_input_monitoring": requestInputMonitoring()
            case "request_screen_recording": requestScreenRecording()
            case "stop": stopAgent()
            case "native_diagnose": writeStatus()
            default: phase = "error"; detail = "Unsupported native command."
            }
            return
        }
        guard let module = object["module"] as? String,
              let arguments = object["args"] as? [String] else {
            phase = "error"; detail = "Request needs module and string args array."
            return
        }
        startChild(module: module, arguments: arguments)
    }

    private func startChild(module: String, arguments: [String]) {
        guard child?.isRunning != true else {
            detail = "An agent is already running; stop it before starting another."
            return
        }
        guard allowedModules.contains(module), arguments.count <= 256, arguments.allSatisfy({ $0.count <= 8192 }) else {
            phase = "error"; detail = "Only bounded arguments to learner CLI modules are allowed."
            return
        }
        do {
            let runDirectory = control.appendingPathComponent("agent-runs", isDirectory: true).appendingPathComponent(requestID, isDirectory: true)
            try FileManager.default.createDirectory(at: runDirectory, withIntermediateDirectories: true)
            stdoutURL = runDirectory.appendingPathComponent("stdout.log")
            stderrURL = runDirectory.appendingPathComponent("stderr.log")
            FileManager.default.createFile(atPath: stdoutURL!.path, contents: Data())
            FileManager.default.createFile(atPath: stderrURL!.path, contents: Data())
            stdoutHandle = try FileHandle(forWritingTo: stdoutURL!)
            stderrHandle = try FileHandle(forWritingTo: stderrURL!)
            let process = Process()
            process.executableURL = URL(fileURLWithPath: config.python)
            process.arguments = ["-u", "-m", module] + arguments
            process.currentDirectoryURL = URL(fileURLWithPath: config.repository, isDirectory: true)
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONPATH"] = URL(fileURLWithPath: config.repository).appendingPathComponent("src").path
            environment["PYTHONUNBUFFERED"] = "1"
            environment["VIRTUAL_ENV"] = URL(fileURLWithPath: config.python).deletingLastPathComponent().deletingLastPathComponent().path
            environment["PATH"] = URL(fileURLWithPath: config.python).deletingLastPathComponent().path + ":/usr/bin:/bin:/usr/sbin:/sbin"
            environment.removeValue(forKey: "PYTHONHOME")
            process.environment = environment
            process.standardOutput = stdoutHandle
            process.standardError = stderrHandle
            process.standardInput = FileHandle.nullDevice
            process.terminationHandler = { [weak self] completed in
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    self.lastExitCode = completed.terminationStatus
                    self.phase = completed.terminationStatus == 0 ? "finished" : "error"
                    self.detail = "Python child exited with code \(completed.terminationStatus)."
                    try? self.stdoutHandle?.close()
                    try? self.stderrHandle?.close()
                    self.child = nil
                    self.writeStatus()
                    self.refreshLog()
                    if self.quitAfterChild { NSApp.reply(toApplicationShouldTerminate: true) }
                }
            }
            child = process
            lastExitCode = nil
            try process.run()
            phase = "running"
            detail = "\(module) · PID \(process.processIdentifier). Escape stops live controls."
        } catch {
            child = nil
            phase = "error"
            detail = error.localizedDescription
        }
        writeStatus()
    }

    private func refreshLog() {
        func tail(_ url: URL?) -> String {
            guard let url = url, let handle = try? FileHandle(forReadingFrom: url) else { return "" }
            defer { try? handle.close() }
            let size = (try? handle.seekToEnd()) ?? 0
            try? handle.seek(toOffset: size > 12000 ? size - 12000 : 0)
            return String(decoding: (try? handle.readToEnd()) ?? Data(), as: UTF8.self)
        }
        let output = tail(stdoutURL)
        let errors = tail(stderrURL)
        let value = output + (errors.isEmpty ? "" : "\nSTDERR\n" + errors)
        if logView.string != value {
            logView.string = value
            logView.scrollToEndOfDocument(nil)
        }
    }

    @objc private func requestAccessibility() {
        _ = AXIsProcessTrustedWithOptions([kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary)
        detail = "Accessibility permission requested for Roblox Learner."
        writeStatus()
    }

    @objc private func requestInputMonitoring() {
        _ = CGRequestListenEventAccess()
        detail = "Input Monitoring permission requested for Roblox Learner."
        writeStatus()
    }

    @objc private func requestScreenRecording() {
        _ = CGRequestScreenCaptureAccess()
        detail = "Screen Recording permission requested. macOS may require reopening this app."
        writeStatus()
    }

    @objc private func diagnose() {
        requestID = "diagnose-" + UUID().uuidString
        startChild(module: "roblox_learner.play", arguments: ["--diagnose"])
    }

    @objc private func stopAgent() {
        if let child = child, child.isRunning {
            child.interrupt() // SIGINT lets Python finally blocks release controls.
            phase = "stopping"
            detail = "Stop requested; waiting for Python to release controls and exit."
        } else {
            phase = "ready"
            detail = "No child process running."
        }
        writeStatus()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if child?.isRunning == true {
            quitAfterChild = true
            stopAgent()
            return .terminateLater
        }
        return .terminateNow
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag { window.makeKeyAndOrderFront(nil) }
        return true
    }
}

let application = NSApplication.shared
let delegate = Launcher()
application.setActivationPolicy(.regular)
application.delegate = delegate
application.run()
