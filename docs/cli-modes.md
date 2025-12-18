# Tinman CLI Modes

This document shows the implicit nodes that Tinman creates in different CLI modes.

## `tinman run`

### Basic Run (no logging)

```bash
tinman run myapp:producer,consumer --no-capture-logs
```

```mermaid
graph LR
    subgraph "User Nodes"
        producer[producer]
        consumer[consumer]
    end
    
    producer -->|channel| consumer
```

### Run with Log Capture (default)

```bash
tinman run myapp:producer,consumer
```

```mermaid
graph LR
    subgraph "User Nodes"
        producer[producer]
        consumer[consumer]
    end
    
    subgraph "Implicit Nodes"
        log_capture[LogHandler.node]
        log_printer[LogPrinter]
    end
    
    producer -->|data| consumer
    log_capture -->|logs| log_printer
    
    style log_capture fill:#e1f5fe
    style log_printer fill:#e1f5fe
```

*Python logging from any node is captured and printed to console.*

### Run with Logging to Disk

```bash
tinman run myapp:producer,consumer --log-dir logs/
```

```mermaid
graph LR
    subgraph "User Nodes"
        producer[producer]
        consumer[consumer]
    end
    
    subgraph "Implicit Nodes"
        log_capture[LogHandler.node]
        log_printer[LogPrinter]
        logger[LoggingNode]
    end
    
    producer -->|data| consumer
    producer -->|data| logger
    log_capture -->|logs| log_printer
    log_capture -->|logs| logger
    
    logger -->|writes| disk[(logs/)]
    
    style log_capture fill:#e1f5fe
    style log_printer fill:#e1f5fe
    style logger fill:#e1f5fe
```

*All output channels (including logs) are written to disk.*

---

## `tinman playback`

### Playback with Log Capture (default)

```bash
tinman playback --from logs/ myapp:processor
```

```mermaid
graph LR
    subgraph "Implicit Nodes"
        playback[PlaybackNode]
        log_capture[LogHandler.node]
        log_printer[LogPrinter]
    end
    
    subgraph "User Nodes"
        processor[processor]
    end
    
    disk[(logs/)] -->|reads| playback
    playback -->|data| processor
    log_capture -->|logs| log_printer
    
    style playback fill:#e1f5fe
    style log_capture fill:#e1f5fe
    style log_printer fill:#e1f5fe
```

*Recorded data is played back. New Python logs from `processor` are captured and printed.*

### Playback without Log Capture (view recorded logs)

```bash
tinman playback --from logs/ myapp:processor --no-capture-logs
```

```mermaid
graph LR
    subgraph "Implicit Nodes"
        playback[PlaybackNode]
        log_printer[LogPrinter]
    end
    
    subgraph "User Nodes"
        processor[processor]
    end
    
    disk[(logs/)] -->|reads| playback
    playback -->|data| processor
    playback -->|logs| log_printer
    
    style playback fill:#e1f5fe
    style log_printer fill:#e1f5fe
```

*Recorded data AND recorded logs are played back. The LogPrinter shows the original logs from when the data was recorded.*

### Playback with Re-logging

```bash
tinman playback --from logs/ myapp:processor --log-dir processed/
```

```mermaid
graph LR
    subgraph "Implicit Nodes"
        playback[PlaybackNode]
        log_capture[LogHandler.node]
        log_printer[LogPrinter]
        logger[LoggingNode]
    end
    
    subgraph "User Nodes"
        processor[processor]
    end
    
    disk_in[(logs/)] -->|reads| playback
    playback -->|data| processor
    processor -->|output| logger
    log_capture -->|logs| log_printer
    log_capture -->|logs| logger
    logger -->|writes| disk_out[(processed/)]
    
    style playback fill:#e1f5fe
    style log_capture fill:#e1f5fe
    style log_printer fill:#e1f5fe
    style logger fill:#e1f5fe
```

*Recorded data is played back through processor. New outputs and new logs are written to a new directory.*

---

## Node Legend

| Node | Purpose |
|------|---------|
| **PlaybackNode** | Reads recorded channels from `.blog` files and publishes them with original timing |
| **LogHandler.node** | Captures Python `logging` calls and publishes them as `LogEntry` messages |
| **LogPrinter** | Subscribes to logs channel and prints entries to console |
| **LoggingNode** | Subscribes to all output channels and writes them to `.blog` files |

All implicit nodes are **daemons** - they are cancelled when user nodes complete.

---

## Channel Flow Summary

```mermaid
flowchart TB
    subgraph "Recording (run --log-dir)"
        A[User Nodes] -->|output channels| B[LoggingNode]
        C[Python logging] --> D[LogHandler]
        D -->|logs channel| B
        B --> E[(disk)]
    end
    
    subgraph "Playback (playback --from)"
        F[(disk)] --> G[PlaybackNode]
        G -->|recorded channels| H[User Nodes]
    end
```
