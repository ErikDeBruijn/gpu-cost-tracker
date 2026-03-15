## Known Issues

- **Double counting**: If an unregistered workload (e.g., ollama inference) runs on the same GPU as a tracked job, the tracked job gets attributed the extra power. Need process-level isolation or PID-based power attribution to fix this.

## Future

- Persist history to disk (currently in-memory, lost on restart)
- CORS headers for cross-origin UI embedding
