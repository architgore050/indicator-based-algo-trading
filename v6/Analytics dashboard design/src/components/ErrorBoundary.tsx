import { Component, type ReactNode, type ErrorInfo } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  public state: State = { hasError: false, error: null };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught:", error);
    console.error("Component stack:", errorInfo.componentStack);
  }

  public render() {
    if (this.state.hasError && this.state.error) {
      return (
        <div style={{
          height: "100vh",
          background: "#0d1117",
          color: "#f85149",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: 20,
          fontFamily: "monospace",
          fontSize: 12,
        }}>
          <div style={{ fontSize: 24, marginBottom: 16, color: "#f85149" }}>⚠ React Error</div>
          <div style={{ maxWidth: 600, textAlign: "center" }}>{this.state.error.message}</div>
          <details style={{ marginTop: 16, maxWidth: 600, textAlign: "left", color: "#8b949e" }}>
            <summary>Stack trace</summary>
            <pre style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>{this.state.error.stack}</pre>
          </details>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginTop: 16,
              padding: "8px 16px",
              background: "#238636",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
            }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
