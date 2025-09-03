"""
N1 backend, extracted from the provided Tkinter GUI.

This module provides a pure-Python client you can import into your own scripts
without any GUI dependencies. It handles:
  • Connecting/disconnecting to the four TCP ports
  • Framed protocol parsing (header/payload/footer with magic numbers)
  • Receiving graph data (8089) and text data (8090 dual-page, 8091 single-page)
  • Sending size-prefixed commands on 8092
  • Threaded receivers with user callbacks

USAGE (minimal):

    from n1_backend_client import N1Client

    def on_graph(frame_num, rows, cols, data, sample_rate_hz):
        print("graph:", frame_num, rows, cols, data.shape, sample_rate_hz)

    def on_text(port, frame_num, rows, cols, lines, checksum):
        print(f"text on {port}: frame={frame_num} rows={rows} cols={cols} checksum={checksum}")
        # print("\n".join(lines))

    client = N1Client(server_ip="192.168.0.10",
                      on_graph_frame=on_graph,
                      on_text_frame=on_text,
                      on_status=print)

    client.connect_all()
    client.send_command("HELLO")
    # ... do work ...
    client.disconnect_all()

Notes:
- Callbacks are optional; if omitted, events are ignored.
- For 8089 (graph data), this class computes dt = 1/cols and applies the same
  column reduction as the GUI (cols_adjusted = round(cols * 0.042667)) before
  reshaping to (rows*4, cols_adjusted).
- For 8090 (dual-page text), this class emits whichever page is active via
  set_dual_page(page_index) where 0 = first, 1 = second.
"""
from __future__ import annotations

import socket
import threading
import struct
import select
import numpy as np
from typing import Callable, Optional, Dict, Any, List


class N1Client:
    """Headless N1 TCP client encapsulating connections and protocol.

    Ports and meanings (defaults):
      8089: Data Stream (graph data)
      8090: Text Display 1 (dual-page)
      8091: Text Display 2 (single-page)
      8092: Command Channel (size-prefixed strings)
    """

    # Protocol magic constants
    START_MAGIC = b"KCLB"
    END_MAGIC = b"DNEB"
    FINAL_MAGIC = b"KCLB"

    def __init__(
        self,
        server_ip: str,
        *,
        ports: Optional[Dict[int, str]] = None,
        on_graph_frame: Optional[Callable[[int, int, int, np.ndarray, float], None]] = None,
        on_text_frame: Optional[Callable[[int, int, int, int, List[str], int], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.server_ip = server_ip
        self.on_graph_frame = on_graph_frame
        self.on_text_frame = on_text_frame
        self.on_status = on_status

        # Connection state
        self.connections: Dict[int, Dict[str, Any]] = {
            8089: {"connected": False, "socket": None, "name": "Data Stream"},
            8090: {"connected": False, "socket": None, "name": "Text Display 1"},
            8091: {"connected": False, "socket": None, "name": "Text Display 2"},
            8092: {"connected": False, "socket": None, "name": "Command Channel"},
        }
        if ports:
            # Allow caller to override/extend the port map (e.g., use different numbers)
            for p, label in ports.items():
                self.connections[p] = {"connected": False, "socket": None, "name": label}

        # Receiver control
        self._recv_threads: Dict[int, threading.Thread] = {}
        self._recv_stop_flags: Dict[int, threading.Event] = {}

        # Text 8090 page selection (0 = first, 1 = second)
        self._dual_page_index = 0

    # ------------------ Public API ------------------
    def set_dual_page(self, index: int) -> None:
        """Select which page (0 or 1) is emitted for 8090 frames."""
        self._dual_page_index = 0 if index <= 0 else 1

    def connect_all(self, timeout_s: float = 5.0) -> None:
        for port in list(self.connections.keys()):
            if not self.connections[port]["connected"]:
                self.connect(port, timeout_s=timeout_s)
        self._log("Attempted to connect to all ports")

    def disconnect_all(self) -> None:
        for port in list(self.connections.keys()):
            if self.connections[port]["connected"]:
                self.disconnect(port)
        self._log("Disconnected from all ports")

    def connect(self, port: int, *, timeout_s: float = 5.0) -> None:
        """Open a TCP socket and start a receiver thread for data ports."""
        try:
            self._log(f"Connecting to {self.server_ip}:{port} ...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout_s)
            s.connect((self.server_ip, port))
            s.settimeout(None)  # switch to blocking

            self.connections[port]["socket"] = s
            self.connections[port]["connected"] = True

            # Start receiver for data ports
            if port in (8089, 8090, 8091):
                stop_evt = threading.Event()
                self._recv_stop_flags[port] = stop_evt
                t = threading.Thread(target=self._recv_loop, args=(port, stop_evt), daemon=True)
                t.start()
                self._recv_threads[port] = t

            self._log(f"Connected on port {port}")
        except Exception as e:
            self._log(f"Connection failed on port {port}: {e}")
            raise

    def disconnect(self, port: int) -> None:
        """Close a TCP socket and stop its receiver thread if any."""
        # Signal thread stop first
        if port in self._recv_stop_flags:
            self._recv_stop_flags[port].set()

        s = self.connections[port].get("socket")
        if s:
            try:
                s.close()
            except Exception as e:
                self._log(f"Error closing socket on port {port}: {e}")
        self.connections[port]["socket"] = None
        self.connections[port]["connected"] = False
        self._log(f"Disconnected from port {port}")

    def send_command(self, text: str) -> None:
        """Send a size-prefixed UTF-8 command via port 8092."""
        if not self.connections.get(8092, {}).get("connected"):
            raise RuntimeError("Not connected to command server on port 8092")
        try:
            payload = text.encode("utf-8")
            size_bytes = struct.pack("!I", len(payload))
            self.connections[8092]["socket"].sendall(size_bytes + payload)
            self._log(f"SENT: {text}")
        except Exception as e:
            self._log(f"Failed to send command: {e}")
            raise

    # ------------------ Internal: receiving ------------------
    def _recv_loop(self, port: int, stop_evt: threading.Event) -> None:
        """Continuously read framed messages on a data port and dispatch."""
        try:
            while self.connections[port]["connected"] and not stop_evt.is_set():
                header = self._read_exact(port, 20, timeout_s=5.0)
                if not header:
                    break
                magic, frame_num, payload_size, rows, cols = struct.unpack("<4sIIII", header)
                if magic != self.START_MAGIC:
                    self._log(f"Invalid start magic on port {port}")
                    continue

                payload = self._read_exact(port, payload_size, timeout_s=5.0)
                if payload is None:
                    continue

                footer = self._read_exact(port, 12, timeout_s=5.0)
                if not footer:
                    continue
                end_magic, checksum, final_magic = struct.unpack("<4sI4s", footer)
                if end_magic != self.END_MAGIC or final_magic != self.FINAL_MAGIC:
                    self._log(f"Invalid footer magic on port {port}")
                    continue

                # Dispatch by port
                if port == 8089:
                    self._handle_graph_payload(frame_num, rows, cols, payload)
                elif port == 8090:
                    self._handle_text_payload(port, frame_num, rows, cols, payload, checksum, dual_page=True)
                elif port == 8091:
                    self._handle_text_payload(port, frame_num, rows, cols, payload, checksum, dual_page=False)
        except Exception as e:
            self._log(f"Error in receive loop on port {port}: {e}")
        finally:
            self._log(f"Receive loop ended for port {port}")

    def _read_exact(self, port: int, nbytes: int, *, timeout_s: float) -> Optional[bytes]:
        s = self.connections[port].get("socket")
        if not s:
            return None
        buf = b""
        while len(buf) < nbytes:
            if not self.connections[port]["connected"]:
                return None
            rready, _, _ = select.select([s], [], [], timeout_s)
            if not rready:
                self._log(f"Timeout waiting for data on port {port}")
                return None
            chunk = s.recv(nbytes - len(buf))
            if not chunk:
                # remote closed
                self.disconnect(port)
                return None
            buf += chunk
        return buf

    # ------------------ Internal: payload handlers ------------------
    def _handle_graph_payload(self, frame_num: int, rows: int, cols: int, payload: bytes) -> None:
        # dt and adjusted columns per original GUI logic
        sample_rate_hz = float(cols)
        cols_adjusted = int(round(cols * 0.042667))
        try:
            data = np.frombuffer(payload, dtype=np.float32)
            data = data.reshape(rows * 4, cols_adjusted)
        except Exception as e:
            self._log(f"Error decoding graph frame {frame_num}: {e}")
            return
        if self.on_graph_frame:
            try:
                self.on_graph_frame(frame_num, rows, cols_adjusted, data, sample_rate_hz)
            except Exception as cb_e:
                self._log(f"on_graph_frame callback error: {cb_e}")

    def _handle_text_payload(
        self,
        port: int,
        frame_num: int,
        rows: int,
        cols: int,
        payload: bytes,
        checksum: int,
        *,
        dual_page: bool,
    ) -> None:
        try:
            if dual_page:
                half_cols = cols // 2
                arr = np.frombuffer(payload, dtype=np.uint8).reshape(rows * 2, half_cols)
                # pages: [0]=first rows, [1]=second rows
                page0 = arr[:rows]
                page1 = arr[rows:]
                page = page0 if self._dual_page_index == 0 else page1
                lines = [self._uint8_row_to_string(row) for row in page]
            else:
                # single page: split by row-width 'cols'
                lines = [self._uint8_row_to_string(payload[i:i+cols]) for i in range(0, len(payload), cols)]
        except Exception as e:
            self._log(f"Error decoding text frame {frame_num} on {port}: {e}")
            return

        if self.on_text_frame:
            try:
                self.on_text_frame(port, frame_num, rows, cols, lines, checksum)
            except Exception as cb_e:
                self._log(f"on_text_frame callback error: {cb_e}")

    # ------------------ Helpers ------------------
    @staticmethod
    def _uint8_row_to_string(uint8_row: np.ndarray | bytes) -> str:
        # Keep printable ASCII (32..126) and allow DC4 (20) like the GUI
        return "".join(chr(b) if (32 <= b <= 126) or b == 20 else "" for b in uint8_row)

    def _log(self, msg: str) -> None:
        if self.on_status:
            try:
                self.on_status(msg)
                return
            except Exception:
                pass
        # Fallback to stdout
        print(msg)


# Optional: convenience main for quick manual testing
if __name__ == "__main__":
    def _g(fnum, rows, cols_adj, data, sr):
        print(f"[GRAPH] frame={fnum} rows={rows} cols_adj={cols_adj} data_shape={data.shape} sr={sr:.1f} Hz")

    def _t(port, fnum, rows, cols, lines, checksum):
        print(f"[TEXT {port}] frame={fnum} rows={rows} cols={cols} checksum={checksum} first_line={lines[0][:60] if lines else ''}")

    client = N1Client("192.168.0.10", on_graph_frame=_g, on_text_frame=_t, on_status=print)
    try:
        client.connect_all()
        client.send_command("PING")
        input("Press Enter to disconnect... ")
    finally:
        client.disconnect_all()
