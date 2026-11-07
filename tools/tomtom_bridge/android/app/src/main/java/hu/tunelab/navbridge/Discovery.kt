// Listen for the comma's UDP :7705 discovery beacon {"ip": "...", "navi_debug": 1}.
package hu.tunelab.navbridge

import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

object Discovery {
  const val PORT = 7705

  /** Blocks until a beacon arrives or timeoutMs elapses; returns the comma IP or null. */
  fun findComma(timeoutMs: Int): String? {
    val sock = DatagramSocket(null).apply {
      reuseAddress = true
      broadcast = true
      bind(java.net.InetSocketAddress(0 + PORT))
      soTimeout = 1000
    }
    val buf = ByteArray(512)
    val deadline = System.currentTimeMillis() + timeoutMs
    try {
      while (System.currentTimeMillis() < deadline) {
        val pkt = DatagramPacket(buf, buf.size)
        try {
          sock.receive(pkt)
        } catch (_: java.net.SocketTimeoutException) {
          continue
        }
        val text = String(pkt.data, 0, pkt.length)
        val ip = Regex("\"ip\"\\s*:\\s*\"([0-9.]+)\"").find(text)?.groupValues?.get(1)
        if (ip != null) return ip
      }
      return null
    } finally {
      sock.close()
    }
  }
}
