// carrot_navi v2 client: control ws -> manifest -> per-item json websockets.
// One instance per session; on failure the owner creates a fresh FeedClient.
package hu.tunelab.navbridge

import com.google.gson.Gson
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.ConcurrentHashMap

class FeedClient(
  private val host: String,
  private val port: Int = 7714,
  private val appVersion: String = "tomtom-bridge-0.5",
  private val onLog: (String) -> Unit = {},
  private val onDead: (String) -> Unit = {},
) {
  private val gson = Gson()
  private val http = OkHttpClient()
  private val seq = ConcurrentHashMap<String, Int>()
  private val socks = ConcurrentHashMap<String, WebSocket>()
  private val lastSent = ConcurrentHashMap<String, String>()
  private var control: WebSocket? = null

  @Volatile var sessionId: String? = null
    private set
  @Volatile var connected = false
    private set

  private val requirements: Map<String, Any> = mapOf(
    "type" to "requirements_query",
    "protocol_version" to Protocol.PROTOCOL_VERSION,
    "catalog_revision" to Protocol.CATALOG_REVISION,
    "streams" to Protocol.CATALOG.map {
      mapOf("kind" to it.first, "name" to it.second, "schema_version" to Protocol.SCHEMA_VERSION)
    },
  )

  fun connect(onReady: () -> Unit = {}) {
    val req = Request.Builder().url("ws://$host:$port/api/navi/ws/v2/control/$appVersion").build()
    control = http.newWebSocket(req, object : WebSocketListener() {
      override fun onOpen(ws: WebSocket, response: Response) {
        ws.send(gson.toJson(requirements))
        onLog("control ws open, catalog sent (${Protocol.CATALOG.size} streams)")
      }
      override fun onMessage(ws: WebSocket, text: String) {
        val map = try {
          gson.fromJson(text, Map::class.java)
        } catch (e: Exception) {
          null
        } ?: return
        when (map["type"]) {
          "subscription_manifest" -> {
            sessionId = map["session_id"] as? String
            openItemSockets()
            connected = true
            onLog("navi session=$sessionId ready")
            onReady()
          }
          "protocol_error" -> {
            onLog("control error: ${map["message"]}")
            onDead(map["message"].toString())
          }
        }
      }
      override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
        connected = false
        onLog("control ws lost: ${t.message}")
        onDead(t.message ?: "control failure")
      }
      override fun onClosing(ws: WebSocket, code: Int, reason: String) {
        connected = false
        onDead("closing: $reason")
      }
    })
  }

  private fun openItemSockets() {
    val sid = sessionId ?: return
    for (name in Protocol.JSON_NAMES) {
      val req = Request.Builder().url("ws://$host:$port/api/navi/ws/v2/json/$sid/$name").build()
      http.newWebSocket(req, object : WebSocketListener() {
        override fun onOpen(ws: WebSocket, response: Response) { socks[name] = ws }
        override fun onFailure(ws: WebSocket, t: Throwable, r: Response?) {
          socks.remove(name)
          onLog("item ws '$name' lost: ${t.message}")
        }
      })
      seq[name] = 0
    }
  }

  /** Push changed items (mirrors Python push_items); returns count sent. */
  fun pushItems(items: Map<String, Map<String, Any?>>, force: Boolean = false): Int {
    if (!connected) return 0
    val sid = sessionId ?: return 0
    var sent = 0
    for ((name, item) in items) {
      val ws = socks[name] ?: continue
      val rendered = gson.toJson(item)
      if (!force && lastSent[name] == rendered) continue
      val s = (seq[name] ?: 0) + 1
      seq[name] = s
      val now = System.currentTimeMillis()
      val env = mutableMapOf<String, Any?>(
        "type" to "item_update", "protocol_version" to Protocol.PROTOCOL_VERSION,
        "session_id" to sid, "kind" to "json", "name" to name,
        "sequence" to s, "source_timestamp_ms" to now, "sent_at_ms" to now)
      if (item["present"] == true) {
        env["present"] = true
        env["value"] = item["value"]
      } else {
        env["present"] = false
        env["value"] = null
        env["reason"] = item["reason"] ?: "source_absent"
      }
      ws.send(gson.toJson(env))
      lastSent[name] = rendered
      sent++
    }
    return sent
  }

  fun close() {
    connected = false
    socks.values.forEach { it.close(1000, "session end") }
    socks.clear()
    control?.close(1000, "session end")
    control = null
  }
}
