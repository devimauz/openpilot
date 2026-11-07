// v0.6 supervisor: auto-discover comma IP (or use given host), auto-reconnect,
// keep the nav source alive across feed drops.
package hu.tunelab.navbridge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

class BridgeService : Service(), NavSource {
  companion object {
    const val EXTRA_HOST = "host"
    const val EXTRA_API_KEY = "key"
    const val EXTRA_DEST = "dest"
    var statusText: String = "idle"
    val logs = ArrayDeque<String>()
  }

  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
  private val state = object { val lock = Any() }
  @Volatile private var feed: FeedClient? = null
  @Volatile private var src: NavSource? = null
  @Volatile private var dest: Pair<Double, Double>? = null
  @Volatile private var manualHost: String? = null
  @Volatile private var running = false

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    createChannel()
    startForeground(1, notif("NavBridge: starting"))
    val d = intent?.getStringExtra(EXTRA_DEST)?.split(",") ?: run { stopSelf(); return START_NOT_STICKY }
    val hostArg = intent?.getStringExtra(EXTRA_HOST)?.trim()
    manualHost = if (hostArg.isNullOrEmpty() || hostArg.equals("auto", true)) null else hostArg
    dest = d[0].trim().toDouble() to d[1].trim().toDouble()

    if (running) {          // re-START (new destination/host): restart session cleanly
      stopSession()
    }
    running = true

    val nav = TomTomRestNavSource(applicationContext,
      intent.getStringExtra(EXTRA_API_KEY) ?: "",
      onUpdate = { snap ->
        val n = feed?.pushItems(snap.buildItems()) ?: 0
        statusText = "feeding — ${snap.routeRemainM} m left (pushed $n)"
        refresh()
      },
      onLog = { log(it) })
    src = nav

    scope.launch {
      while (isActive) {
        val host = manualHost ?: withContext2 { Discovery.findComma(6000) }
        if (host == null) {
          log("waiting for comma beacon (UDP :7705)...")
          delay(4000)
          continue
        }
        connectAndHold(host)        // returns only when the feed died / connect failed
        if (!isActive) break
        log("feed to $host down — retry in 4 s")
        delay(4000)
      }
    }

    nav.startRoute(dest!!.first, dest!!.second)
    return START_NOT_STICKY
  }

  private suspend fun withContext2(block: suspend () -> String?): String? =
    kotlinx.coroutines.withContext(Dispatchers.IO) { block() }

  /** Suspends until the feed session ends (dead or connect failure). */
  private suspend fun connectAndHold(host: String) {
    suspendCancellableCoroutine<Unit> { cont ->
      val client = FeedClient(host, onLog = { log(it) }, onDead = {
        feed = null
        if (cont.isActive) cont.resume(Unit)
      })
      feed = client
      client.connect { log("feed ready on $host") }
      cont.invokeOnCancellation { client.close() }
    }
  }

  override fun startRoute(destLat: Double, destLon: Double) { dest = destLat to destLon }

  private fun stopSession() {
    feed?.close(); feed = null
    src?.stop(); src = null
    running = false
  }

  override fun stop() { stopSession(); scope.cancel(); stopSelf() }
  override fun onDestroy() { stop() }

  private fun log(s: String) {
    synchronized(state) {
      logs.addLast(s)
      while (logs.size > 60) logs.removeFirst()
    }
    statusText = s
    refresh()
  }

  private fun refresh() {
    val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    nm.notify(1, notif("NavBridge: $statusText"))
  }

  private fun notif(text: String): Notification = NotificationCompat.Builder(this, "bridge")
    .setContentTitle("NavBridge")
    .setContentText(text)
    .setSmallIcon(android.R.drawable.ic_dialog_info)
    .build()

  private fun createChannel() {
    val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    nm.createNotificationChannel(NotificationChannel("bridge", "NavBridge",
      NotificationManager.IMPORTANCE_LOW))
  }
}
