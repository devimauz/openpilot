package hu.tunelab.navbridge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat

class BridgeService : Service(), NavSource {
  companion object {
    const val EXTRA_HOST = "host"
    const val EXTRA_API_KEY = "key"
    const val EXTRA_DEST = "dest"            // "lat,lon"
    var statusText: String = "idle"
    val logs = ArrayDeque<String>()
  }

  private var feed: FeedClient? = null
  private var inner: NavSource? = null
  private val state = object { val lock = Any() }

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    createChannel()
    val host = intent?.getStringExtra(EXTRA_HOST) ?: run { stopSelf(); return START_NOT_STICKY }
    val key = intent?.getStringExtra(EXTRA_API_KEY) ?: ""
    val dest = intent?.getStringExtra(EXTRA_DEST)?.split(",")
      ?: run { stopSelf(); return START_NOT_STICKY }
    startForeground(1, notif("NavBridge: connecting $host"))

    log("bridge start -> $host")
    feed?.close(); inner?.stop()
    val client = FeedClient(host, onLog = { log(it) }, onDead = {
      log("feed dead: $it — restart the bridge after 5 s")
      statusText = "dead: $it"; refresh()
    })
    feed = client
    client.connect {
      val src = TomTomRestNavSource(applicationContext, key,
        onUpdate = { snap ->
          val n = client.pushItems(snap.buildItems())
          statusText = "feeding — ${snap.routeRemainM} m left, sdi pushed=$n"
          refresh()
        },
        onLog = { log(it) })
      inner = src
      src.startRoute(dest[0].toDouble(), dest[1].toDouble())
    }
    return START_NOT_STICKY
  }

  override fun startRoute(destLat: Double, destLon: Double) { /* driven via Intent */ }
  override fun stop() { feed?.close(); feed = null; inner?.stop(); stopSelf() }

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
    sendBroadcast(Intent("hu.tunelab.navbridge.LOG"))
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
