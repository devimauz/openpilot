package hu.tunelab.navbridge

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat

class MainActivity : AppCompatActivity() {
  private lateinit var tvLog: TextView

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)
    tvLog = findViewById(R.id.tvLog)

    findViewById<EditText>(R.id.etHost).setText("192.168.0.22")
    findViewById<EditText>(R.id.etDest).setText("47.2254,16.6288")

    findViewById<Button>(R.id.btnGo).setOnClickListener { ensurePerms { startBridge() } }
    findViewById<Button>(R.id.btnStop).setOnClickListener {
      stopService(Intent(this, BridgeService::class.java))
      tvLog.text = "stopped"
    }
  }

  private fun startBridge() {
    val host = findViewById<EditText>(R.id.etHost).text.toString().trim()
    val dest = findViewById<EditText>(R.id.etDest).text.toString().trim()
    if (dest.count { it == ',' } != 1) {
      tvLog.text = "need 'lat,lon' destination (host empty = auto-discover)"; return
    }
    startForegroundService(Intent(this, BridgeService::class.java).apply {
      putExtra(BridgeService.EXTRA_HOST, host)
      putExtra(BridgeService.EXTRA_API_KEY, BuildConfig.TOMTOM_API_KEY)
      putExtra(BridgeService.EXTRA_DEST, dest)
    })
    tvLog.text = "starting bridge -> host=${host.ifEmpty { "AUTO" }}, dest $dest\n(watch the notification for live status)"
  }

  private fun ensurePerms(go: () -> Unit) {
    val needed = listOf(Manifest.permission.ACCESS_FINE_LOCATION,
      Manifest.permission.ACCESS_COARSE_LOCATION, Manifest.permission.POST_NOTIFICATIONS)
      .filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
    if (needed.isEmpty()) go()
    else ActivityCompat.requestPermissions(this, needed.toTypedArray(), 7)
  }

  override fun onRequestPermissionsResult(requestCode: Int, perms: Array<out String>, results: IntArray) {
    super.onRequestPermissionsResult(requestCode, perms, results)
    if (requestCode == 7 && results.isNotEmpty() && results.all { it == PackageManager.PERMISSION_GRANTED })
      startBridge()
  }
}
