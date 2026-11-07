// TomTom Routing API (v1, instructionsType=text + sectionType=speedLimit) -> NaviSnapshot.
// Verified against the real response shape: routes[0].legs[0].points geometry,
// guidance.instructions[] with routeOffsetInMeters / maneuver / junctionType /
// roundaboutExitNumber / turnAngleInDecimalDegrees, flat sections[] SPEED_LIMIT ranges.
package hu.tunelab.navbridge

import android.content.Context
import android.location.Location
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.Priority
import com.google.gson.Gson
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.cos
import kotlin.math.hypot

class TomTomRestNavSource(
  private val context: Context,
  private val apiKey: String,
  private val onUpdate: (NaviSnapshot) -> Unit,
  private val onLog: (String) -> Unit,
) : NavSource {
  data class RoutePt(val lat: Double, val lon: Double)
  private data class Instruction(
    val maneuver: String, val offsetM: Double, val street: String, val message: String,
    val junction: String?, val exitNumber: Int?, val turnAngle: Double?,
    val point: Pair<Double, Double>?, val timeSec: Int,
  )

  private var routePoints: List<RoutePt> = emptyList()
  private var cum: DoubleArray = DoubleArray(0)
  private var totalM = 0
  private var instructions: List<Instruction> = emptyList()
  private var limitSections: List<Triple<Int, Int, Int>> = emptyList() // startIdx,endIdx,kph
  private var lastLoc: Location? = null
  private var movedM = 0.0
  private val fused = com.google.android.gms.location.LocationServices
    .getFusedLocationProviderClient(context)
  private val scope = CoroutineScope(Dispatchers.Default)
  private var loopJob: Job? = null
  private var pendingDest: Pair<Double, Double>? = null
  @Volatile private var routeFetching = false
  private var routeRetryAt = 0L

  override fun startRoute(destLat: Double, destLon: Double) {
    pendingDest = destLat to destLon
    routePoints = emptyList()
    startLocation()
    loopJob?.cancel()
    loopJob = scope.launch { while (true) { tick(); delay(1000) } }
  }

  override fun stop() { loopJob?.cancel(); fused.removeLocationUpdates(cb); pendingDest = null }

  private val cb = object : LocationCallback() {
    override fun onLocationResult(r: LocationResult) { r.lastLocation?.let { lastLoc = it } }
  }

  private fun startLocation() {
    fused.requestLocationUpdates(
      LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L).build(), cb, context.mainLooper)
  }

  private fun fetchRoute(dest: Pair<Double, Double>) {
    val origin = lastLoc ?: return
    routeFetching = true
    scope.launch(Dispatchers.IO) {
      val url = "https://api.tomtom.com/routing/1/calculateRoute/" +
        "${origin.latitude},${origin.longitude}:${dest.first},${dest.second}/json" +
        "?instructionsType=text&language=en-GB&sectionType=speedLimit&key=$apiKey"
      var body = ""
      try {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 8000; conn.readTimeout = 10000
        body = (if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream)
          .bufferedReader().readText()
        parseRoute(body)
        pendingDest = null
        onLog("route ok: ${routePoints.size} pts, ${instructions.size} maneuvers, " +
          "${limitSections.size} limit sections, ${totalM} m")
      } catch (e: Exception) {
        routeRetryAt = System.currentTimeMillis() + 5000
        onLog("route failed: ${e.message} ${body.take(120)} — retry 5 s")
      } finally {
        routeFetching = false
      }
    }
  }

  @Suppress("UNCHECKED_CAST")
  private fun parseRoute(json: String) {
    val root = Gson().fromJson(json, Map::class.java) as Map<String, Any?>
    (root["detailedError"] as? Map<*, *>)?.let {
      error("tomtom: ${it["message"] ?: it["code"]}")
    }
    val route = ((root["routes"] as? List<Map<String, Any?>>)?.firstOrNull())
      ?: error("no route in response")
    val summary = route["summary"] as Map<String, Any?>
    totalM = ((summary["lengthInMeters"] as? Double)?.toInt()) ?: 0
    val leg = (((route["legs"] as List<Map<String, Any?>>).firstOrNull())
      ?: error("no leg"))
    val pts = (leg["points"] as List<Map<String, Any?>>).map {
      RoutePt(it["latitude"] as Double, it["longitude"] as Double)
    }
    routePoints = pts
    cum = DoubleArray(pts.size)
    for (i in 1 until pts.size) cum[i] = cum[i - 1] + meters(pts[i - 1], pts[i])

    instructions = buildList {
      val ins = (((route["guidance"] as? Map<String, Any?>)?.get("instructions")
        as? List<Map<String, Any?>>).orEmpty())
      for (x in ins) {
        val mnr = x["maneuver"]?.toString() ?: continue
        val off = (x["routeOffsetInMeters"] as? Double) ?: continue
        val street = when (val s = x["street"]) {
          is String -> s
          is List<*> -> (s.firstOrNull() as? Map<*, *>)?.get("value")?.toString() ?: ""
          is Map<*, *> -> s["value"]?.toString() ?: ""
          else -> ""
        }
        val pt = (x["point"] as? Map<String, Any?>)?.let {
          (it["latitude"] as? Double)?.let { la -> (it["longitude"] as? Double)?.let { lo -> la to lo } }
        }
        add(Instruction(mnr, off, street.take(96), (x["message"]?.toString() ?: "").take(160),
          x["junctionType"]?.toString(),
          (x["roundaboutExitNumber"] as? Double)?.toInt(),
          (x["turnAngleInDecimalDegrees"] as? Double),
          pt, ((x["travelTimeInSeconds"] as? Double)?.toInt() ?: 0)))
      }
    }
    limitSections = buildList {
      val secs = (route["sections"] as? List<Map<String, Any?>>).orEmpty()
      for (s in secs) {
        if (s["sectionType"]?.toString() != "SPEED_LIMIT") continue
        val st = (s["startPointIndex"] as? Double)?.toInt() ?: continue
        val en = (s["endPointIndex"] as? Double)?.toInt() ?: continue
        val kmh = (s["maxSpeedLimitInKph"] as? Double)?.toInt()
          ?: (s["maxSpeedLimitInKmh"] as? Double)?.toInt() ?: continue
        add(Triple(st, en, kmh))
      }
    }
  }

  private fun tick() {
    val loc = lastLoc ?: return
    val dest = pendingDest
    if (dest != null && !routeFetching && System.currentTimeMillis() >= routeRetryAt) {
      onLog("waiting for GPS fix...")
      fetchRoute(dest)
    }
    if (routePoints.isEmpty()) return
    movedM = projectProgress(loc.latitude, loc.longitude)
    val remaining = (totalM - movedM).coerceAtLeast(0.0)

    val here = limitSections.firstOrNull { movedM >= cum[it.first] && movedM < cum[it.second] }
      ?: limitSections.lastOrNull { movedM >= cum[it.first] }   // carry forward through data gaps
    val upcoming = limitSections.filter { cum[it.first] > movedM }.sortedBy { cum[it.first] }
    val nextChange = upcoming.firstOrNull()?.let { LimitChange((cum[it.first] - movedM).toInt(), it.third) }
    val secondChange = upcoming.drop(1).firstOrNull()?.let { LimitChange((cum[it.first] - movedM).toInt(), it.third) }

    val ahead = instructions.filter { it.offsetM > movedM - 20 && it.maneuver != "DEPART" }
      .sortedBy { it.offsetM }
    val now = ahead.firstOrNull()?.toManeuver(movedM)
    val nxt = ahead.drop(1).firstOrNull()?.toManeuver(movedM)

    val snap = NaviSnapshot(
      lat = loc.latitude, lon = loc.longitude,
      headingDeg = if (loc.hasBearing()) loc.bearing.toDouble() else 0.0,
      speedKph = if (loc.hasSpeed()) loc.speed * 3.6 else 0.0,
      currentLimitKph = here?.third ?: 0,
      nextChange = nextChange, secondChange = secondChange,
      maneuverNow = now, maneuverNext = nxt,
      routePolyline = routePoints.map { it.lat to it.lon },
      routeRemainM = remaining.toInt(), routeMovedM = movedM.toInt(), routeTotalM = totalM,
      routeRemainSec = ((totalM - movedM) / 13.9).toInt(),
      guidanceActive = true, offRoute = offRoute(loc.latitude, loc.longitude),
      roadName = now?.roadName ?: "",
    )
    onUpdate(snap)
  }

  private fun Instruction.toManeuver(moved: Double) = Maneuver(
    type = maneuver, distanceM = (offsetM - moved).coerceAtLeast(0.0).toInt(),
    timeSec = timeSec, roadName = street, mainText = message,
    point = point, exitNumber = exitNumber, junction = junction, turnAngleDeg = turnAngle,
  )

  private fun projectProgress(lat: Double, lon: Double): Double {
    // v0.13: reverted to the v0.11 midpoint approximation on user request (v0.12 exact
    // perpendicular projection rolled back). moved_m can lead by up to half a segment.
    var best = 0.0
    var bestD = Double.MAX_VALUE
    for (i in 0 until routePoints.size - 1) {
      val d = pointSegMeters(lat, lon, routePoints[i], routePoints[i + 1])
      if (d < bestD) { bestD = d; best = cum[i] + meters(routePoints[i], routePoints[i + 1]) * 0.5 }
    }
    return best
  }

  private fun offRoute(lat: Double, lon: Double): Boolean {
    var best = Double.MAX_VALUE
    for (i in 0 until routePoints.size - 1)
      best = minOf(best, pointSegMeters(lat, lon, routePoints[i], routePoints[i + 1]))
    return best > 50
  }

  private fun meters(a: RoutePt, b: RoutePt) = hypot(
    Math.toRadians(b.lat - a.lat) * 6371000.0,
    Math.toRadians(b.lon - a.lon) * 6371000.0 * cos(Math.toRadians((a.lat + b.lat) / 2)))

  private fun pointSegMeters(lat: Double, lon: Double, a: RoutePt, b: RoutePt): Double {
    val kx = 111320.0 * cos(Math.toRadians(lat))
    val px = (lon - a.lon) * kx
    val py = (lat - a.lat) * 110540.0
    val dx = (b.lon - a.lon) * kx
    val dy = (b.lat - a.lat) * 110540.0
    val len2 = dx * dx + dy * dy
    val t = if (len2 == 0.0) 0.0 else ((px * dx + py * dy) / len2).coerceIn(0.0, 1.0)
    return hypot(px - t * dx, py - t * dy)
  }
}
