// JVM unit test mirroring the Python selftest (tools/tomtom_bridge/feed_sim.py --selftest).
package hu.tunelab.navbridge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ProtocolSelfTest {
  private fun snap(t: Int): NaviSnapshot {
    val lat0 = 47.2254; val lon0 = 16.6288
    val s = NaviSnapshot(lat0 + 0.00001 * t, lon0 + 0.000012 * t,
      headingDeg = (90 + 2 * t) % 360.0, speedKph = if (t < 30) 50.0 else 30.0,
      currentLimitKph = if (t < 31) 50 else 30,
      routeRemainM = maxOf(0, 4200 - t * 12), routeTotalM = 4200,
      routeMovedM = t * 12, roadName = "Koeszeg ut")
    s.routePolyline = (0 until 64).map {
      lat0 + 0.004 * Math.sin(it / 64.0 * 2 * Math.PI) to
        lon0 + 0.005 * Math.cos(it / 64.0 * 2 * Math.PI)
    }
    when (t) {
      in 0..30 -> {
        s.nextChange = LimitChange(maxOf(0, 400 - t * 13), 30)
        s.secondChange = LimitChange(900, 50)
        s.maneuverNext = Maneuver("ROUNDABOUT_RIGHT", distanceM = maxOf(0, 620 - t * 16),
          roadName = "Fo ter", mainText = "Roundabout - take exit 2")
      }
      in 31..55 -> {
        s.secondChange = LimitChange(450, 50)
        val d = maxOf(0, 300 - (t - 31) * 11)
        s.maneuverNow = Maneuver("EXIT_ROUNDABOUT", distanceM = d, roadName = "Vorosmarty",
          mainText = "Roundabout - exit 2", exitNumber = 2, numExits = 4,
          radiusM = if (d < 120) 18.0 else null)
      }
      in 56..80 -> {
        s.nextChange = LimitChange(maxOf(0, 150 - (t - 56) * 6), 50)
        s.secondChange = LimitChange(900, 50)
        val d = maxOf(0, 200 - (t - 56) * 8)
        s.maneuverNow = Maneuver("TURN_RIGHT", distanceM = d, roadName = "Szell Kalman korut",
          radiusM = if (d < 80) 14.0 else null)
      }
      else -> s.maneuverNow = Maneuver("ARRIVE", distanceM = maxOf(0, 120 - (t - 81) * 5))
    }
    return s
  }

  @Test fun protocolInvariants() {
    assertEquals(28, Protocol.CATALOG.size)
    assertEquals(13, Protocol.JSON_NAMES.size)
    for (t in intArrayOf(0, 20, 31, 45, 60, 65, 85)) {
      val items = snap(t).buildItems()
      assertEquals(Protocol.JSON_NAMES.toSet(), items.keys)
      assertTrue(items["vehicle"]!!["present"] == true)
      val speed = items["speed"]!!["value"] as Map<*, *>
      (speed["road_limit_kph"] as? Int)?.let {
        assertTrue("limit step", it % 10 == 0 && it in 1..200)
      }
      assertTrue("sdi is enforcement-only", !speed.containsKey("sdi"))
      assertTrue("no sdi_secondary", !speed.containsKey("sdi_secondary"))
    }
    val sp20 = snap(20).buildItems()["speed"]!!["value"] as Map<*, *>
    assertEquals(50, sp20["road_limit_kph"])
    assertEquals("next limit change is NOT fed as sdi", null, sp20["sdi"])
    val sp45 = snap(45).buildItems()["speed"]!!["value"] as Map<*, *>
    assertEquals("roundabout decel rides guidance+road sources", null, sp45["sdi"])
    val sp60 = snap(60).buildItems()["speed"]!!["value"] as Map<*, *>
    assertEquals(30, sp60["road_limit_kph"])
    assertEquals(13, snap(65).maneuverNow!!.turnCode())
    assertEquals(133, snap(45).maneuverNow!!.turnCode())
  }

  @Test fun realTomTomRoundaboutShapes() {
    // junctionType=ROUNDABOUT + turnAngleInDecimalDegrees (negative = right through circle)
    assertEquals(133, Maneuver("ROUNDABOUT_CROSS", junction = "ROUNDABOUT",
      turnAngleDeg = -45.0, exitNumber = 2).turnCode())
    assertEquals(142, Maneuver("ROUNDABOUT_CROSS", junction = "ROUNDABOUT", turnAngleDeg = 0.0).turnCode())
    assertEquals(139, Maneuver("ROUNDABOUT_CROSS", junction = "ROUNDABOUT", turnAngleDeg = 90.0).turnCode())
    assertEquals(135, Maneuver("ROUNDABOUT_CROSS", junction = "ROUNDABOUT", turnAngleDeg = -160.0).turnCode())
    assertEquals(131, Maneuver("ROUNDABOUT_CROSS", junction = "ROUNDABOUT", turnAngleDeg = -15.0).turnCode())
    assertEquals(134, Maneuver("ROUNDABOUT_CROSS", junction = "ROUNDABOUT", turnAngleDeg = -100.0).turnCode())
    // fallback when only exit number is present
    assertEquals(133, Maneuver("ROUNDABOUT_CROSS", junction = "ROUNDABOUT",
      exitNumber = 2, numExits = 4).turnCode())
    // plain turns pass through the code table unchanged
    assertEquals(12, Maneuver("TURN_LEFT").turnCode())
    // slights suppressed -> carrot ATC must not brake for priority-road curves
    assertEquals(0, Maneuver("SLIGHT_RIGHT").turnCode())
  }
}
