// Common nav-source interface. v0.5 ships TomTomRestNavSource; v1.1 can add a
// TomTom Maps SDK source (guidance callbacks) implementing the same contract.
package hu.tunelab.navbridge

interface NavSource {
  fun startRoute(destLat: Double, destLon: Double)
  fun stop()
}
