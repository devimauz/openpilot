// NavBridge — phone-side TomTom navigation feed for comma/carrot (carrot_navi v2 protocol)
pluginManagement {
  repositories {
    google()
    mavenCentral()
    gradlePluginPortal()
  }
}
dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
    // --- TomTom Maps SDK (v1.1 path; uncomment with the version you onboarded to) ---
    // maven { url = uri("https://repositories.tomtom.com/artifactory/maven") }
  }
}
rootProject.name = "navbridge"
include(":app")
