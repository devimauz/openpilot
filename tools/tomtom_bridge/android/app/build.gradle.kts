plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

android {
  namespace = "hu.tunelab.navbridge"
  compileSdk = 34

  defaultConfig {
    applicationId = "hu.tunelab.navbridge"
    minSdk = 26
    targetSdk = 34
    versionCode = 7
    versionName = "0.7"
    // Put these in gradle.properties / local.properties or use BuildConfig fields
    buildConfigField("String", "TOMTOM_API_KEY", "\"${project.findProperty("TomTomApiKey") ?: ""}\"")
  }

  buildFeatures { buildConfig = true }

  buildTypes {
    release {
      isMinifyEnabled = false
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  kotlinOptions { jvmTarget = "17" }
}

dependencies {
  implementation("androidx.core:core-ktx:1.13.1")
  implementation("androidx.appcompat:appcompat:1.7.0")
  implementation("com.google.android.material:material:1.12.0")
  implementation("com.squareup.okhttp3:okhttp:4.12.0")
  implementation("com.google.code.gson:gson:2.11.0")
  implementation("com.google.android.gms:play-services-location:21.3.0")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
  // --- TomTom SDK (v1.1; also uncomment repo in settings.gradle.kts) ---
  // implementation("com.tomtom.android.maplibrary:android-sdk:<pinned-version>")
  testImplementation("junit:junit:4.13.2")
}
