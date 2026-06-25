import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, BackHandler, Platform, StyleSheet, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Camera } from 'expo-camera';
import { WebView } from 'react-native-webview';

const SITE_URL = 'https://brianchen0622-a11y.github.io/building-style-id/';

export default function App() {
  const webviewRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [canGoBack, setCanGoBack] = useState(false);

  const handleAndroidPermissionRequest = useCallback((request) => {
    request.grant();
  }, []);

  useEffect(() => {
    if (Platform.OS === 'android') {
      Camera.requestCameraPermissionsAsync();
    }
  }, []);

  useEffect(() => {
    if (Platform.OS !== 'android') return;
    const onBackPress = () => {
      if (canGoBack && webviewRef.current) {
        webviewRef.current.goBack();
        return true;
      }
      return false;
    };
    const sub = BackHandler.addEventListener('hardwareBackPress', onBackPress);
    return () => sub.remove();
  }, [canGoBack]);

  return (
    <View style={styles.container}>
      <StatusBar style="dark" />
      <WebView
        ref={webviewRef}
        source={{ uri: SITE_URL }}
        style={styles.webview}
        onLoadEnd={() => setLoading(false)}
        onNavigationStateChange={(nav) => setCanGoBack(nav.canGoBack)}
        onPermissionRequest={handleAndroidPermissionRequest}
        mediaCapturePermissionGrantType="grant"
        allowsInlineMediaPlayback
        mediaPlaybackRequiresUserAction={false}
        javaScriptEnabled
        domStorageEnabled
        startInLoadingState={false}
      />
      {loading && (
        <View style={styles.loadingOverlay} pointerEvents="none">
          <ActivityIndicator size="large" color="#b5651d" />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  webview: {
    flex: 1,
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
  },
});
