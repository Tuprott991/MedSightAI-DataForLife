// Service Worker for aggressive image caching
const CACHE_NAME = 'medsight-images-v1';
const IMAGE_CACHE_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 days

self.addEventListener('install', (event) => {
  console.log('Service Worker: Installed');
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activated');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker: Clearing old cache', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only cache image proxy requests
  if (url.pathname.includes('/api/v1/patients/image/proxy')) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        // Try to get from cache first
        const cachedResponse = await cache.match(request);
        
        if (cachedResponse) {
          const cachedDate = cachedResponse.headers.get('date');
          const cacheAge = Date.now() - new Date(cachedDate).getTime();
          
          // Return cached if still valid
          if (cacheAge < IMAGE_CACHE_DURATION) {
            console.log('SW Cache HIT:', url.pathname);
            return cachedResponse;
          }
        }

        // Fetch from network
        try {
          console.log('SW Cache MISS:', url.pathname);
          const networkResponse = await fetch(request);
          
          // Cache the new response
          if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
          }
          
          return networkResponse;
        } catch (error) {
          console.error('SW Fetch failed:', error);
          
          // Return cached even if expired as fallback
          if (cachedResponse) {
            return cachedResponse;
          }
          
          throw error;
        }
      })
    );
  }
});
