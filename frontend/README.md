# Frontend — Vue 3 Dashboard

Real-time Traffic Infrastructure Analytics Dashboard built with **Vue 3 + Vite**.

---

## Tech Stack

| Package | Role |
|---------|------|
| Vue 3 (Composition API) | UI framework |
| Vite | Build tool & dev server |
| Vue Router 4 | Client-side routing |
| Pinia | State management (auth store) |
| Chart.js + vue-chartjs | KPI charts & distributions |
| Firebase JS SDK (v10) | Firestore listener, Auth |
| vue-feather | Icon set |

## Project Structure

```
frontend/
├── src/
│   ├── components/         # Reusable UI blocks
│   │   ├── AnalyticsCharts.vue   # Firestore listener + orchestrator
│   │   ├── StatCards.vue         # KPI cards
│   │   ├── DistributionCharts.vue # Doughnut + bar charts
│   │   ├── TimeSeriesCharts.vue  # Line chart (detections over time)
│   │   ├── DetectionTable.vue    # CRUD detection log
│   │   ├── RealtimeFeed.vue      # Live detection feed
│   │   ├── Sidebar.vue
│   │   └── Header.vue
│   ├── views/              # Route-level pages
│   │   ├── Dashboard.vue
│   │   ├── Login.vue
│   │   └── Profile.vue
│   ├── stores/             # Pinia stores
│   │   └── authStore.js
│   ├── firebase/
│   │   └── config.js       # Firebase SDK init (reads VITE_FIREBASE_*)
│   ├── router/index.js
│   ├── App.vue
│   └── main.js
├── index.html
├── vite.config.js
└── package.json
```

## Setup

```bash
# 1. Install dependencies
npm install

# 2. Configure Firebase
cp .env.example .env.local
# Then fill in your project values (see below)

# 3. Start dev server
npm run dev        # → http://localhost:5173

# 4. Production build
npm run build
```

## Environment Variables

> ⚠️ **Required — never commit `.env.local` to Git.**

Create `frontend/.env.local` with your Firebase project credentials:

```env
VITE_FIREBASE_API_KEY=AIza...
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=1:123456789:web:abc123
```

Find these values in the Firebase Console → Project Settings → Your apps → Web app config.

## Data Flow

```
Firestore 'detections' collection
        │  onSnapshot() listener
        ▼
AnalyticsCharts.vue (allDocs ref)
        │  props :docs
        ├──▶ StatCards.vue        (KPIs)
        ├──▶ DistributionCharts.vue (pie / bar)
        ├──▶ TimeSeriesCharts.vue   (line)
        └──▶ DetectionTable.vue    (CRUD log)
```
