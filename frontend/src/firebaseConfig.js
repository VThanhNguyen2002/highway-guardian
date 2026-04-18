/**
 * frontend/src/firebaseConfig.js
 *
 * Re-export of the Firebase services initialised in ./firebase/config.ts.
 * Provides the `auth` and `db` singletons at this path for backwards
 * compatibility with imports that reference `firebaseConfig.js`.
 */
export { auth, db } from './firebase/config';
