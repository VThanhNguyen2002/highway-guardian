// frontend/src/stores/authStore.ts

import { defineStore } from 'pinia';
import { ref } from 'vue';
// Đường dẫn này giờ đã đúng (config.ts)
import { auth, db } from '../firebase/config'; 
// SỬA 1: Dùng 'import type' cho các kiểu từ Firebase
import { signInWithEmailAndPassword, signOut, onAuthStateChanged } from 'firebase/auth';
import type { User as FirebaseUser } from 'firebase/auth'; // Import kiểu User
import { doc, getDoc } from 'firebase/firestore';
import type { DocumentData } from 'firebase/firestore'; // Import kiểu DocumentData

// Định nghĩa interface UserProfile (giữ nguyên)
interface UserProfile {
  uid: string;
  email: string;
  displayName: string;
}

export const useAuthStore = defineStore('auth', () => {
  // Gán kiểu cho các ref (giữ nguyên)
  const user = ref<UserProfile | null>(null);
  const isLoggedIn = ref<boolean>(false);
  const loading = ref<boolean>(false);

  // Gán kiểu cho hàm login (giữ nguyên)
  async function login(email: string, password: string): Promise<true | string> {
    loading.value = true;
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      // Gán kiểu FirebaseUser rõ ràng
      const authUser: FirebaseUser = userCredential.user; 
      
      const userDocRef = doc(db, 'users', authUser.uid);
      const userDoc = await getDoc(userDocRef);

      if (userDoc.exists()) {
        // Gán kiểu DocumentData rõ ràng
        const userData: DocumentData = userDoc.data(); 
        user.value = {
          uid: authUser.uid,
          email: authUser.email!, 
          displayName: userData.displayName || 'Người dùng' 
        };
        isLoggedIn.value = true;
        return true; 
      } else {
        // Nếu user tồn tại trong Auth nhưng không có doc trong Firestore
        // Có thể logout user ở đây hoặc hiển thị lỗi cụ thể hơn
        await signOut(auth); // Ví dụ: tự động logout
        user.value = null;
        isLoggedIn.value = false;
        console.error("User document not found in Firestore for UID:", authUser.uid);
        return 'auth/user-doc-not-found'; // Trả về mã lỗi tùy chỉnh
        // throw new Error("Không tìm thấy thông tin user trong Firestore.");
      }
    } catch (error: any) { 
      console.error("Lỗi đăng nhập:", error);
      return error.code || 'UNKNOWN_ERROR'; 
    } finally {
      loading.value = false;
    }
  }

  // Gán kiểu cho hàm logout (giữ nguyên)
  async function logout(): Promise<void> {
    await signOut(auth);
    user.value = null;
    isLoggedIn.value = false;
  }
  
  // Gán kiểu cho hàm checkAuth (giữ nguyên)
  function checkAuth(): Promise<void> {
    return new Promise((resolve) => {
      const unsubscribe = onAuthStateChanged(auth, async (authUser) => {
        if (authUser) {
          // Chỉ fetch Firestore nếu user hiện tại là null (tránh fetch lặp lại)
          if (user.value === null) { 
            const userDocRef = doc(db, 'users', authUser.uid);
            const userDoc = await getDoc(userDocRef);
            if (userDoc.exists()) {
              const userData: DocumentData = userDoc.data();
              user.value = {
                uid: authUser.uid,
                email: authUser.email!,
                displayName: userData.displayName || 'Người dùng'
              };
              isLoggedIn.value = true;
            } else {
               // User tồn tại trong Auth nhưng không có trong Firestore khi checkAuth
               // Tự động logout
               console.error("User document not found during auth check for UID:", authUser.uid);
               await signOut(auth);
               user.value = null;
               isLoggedIn.value = false;
            }
          }
        } else {
          user.value = null;
          isLoggedIn.value = false;
        }
        resolve(); 
        unsubscribe();
      });
    });
  }

  return { user, isLoggedIn, loading, login, logout, checkAuth };
});