import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import CollectionDetailPage from "./pages/CollectionDetailPage";
import CollectionsPage from "./pages/CollectionsPage";

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>VectorDB Explorer</h1>
        <nav>
          <NavLink to="/collections">Collections</NavLink>
        </nav>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Navigate to="/collections" replace />} />
          <Route path="/collections" element={<CollectionsPage />} />
          <Route path="/collections/:collectionId" element={<CollectionDetailPage />} />
        </Routes>
      </main>
    </div>
  );
}
