# Frontend Changes

## Testing Feature

This feature (enhancing the testing framework) did not require any changes to the frontend. All modifications were in the backend test infrastructure:

- `pyproject.toml` — Added `[tool.pytest.ini_options]` config and `httpx` dev dependency
- `backend/tests/conftest.py` — Added API test fixtures (mock RAGSystem, test FastAPI app, TestClient)
- `backend/tests/test_api.py` — New file: 27 API endpoint tests for `/api/query`, `/api/courses`, `/api/session/clear`

## Dark/Light Theme Toggle

### Files Modified

#### `frontend/style.css`
- **Added light theme CSS variables** under `[data-theme="light"]` selector with appropriate colors: light backgrounds (#f8fafc, #ffffff), dark text (#0f172a), adjusted borders (#e2e8f0), and proper contrast ratios
- **Added new CSS variables** to `:root` for theme-dependent values that were previously hardcoded: `--code-bg`, `--sources-bg`, `--sources-summary-hover`, `--source-link-bg`, `--source-link-border`, `--source-link-hover-bg`, `--source-link-hover-border`, `--source-text-bg`, `--error-bg`, `--error-text`, `--error-border`, `--success-bg`, `--success-text`, `--success-border`
- **Replaced hardcoded rgba values** in `.sources-collapsible`, `.sources-collapsible summary:hover`, `.source-link`, `.source-link:hover`, `.source-text`, `.message-content code`, `.message-content pre`, `.error-message`, and `.success-message` with the corresponding CSS variables
- **Added `.theme-toggle` button styles**: fixed position top-right, circular 40px button with border, background, hover rotation effect, focus ring for accessibility
- **Added icon transition animations**: `.icon-sun` and `.icon-moon` with opacity and rotation transitions so they smoothly swap when toggling themes
- **Added `transition: background-color 0.3s ease, color 0.3s ease`** to `body` and `.message.assistant .message-content` for smooth theme switching

#### `frontend/index.html`
- **Added theme toggle button** immediately after `<body>` with:
  - `id="themeToggle"` for JavaScript binding
  - `aria-label="Toggle theme"` and `title` for accessibility
  - Sun icon SVG (`.icon-sun`) — visible in light mode
  - Moon icon SVG (`.icon-moon`) — visible in dark mode

#### `frontend/script.js`
- **Added `initTheme()` function**: reads `theme` from `localStorage` and sets `data-theme="light"` on `<html>` if saved, otherwise defaults to dark
- **Added `toggleTheme()` function**: toggles `data-theme` attribute on `<html>` between `"light"` and no attribute (dark), persists choice to `localStorage`
- **Wired up the toggle button**: added `initTheme()` call and click event listener on `#themeToggle` in the `DOMContentLoaded` handler

### Design Decisions
- Dark theme is the default (matching existing design) — light theme is opt-in
- Theme preference persists across sessions via `localStorage`
- `data-theme` is set on `<html>` element so CSS variables cascade to all children
- Smooth 0.3s transitions on background/color changes for polished feel
- Toggle button uses fixed positioning (top-right) so it's always accessible
- Sun/moon icons rotate and fade in/out during toggle for visual feedback
