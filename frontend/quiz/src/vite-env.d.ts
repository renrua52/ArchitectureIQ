/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TELEMETRY_URL?: string;
  readonly VITE_TELEMETRY_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
