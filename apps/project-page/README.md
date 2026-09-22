# DataEvolver Project Page

A self-contained public website and read-only trace API. Everything needed to
develop, build or serve the site lives in this directory. There is no dependency
on the private Harness, Python training environment, experiment roots or GPUs.

```text
apps/project-page/
├── frontend/       # Existing homepage, assets and /traces/ subpage
├── backend/        # Read-only Node HTTP API and tests
├── tooling/        # Build, isolation tests and Pages dispatch
├── deploy/         # Docker Compose, systemd, Nginx and deployment guide
├── package.json
├── package-lock.json
└── dist/           # Generated, ignored, and local to this application
```

## Run locally

Node.js 22+ and npm are required. Run from this folder, not the Harness root:

```sh
cd apps/project-page
npm ci
npm run dev
```

Open `http://127.0.0.1:4173/traces/?lang=zh`. Development serves the frontend and
the read-only API together. It does not start models or change experiments.

```sh
npm test
npm run test:evidence    # Optional locally; requires Python 3 stdlib only
npm run build
npm run preview         # Static preview; 127.0.0.1:8787
```

For a same-origin production service, build with `TRACE_API_BASE_URL=/ npm run
build` and then run `npm start`. Use `npm run start:api` for the API only.
The app can also be copied to a different directory and built without the rest
of this repository; the isolation test verifies this boundary.

## Publish the Project Page

The repository workflow is the only deployment integration outside this folder:
`.github/workflows/deploy-pages.yml`. Pull requests run tests and upload a static
build artifact but **never deploy**. A reviewed change merged into `main` publishes
`apps/project-page/dist/` automatically. `npm run deploy` manually dispatches the
already committed `main` version; it does not commit, push or merge anything.

The public URLs stay unchanged:

- <https://pris-cv.github.io/DataEvolver/>
- <https://pris-cv.github.io/DataEvolver/traces/?lang=zh>

GitHub Pages serves the frontend only. By default the page uses the bundled
public archive. The optional backend must be hosted separately with HTTPS.
See [deployment instructions](deploy/README.md) for CORS, templates and API routes.

## Isolation contract

- No imports from `src/dataevolver`, experiments or the private Harness.
- No root-level npm dependencies or generated site output.
- The Docker context is this app directory only, never the repository root.
- Backend endpoints only read the curated public archive; no training controls.
- Build and tests do not call image models, VLMs, renderers or training workers.
- Original frames and evidence stay byte-identical during build and deployment.

See [the trace evidence policy](frontend/traces/README.md) before adding examples.
