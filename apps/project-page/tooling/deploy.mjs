import { spawnSync } from 'node:child_process';

// Deployment publishes committed main, never an unreviewed working directory.
// No implicit add/commit/push, credentials, or branch changes occur here.
console.log('Requesting GitHub Pages deployment of committed main. Local changes are NOT uploaded.');
const result = spawnSync('gh', ['workflow', 'run', 'deploy-pages.yml', '--repo', 'PRIS-CV/DataEvolver', '--ref', 'main'], { stdio: 'inherit' });
if (result.error) console.error('GitHub CLI is required. Install gh and authenticate, or use the Actions tab.');
process.exitCode = result.error ? 1 : result.status ?? 1;
