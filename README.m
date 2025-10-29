# Product Scanner — GitHub + Render deployment


This repo contains a static front-end (docs/index.html) for GitHub Pages and a Node/Express backend (server.js) to run Vision/OpenAI/search APIs.


## Quick setup (local)
1. Copy files from this document into a folder `product-scanner/`.
2. Run `npm install`.
3. Create a `.env` file with the following keys (DO NOT commit this file): OPENAI_API_KEY=sk-... GOOGLE_CREDENTIALS={... full JSON ...} EBAY_APP_ID=... CORS_ORIGIN=https://YOUR_USERNAME.github.io
4. 4. Start locally: `node server.js`.
5. Serve the front-end locally: `npx http-server docs -p 8080` and open `http://localhost:8080`.


## Deploy front-end: GitHub Pages
1. Commit and push repository to GitHub.
2. In GitHub repo -> Settings -> Pages -> Source select `main` branch and `/docs` folder. Save.
3. After a minute your site will be at `https://YOUR_USERNAME.github.io/REPO/`.


## Deploy backend: Render (example)
1. Create an account at https://render.com and connect GitHub.
2. Create a **Web Service** and pick this repo + `main` branch.
3. Build command: `npm install`
Start command: `node server.js`
4. Add environment variables under the Render dashboard: `OPENAI_API_KEY`, `GOOGLE_CREDENTIALS` (paste JSON), `EBAY_APP_ID`, `CORS_ORIGIN`.
5. Deploy and copy the Render URL.
6. Upda