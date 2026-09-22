FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts
COPY web ./web
COPY tooling/project-page ./tooling/project-page
ARG TRACE_API_BASE_URL=/
ENV TRACE_API_BASE_URL=$TRACE_API_BASE_URL
RUN npm run build

FROM node:24-alpine
ENV NODE_ENV=production HOST=0.0.0.0 PORT=8787 TRACE_DATA_DIR=/app/dist/traces
WORKDIR /app
COPY --from=build --chown=node:node /app/dist ./dist
COPY --chown=node:node package.json ./
COPY --chown=node:node services/trace-api/server.mjs ./services/trace-api/server.mjs
COPY --chown=node:node web/traces/data-source.js ./web/traces/data-source.js
USER node
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD node -e "fetch('http://127.0.0.1:8787/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "services/trace-api/server.mjs"]
