FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Vite inlines VITE_* from the build env (set VITE_API_URL as a Railway build variable)
RUN npm run build

FROM node:22-alpine

WORKDIR /app

RUN npm install -g serve@14.2.6

COPY --from=build /app/dist ./dist

ENV PORT=3000
EXPOSE 3000

CMD ["sh", "-c", "serve -s dist -l tcp://0.0.0.0:${PORT}"]
