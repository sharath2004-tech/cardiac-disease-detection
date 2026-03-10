# HeartSense AI

## Project info

HeartSense AI - Heart Health Monitoring Application with AI-powered cardiac risk assessment

## Architecture

This application consists of:
- **Frontend**: React + TypeScript + Vite + shadcn/ui
- **Backend**: Node.js + Express + MongoDB Atlas
- **Database**: MongoDB Atlas (Cloud NoSQL database)

## Setup Instructions

### Prerequisites

- Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)
- MongoDB Atlas account (free tier available)

### Installation Steps

```sh
# Step 1: Clone the repository
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory
cd heartsense-ai-main

# Step 3: Install frontend dependencies
npm install

# Step 4: Install backend dependencies
npm run server:install

# Step 5: Configure MongoDB Atlas
# - Go to https://www.mongodb.com/cloud/atlas
# - Create a free cluster
# - Get your connection string
# - Update server/.env with your MONGODB_URI

# Step 6: Start the backend server (in one terminal)
npm run server

# Step 7: Start the frontend (in another terminal)
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## Backend Configuration

### MongoDB Atlas Setup

1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a new cluster (Free M0 tier is sufficient)
3. Click "Connect" → "Connect your application"
4. Copy the connection string
5. Update `server/.env`:
   ```
   MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/heartsense-ai
   JWT_SECRET=your-secure-random-string
   ```

### Environment Variables

**Frontend** (`.env`):
```
VITE_API_URL=http://localhost:5000/api
```

**Backend** (`server/.env`):
```
PORT=5000
MONGODB_URI=your-mongodb-connection-string
JWT_SECRET=your-jwt-secret
FRONTEND_URL=http://localhost:5173
```

## API Documentation

See [server/README.md](server/README.md) for detailed API documentation.

## Available Scripts

### Frontend
- `npm run dev` - Start frontend development server
- `npm run build` - Build for production
- `npm run test` - Run tests

### Backend
- `npm run server` - Start backend development server
- `npm run server:install` - Install backend dependencies

## What technologies are used for this project?

### Frontend
- React 18
- TypeScript
- Vite
- shadcn/ui components
- Tailwind CSS
- React Router
- React Hook Form
- Axios

### Backend
- Node.js
- Express.js
- MongoDB Atlas
- Mongoose
- JWT Authentication
- bcryptjs

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

You can deploy this project using any static hosting service like Vercel, Netlify, or GitHub Pages.

## Can I connect a custom domain?

Yes, you can connect a custom domain through your hosting provider's settings.
